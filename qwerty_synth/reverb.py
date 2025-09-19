"""Improved algorithmic reverb for QWERTY Synth.

The original reverb relied on a lightweight Schroeder-style network. While
cheap to run, it produced an uneven decay, audible zipper noise when the mix
parameter changed abruptly, and an aggressive loss of level at higher mix
values.  This rewrite keeps the familiar interface but rebuilds the signal flow
around a more balanced feedback delay network with:

* Early reflection taps for immediate spatial cues.
* Stereo decorrelated feedback comb filters to create the late tail.
* A chain of all-pass diffusers for smooth decay.
* Constant-power wet/dry crossfade with gentle parameter smoothing to remove
  clicks when notes are released or when the mix knob is moved.
"""

from __future__ import annotations

import numpy as np
from numba import jit

from qwerty_synth import config


# Early reflection taps (in samples @ 44.1kHz). These provide fast transient
# cues so the effect is audible even on the very first processing block.
EARLY_DELAYS = [45, 67, 89, 131]
EARLY_STEREO_OFFSET = 11

# Feedback comb network (5-18ms) for the late reverb tail. The right channel
# receives a prime offset to help decorrelate the stereo field.
COMB_DELAYS = [233, 307, 389, 461, 499, 571, 653, 733]
COMB_STEREO_OFFSET = 37

# All-pass diffusers (1-6ms) to blur individual echoes.
ALLPASS_DELAYS = [79, 113, 149, 193]
ALLPASS_STEREO_OFFSET = 17

# Scaling constants tuned by ear.
EARLY_GAIN = 0.4
COMB_GAIN = 1.0 / len(COMB_DELAYS)
MIX_SMOOTH_COEFF = 0.15  # Single-pole smoothing for mix parameter


@jit(nopython=True, cache=True, fastmath=True)
def _process_delay_line(input_samples, buffer, write_idx, delay_samples):
    """Simple tapped delay used for early reflections."""
    output = np.empty_like(input_samples)
    buffer_len = len(buffer)

    for i in range(len(input_samples)):
        read_idx = write_idx - delay_samples
        if read_idx < 0:
            read_idx += buffer_len

        output[i] = buffer[read_idx]
        buffer[write_idx] = input_samples[i]

        write_idx += 1
        if write_idx >= buffer_len:
            write_idx = 0

    return output, write_idx


@jit(nopython=True, cache=True, fastmath=True)
def _process_comb_filter(input_samples, buffer, write_idx, delay_samples,
                        feedback, damping, filter_state):
    """Low-pass feedback comb filter used in the late reverb network."""
    output = np.empty_like(input_samples)
    buffer_len = len(buffer)

    for i in range(len(input_samples)):
        read_idx = write_idx - delay_samples
        if read_idx < 0:
            read_idx += buffer_len

        delayed = buffer[read_idx]
        filter_state = delayed + damping * (filter_state - delayed)
        buffer[write_idx] = input_samples[i] + filter_state * feedback
        output[i] = delayed

        write_idx += 1
        if write_idx >= buffer_len:
            write_idx = 0

    return output, write_idx, filter_state


@jit(nopython=True, cache=True, fastmath=True)
def _process_allpass_filter(input_samples, buffer, write_idx, delay_samples,
                           feedback):
    """First-order all-pass filter for diffusion."""
    output = np.empty_like(input_samples)
    buffer_len = len(buffer)

    for i in range(len(input_samples)):
        read_idx = write_idx - delay_samples
        if read_idx < 0:
            read_idx += buffer_len

        delayed = buffer[read_idx]
        sample = input_samples[i]

        output_sample = delayed - feedback * sample
        buffer[write_idx] = sample + feedback * output_sample
        output[i] = output_sample

        write_idx += 1
        if write_idx >= buffer_len:
            write_idx = 0

    return output, write_idx


class Reverb:
    """Stereo algorithmic reverb effect."""

    def __init__(self, sample_rate: int | None = None) -> None:
        self.sample_rate = sample_rate if sample_rate is not None else config.sample_rate

        # Delay line containers
        self.early_buffers_L: list[np.ndarray] = []
        self.early_buffers_R: list[np.ndarray] = []
        self.early_indices_L: list[int] = []
        self.early_indices_R: list[int] = []
        self.early_delays_L: list[int] = []
        self.early_delays_R: list[int] = []

        self.comb_buffers_L: list[np.ndarray] = []
        self.comb_buffers_R: list[np.ndarray] = []
        self.comb_indices_L: list[int] = []
        self.comb_indices_R: list[int] = []
        self.comb_filter_states_L: list[float] = []
        self.comb_filter_states_R: list[float] = []
        self.comb_delays_L: list[int] = []
        self.comb_delays_R: list[int] = []

        self.allpass_buffers_L: list[np.ndarray] = []
        self.allpass_buffers_R: list[np.ndarray] = []
        self.allpass_indices_L: list[int] = []
        self.allpass_indices_R: list[int] = []
        self.allpass_delays_L: list[int] = []
        self.allpass_delays_R: list[int] = []

        # Cached parameters
        self.feedback = 0.0
        self.damping = 0.0
        self.allpass_feedback = 0.5
        self.mix_target = getattr(config, "reverb_mix", 0.25)
        self.mix_state = float(self.mix_target)
        self.mix = float(self.mix_target)
        self.dry_gain = 1.0
        self.wet_gain = 0.0

        self._initialise_buffers()
        self._update_cached_values()

    # ------------------------------------------------------------------
    # Buffer setup
    # ------------------------------------------------------------------
    def _initialise_buffers(self) -> None:
        scale = self.sample_rate / 44_100.0

        def scaled(values, offset=0):
            return [max(1, int(round((value + offset) * scale))) for value in values]

        self.early_delays_L = scaled(EARLY_DELAYS)
        self.early_delays_R = scaled(EARLY_DELAYS, EARLY_STEREO_OFFSET)
        self.comb_delays_L = scaled(COMB_DELAYS)
        self.comb_delays_R = scaled(COMB_DELAYS, COMB_STEREO_OFFSET)
        self.allpass_delays_L = scaled(ALLPASS_DELAYS)
        self.allpass_delays_R = scaled(ALLPASS_DELAYS, ALLPASS_STEREO_OFFSET)

        for delay in self.early_delays_L:
            self.early_buffers_L.append(np.zeros(delay + 1, dtype=np.float32))
            self.early_indices_L.append(0)

        for delay in self.early_delays_R:
            self.early_buffers_R.append(np.zeros(delay + 1, dtype=np.float32))
            self.early_indices_R.append(0)

        for delay in self.comb_delays_L:
            self.comb_buffers_L.append(np.zeros(delay + 1, dtype=np.float32))
            self.comb_indices_L.append(0)
            self.comb_filter_states_L.append(0.0)

        for delay in self.comb_delays_R:
            self.comb_buffers_R.append(np.zeros(delay + 1, dtype=np.float32))
            self.comb_indices_R.append(0)
            self.comb_filter_states_R.append(0.0)

        for delay in self.allpass_delays_L:
            self.allpass_buffers_L.append(np.zeros(delay + 1, dtype=np.float32))
            self.allpass_indices_L.append(0)

        for delay in self.allpass_delays_R:
            self.allpass_buffers_R.append(np.zeros(delay + 1, dtype=np.float32))
            self.allpass_indices_R.append(0)

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def _update_cached_values(self) -> None:
        room_size = float(np.clip(getattr(config, "reverb_room_size", 0.5), 0.0, 1.0))
        damping = float(np.clip(getattr(config, "reverb_damping", 0.5), 0.0, 1.0))
        mix = float(np.clip(getattr(config, "reverb_mix", 0.25), 0.0, 1.0))

        # Map parameters into useful ranges.
        self.feedback = 0.45 + room_size * 0.48  # 0.45 .. 0.93
        self.feedback = min(self.feedback, 0.97)

        self.damping = 0.05 + damping * 0.85  # 0.05 .. 0.90
        self.mix_target = mix
        self.mix = mix

    def set_room_size(self, room_size: float) -> None:
        config.reverb_room_size = max(0.0, min(1.0, room_size))
        self._update_cached_values()

    def set_damping(self, damping: float) -> None:
        config.reverb_damping = max(0.0, min(1.0, damping))
        self._update_cached_values()

    def set_mix(self, mix: float) -> None:
        config.reverb_mix = max(0.0, min(1.0, mix))
        self._update_cached_values()

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------
    def process(self, L: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if L.size == 0 or R.size == 0:
            return L, R

        self._update_cached_values()

        if self.mix_target <= 1e-5:
            self.mix_state = 0.0
            return L, R

        # Smooth mix transitions to avoid zipper noise when controls move.
        self.mix_state += (self.mix_target - self.mix_state) * MIX_SMOOTH_COEFF
        self.mix_state = float(np.clip(self.mix_state, 0.0, 1.0))

        if self.mix_state <= 1e-5:
            return L, R

        # Constant-power crossfade retains perceived loudness at high mix
        # settings and reduces the step when the dry path drops out.
        self.dry_gain = float(np.sqrt(max(0.0, 1.0 - self.mix_state)))
        self.wet_gain = float(np.sqrt(self.mix_state))

        mono_input = (L + R) * 0.5

        early_L = np.zeros_like(mono_input)
        early_R = np.zeros_like(mono_input)

        for i in range(len(self.early_buffers_L)):
            buffer = self.early_buffers_L[i]
            out, idx = _process_delay_line(
                mono_input, buffer, self.early_indices_L[i], self.early_delays_L[i]
            )
            self.early_indices_L[i] = idx
            early_L += out

        for i in range(len(self.early_buffers_R)):
            buffer = self.early_buffers_R[i]
            out, idx = _process_delay_line(
                mono_input, buffer, self.early_indices_R[i], self.early_delays_R[i]
            )
            self.early_indices_R[i] = idx
            early_R += out

        comb_L = np.zeros_like(mono_input)
        comb_R = np.zeros_like(mono_input)

        for i in range(len(self.comb_buffers_L)):
            buffer_L = self.comb_buffers_L[i]
            buffer_R = self.comb_buffers_R[i]

            out_L, idx_L, state_L = _process_comb_filter(
                mono_input, buffer_L, self.comb_indices_L[i],
                self.comb_delays_L[i], self.feedback, self.damping,
                self.comb_filter_states_L[i]
            )
            out_R, idx_R, state_R = _process_comb_filter(
                mono_input, buffer_R, self.comb_indices_R[i],
                self.comb_delays_R[i], self.feedback, self.damping,
                self.comb_filter_states_R[i]
            )

            self.comb_indices_L[i] = idx_L
            self.comb_indices_R[i] = idx_R
            self.comb_filter_states_L[i] = state_L
            self.comb_filter_states_R[i] = state_R

            comb_L += out_L
            comb_R += out_R

        comb_L *= COMB_GAIN
        comb_R *= COMB_GAIN

        diffused_L = comb_L
        diffused_R = comb_R

        for i in range(len(self.allpass_buffers_L)):
            buffer_L = self.allpass_buffers_L[i]
            buffer_R = self.allpass_buffers_R[i]

            diffused_L, idx_L = _process_allpass_filter(
                diffused_L, buffer_L, self.allpass_indices_L[i],
                self.allpass_delays_L[i], self.allpass_feedback
            )
            diffused_R, idx_R = _process_allpass_filter(
                diffused_R, buffer_R, self.allpass_indices_R[i],
                self.allpass_delays_R[i], self.allpass_feedback
            )

            self.allpass_indices_L[i] = idx_L
            self.allpass_indices_R[i] = idx_R

        wet_L = EARLY_GAIN * early_L + diffused_L
        wet_R = EARLY_GAIN * early_R + diffused_R

        # Small stereo width injection: crossfeed a portion of the opposite
        # channel to widen the image while keeping mono compatibility.
        stereo_wet_L = wet_L * 0.85 + wet_R * 0.15
        stereo_wet_R = wet_R * 0.85 + wet_L * 0.15

        out_L = L * self.dry_gain + stereo_wet_L * self.wet_gain
        out_R = R * self.dry_gain + stereo_wet_R * self.wet_gain

        return out_L.astype(np.float32), out_R.astype(np.float32)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        for buffer in self.early_buffers_L + self.early_buffers_R:
            buffer.fill(0.0)

        for buffer in self.comb_buffers_L + self.comb_buffers_R:
            buffer.fill(0.0)

        for buffer in self.allpass_buffers_L + self.allpass_buffers_R:
            buffer.fill(0.0)

        for i in range(len(self.early_indices_L)):
            self.early_indices_L[i] = 0

        for i in range(len(self.early_indices_R)):
            self.early_indices_R[i] = 0

        for i in range(len(self.comb_indices_L)):
            self.comb_indices_L[i] = 0
            self.comb_filter_states_L[i] = 0.0

        for i in range(len(self.comb_indices_R)):
            self.comb_indices_R[i] = 0
            self.comb_filter_states_R[i] = 0.0

        for i in range(len(self.allpass_indices_L)):
            self.allpass_indices_L[i] = 0

        for i in range(len(self.allpass_indices_R)):
            self.allpass_indices_R[i] = 0
