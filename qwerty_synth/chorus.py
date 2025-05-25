"""Chorus effect module for QWERTY Synth.

A chorus effect creates a richer sound by mixing the input signal with
slightly delayed copies that have a time-varying delay, creating a
sense of multiple voices playing in unison.
"""

import numpy as np
from numba import jit
from qwerty_synth import config


@jit(nopython=True, cache=True, fastmath=True)
def _process_chorus_single_voice_jit(L, R, out_L, out_R, buffer_L, buffer_R,
                                    phase_values, lfo_values, delay_samples,
                                    base_delay_samples, write_idx, mask,
                                    dry_gain, wet_gain):
    """JIT-compiled single voice chorus processing."""
    buffer_len = len(L)

    for i in range(buffer_len):
        # Store the current sample in the buffer
        buffer_L[write_idx] = L[i]
        buffer_R[write_idx] = R[i]

        # Fast delay calculation with optimized math operations
        delay = delay_samples[i]
        read_pos = write_idx - delay
        read_idx = int(read_pos) & mask  # Faster than np.floor + modulo
        frac = read_pos - int(read_pos)  # Faster than separate floor operation
        next_idx = (read_idx + 1) & mask

        # Optimized linear interpolation with cached calculations
        inv_frac = 1.0 - frac
        sample_L = inv_frac * buffer_L[read_idx] + frac * buffer_L[next_idx]
        sample_R = inv_frac * buffer_R[read_idx] + frac * buffer_R[next_idx]

        # Direct mixing with pre-computed gains
        out_L[i] = L[i] * dry_gain + sample_L * wet_gain
        out_R[i] = R[i] * dry_gain + sample_R * wet_gain

        # Update write index
        write_idx = (write_idx + 1) & mask

    return write_idx


@jit(nopython=True, cache=True, fastmath=True)
def _process_chorus_multi_voice_jit(L, R, out_L, out_R, buffer_L, buffer_R,
                                   phases, base_delay_samples, depth_samples,
                                   phase_inc, write_idx, mask, dry_gain, voice_mix):
    """JIT-compiled multi-voice chorus processing."""
    buffer_len = len(L)
    voices = len(phases)

    # Add dry signal once (vectorized operation)
    for i in range(buffer_len):
        out_L[i] = L[i] * dry_gain
        out_R[i] = R[i] * dry_gain

    # Process each sample
    for i in range(buffer_len):
        # Store the current sample in the buffer
        buffer_L[write_idx] = L[i]
        buffer_R[write_idx] = R[i]

        # Process each chorus voice
        for v in range(voices):
            # Calculate LFO output for this voice (cached phase increment)
            lfo = np.sin(phases[v])

            # Calculate delay samples for this voice
            delay_samples = base_delay_samples + lfo * depth_samples

            # Optimized fractional sample index calculation
            read_pos = write_idx - delay_samples
            read_idx = int(read_pos) & mask
            frac = read_pos - int(read_pos)
            next_idx = (read_idx + 1) & mask

            # Optimized linear interpolation
            inv_frac = 1.0 - frac
            sample_L = inv_frac * buffer_L[read_idx] + frac * buffer_L[next_idx]
            sample_R = inv_frac * buffer_R[read_idx] + frac * buffer_R[next_idx]

            # Add to output with pre-computed voice mix level
            out_L[i] += sample_L * voice_mix
            out_R[i] += sample_R * voice_mix

            # Update phase for this voice (cached phase increment)
            phases[v] = (phases[v] + phase_inc) % (2 * np.pi)

        # Update write index
        write_idx = (write_idx + 1) & mask

    return write_idx


class Chorus:
    """Chorus effect processor.

    The chorus effect creates multiple delayed versions of the input signal,
    with the delay times modulated by sine waves. This creates the illusion
    of multiple instruments playing the same part.
    """

    def __init__(self, sample_rate=None):
        """Initialize chorus effect with default settings.

        Args:
            sample_rate: Sample rate in Hz (defaults to config.sample_rate)
        """
        self.sample_rate = sample_rate if sample_rate is not None else config.sample_rate
        self.rate = config.chorus_rate
        self.depth = config.chorus_depth
        self.mix = config.chorus_mix
        self.voices = config.chorus_voices

        # LFO phase for each voice - for single voice, this is just one value
        self.phase = 0.0
        if self.voices > 1:
            self.phases = np.linspace(0, 2*np.pi, self.voices, endpoint=False)
        else:
            self.phases = np.array([0.0])  # Single phase for efficiency

        # Calculate maximum delay needed based on depth
        max_delay_samples = int(self.depth * 2 * self.sample_rate)
        self._buf_len = 1
        while self._buf_len < max_delay_samples:
            self._buf_len *= 2

        self._mask = self._buf_len - 1

        # Delay buffers for each channel
        self._buffer_L = np.zeros(self._buf_len, dtype=np.float32)
        self._buffer_R = np.zeros(self._buf_len, dtype=np.float32)
        self._write_idx = 0

        # Pre-compute base delay samples (center of modulation)
        self.base_delay_samples = int(0.015 * self.sample_rate)  # 15ms base delay

        # Pre-allocated temporary arrays for performance
        self._temp_out_L = None
        self._temp_out_R = None
        self._temp_phase_values = None
        self._temp_lfo_values = None
        self._temp_delay_samples = None
        self._max_block_size = 1024  # Initial size, will grow as needed

        # Pre-computed constants for performance optimization
        self._update_cached_values()

    def _update_cached_values(self):
        """Update cached values when parameters change."""
        # Cache frequently used calculations
        self._phase_inc = 2 * np.pi * self.rate / self.sample_rate
        self._depth_samples = self.depth * self.sample_rate
        self._dry_gain = 1.0 - self.mix
        self._wet_gain = self.mix
        self._voice_mix = self.mix / max(self.voices, 1)

    def _ensure_temp_arrays(self, block_size):
        """Ensure temporary arrays are large enough for the block size."""
        if (self._temp_out_L is None or
            len(self._temp_out_L) < block_size):
            self._max_block_size = max(self._max_block_size, block_size)
            self._temp_out_L = np.empty(self._max_block_size, dtype=np.float32)
            self._temp_out_R = np.empty(self._max_block_size, dtype=np.float32)
            self._temp_phase_values = np.empty(self._max_block_size, dtype=np.float32)
            self._temp_lfo_values = np.empty(self._max_block_size, dtype=np.float32)
            self._temp_delay_samples = np.empty(self._max_block_size, dtype=np.float32)

    def _resize_buffer(self):
        """Resize internal buffers based on current depth setting."""
        max_delay_samples = int(self.depth * 2 * self.sample_rate)
        if max_delay_samples <= self._buf_len:
            return  # Buffer is already large enough

        # Resize to next power of 2
        new_size = 1
        while new_size < max_delay_samples:
            new_size *= 2

        old_buf_len = self._buf_len
        self._buf_len = new_size
        self._mask = new_size - 1

        # Create new buffers and preserve old data efficiently
        new_buffer_L = np.zeros(new_size, dtype=np.float32)
        new_buffer_R = np.zeros(new_size, dtype=np.float32)

        # Copy existing data using numpy's efficient array operations
        if hasattr(self, '_buffer_L') and self._buffer_L is not None:
            copy_len = min(old_buf_len, new_size)
            new_buffer_L[:copy_len] = self._buffer_L[:copy_len]
            new_buffer_R[:copy_len] = self._buffer_R[:copy_len]

        self._buffer_L = new_buffer_L
        self._buffer_R = new_buffer_R

    def set_rate(self, rate):
        """Set the rate of the chorus modulation in Hz."""
        self.rate = max(0.1, min(10.0, rate))  # Clamp between 0.1 and 10 Hz
        self._update_cached_values()

    def set_depth(self, depth):
        """Set the depth of the chorus modulation in seconds."""
        depth = max(0.001, min(0.030, depth))  # Clamp between 1ms and 30ms
        self.depth = depth
        self._resize_buffer()
        self._update_cached_values()

    def set_mix(self, mix):
        """Set the dry/wet mix ratio (0.0 = dry, 1.0 = wet)."""
        self.mix = max(0.0, min(1.0, mix))
        self._update_cached_values()

    def set_voices(self, voices):
        """Set the number of chorus voices."""
        voices = int(max(1, min(4, voices)))  # Clamp between 1 and 4 voices
        if voices != self.voices:
            self.voices = voices
            # Reset phases when voice count changes
            if voices > 1:
                self.phases = np.linspace(0, 2*np.pi, self.voices, endpoint=False)
            else:
                self.phases = np.array([0.0])  # Single phase for efficiency
                self.phase = 0.0
            self._update_cached_values()

    def process(self, L, R):
        """Process stereo audio with chorus effect.

        Args:
            L: Left channel audio array
            R: Right channel audio array

        Returns:
            out_L, out_R: Processed left and right channels
        """
        # Fast bypass for disabled effect (most efficient case)
        if self.mix <= 0.0:
            return L, R

        # Optimize for single voice case (most common)
        if self.voices == 1:
            return self._process_single_voice(L, R)
        else:
            return self._process_multi_voice(L, R)

    def _process_single_voice(self, L, R):
        """Optimized processing for a single chorus voice."""
        buffer_len = len(L)
        self._ensure_temp_arrays(buffer_len)

        # Use pre-allocated slices of temporary arrays
        out_L = self._temp_out_L[:buffer_len]
        out_R = self._temp_out_R[:buffer_len]
        phase_values = self._temp_phase_values[:buffer_len]
        lfo_values = self._temp_lfo_values[:buffer_len]
        delay_samples = self._temp_delay_samples[:buffer_len]

        # Pre-compute all phase values for the entire buffer at once
        phase_values[:] = (self.phase + self._phase_inc * np.arange(buffer_len)) % (2 * np.pi)

        # Calculate all LFO values at once using vectorized operations
        np.sin(phase_values, out=lfo_values)

        # Calculate all delay values
        delay_samples[:] = self.base_delay_samples + lfo_values * self._depth_samples

        # JIT-compiled processing
        self._write_idx = _process_chorus_single_voice_jit(
            L, R, out_L, out_R, self._buffer_L, self._buffer_R,
            phase_values, lfo_values, delay_samples, self.base_delay_samples,
            self._write_idx, self._mask, self._dry_gain, self._wet_gain)

        # Update the phase for next buffer (handle empty arrays)
        if buffer_len > 0:
            self.phase = phase_values[-1]

        return out_L.copy(), out_R.copy()

    def _process_multi_voice(self, L, R):
        """Optimized processing with multiple chorus voices."""
        buffer_len = len(L)
        self._ensure_temp_arrays(buffer_len)

        # Use pre-allocated slices of temporary arrays
        out_L = self._temp_out_L[:buffer_len]
        out_R = self._temp_out_R[:buffer_len]

        # JIT-compiled processing
        self._write_idx = _process_chorus_multi_voice_jit(
            L, R, out_L, out_R, self._buffer_L, self._buffer_R,
            self.phases, self.base_delay_samples, self._depth_samples,
            self._phase_inc, self._write_idx, self._mask,
            self._dry_gain, self._voice_mix)

        return out_L.copy(), out_R.copy()

    def clear_cache(self):
        """Clear the chorus buffers and reset indices efficiently."""
        # Use numpy's optimized fill operations
        self._buffer_L.fill(0.0)
        self._buffer_R.fill(0.0)
        self._write_idx = 0
        self.phase = 0.0
        if len(self.phases) > 0:
            self.phases.fill(0.0)
