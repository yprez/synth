"""Chorus effect module for QWERTY Synth.

A chorus effect creates a richer sound by mixing the input signal with
slightly delayed copies that have a time-varying delay, creating a
sense of multiple voices playing in unison.
"""

import numpy as np
from qwerty_synth.config import sample_rate

# Default chorus parameters
DEFAULT_RATE = 0.5  # Hz - speed of modulation
DEFAULT_DEPTH = 0.007  # Seconds - depth of modulation (7ms is typical)
DEFAULT_MIX = 0.5  # Dry/wet mix
DEFAULT_VOICES = 1  # Number of chorus voices (reduced to 1 for CPU efficiency)


class Chorus:
    """Chorus effect processor.

    The chorus effect creates multiple delayed versions of the input signal,
    with the delay times modulated by sine waves. This creates the illusion
    of multiple instruments playing the same part.
    """

    def __init__(self, sample_rate=sample_rate):
        """Initialize chorus effect with default settings.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.rate = DEFAULT_RATE
        self.depth = DEFAULT_DEPTH
        self.mix = DEFAULT_MIX
        self.voices = DEFAULT_VOICES

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

    def _resize_buffer(self):
        """Resize internal buffers based on current depth setting."""
        max_delay_samples = int(self.depth * 2 * self.sample_rate)
        if max_delay_samples <= self._buf_len:
            return  # Buffer is already large enough

        # Resize to next power of 2
        new_size = 1
        while new_size < max_delay_samples:
            new_size *= 2

        self._buf_len = new_size
        self._mask = new_size - 1

        # Create new buffers and preserve old data
        new_buffer_L = np.zeros(new_size, dtype=np.float32)
        new_buffer_R = np.zeros(new_size, dtype=np.float32)

        # Copy existing data (if any)
        if hasattr(self, '_buffer_L') and self._buffer_L is not None:
            for i in range(len(self._buffer_L)):
                new_buffer_L[i] = self._buffer_L[i]
                new_buffer_R[i] = self._buffer_R[i]

        self._buffer_L = new_buffer_L
        self._buffer_R = new_buffer_R

    def set_rate(self, rate):
        """Set the rate of the chorus modulation in Hz."""
        self.rate = max(0.1, min(10.0, rate))  # Clamp between 0.1 and 10 Hz

    def set_depth(self, depth):
        """Set the depth of the chorus modulation in seconds."""
        depth = max(0.001, min(0.030, depth))  # Clamp between 1ms and 30ms
        self.depth = depth
        self._resize_buffer()

    def set_mix(self, mix):
        """Set the dry/wet mix ratio (0.0 = dry, 1.0 = wet)."""
        self.mix = max(0.0, min(1.0, mix))

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

    def process(self, L, R):
        """Process stereo audio with chorus effect.

        Args:
            L: Left channel audio array
            R: Right channel audio array

        Returns:
            out_L, out_R: Processed left and right channels
        """
        # Special case for no processing (bypassed)
        if self.mix <= 0.0:
            return L, R

        # Optimize for single voice case (most common)
        if self.voices == 1:
            return self._process_single_voice(L, R)
        else:
            return self._process_multi_voice(L, R)

    def _process_single_voice(self, L, R):
        """Optimized processing for a single chorus voice."""
        # Calculate all phase values for the entire buffer at once
        buffer_len = len(L)
        phase_values = (self.phase + 2 * np.pi * self.rate / self.sample_rate *
                        np.arange(buffer_len)) % (2 * np.pi)

        # Calculate all LFO values at once
        lfo_values = np.sin(phase_values)

        # Calculate all delay values
        delay_samples = self.base_delay_samples + lfo_values * (self.depth * self.sample_rate)

        # Pre-allocate output arrays
        out_L = np.empty_like(L)
        out_R = np.empty_like(R)

        # Circular buffer management still needs sample-by-sample processing
        # due to the dependency of each sample on the buffer state
        for i in range(buffer_len):
            # Store the current sample in the buffer
            self._buffer_L[self._write_idx] = L[i]
            self._buffer_R[self._write_idx] = R[i]

            # Calculate read indices and fractions for interpolation
            read_idx_f = self._write_idx - delay_samples[i]
            read_idx_i = int(np.floor(read_idx_f)) & self._mask
            read_idx_f_frac = read_idx_f - np.floor(read_idx_f)
            next_idx = (read_idx_i + 1) & self._mask

            # Linear interpolation for delayed samples
            sample_L = ((1.0 - read_idx_f_frac) * self._buffer_L[read_idx_i] +
                       read_idx_f_frac * self._buffer_L[next_idx])

            sample_R = ((1.0 - read_idx_f_frac) * self._buffer_R[read_idx_i] +
                       read_idx_f_frac * self._buffer_R[next_idx])

            # Mix dry and wet signals
            out_L[i] = L[i] * (1.0 - self.mix) + sample_L * self.mix
            out_R[i] = R[i] * (1.0 - self.mix) + sample_R * self.mix

            # Update write index
            self._write_idx = (self._write_idx + 1) & self._mask

        # Update the phase for next buffer
        self.phase = phase_values[-1]

        return out_L, out_R

    def _process_multi_voice(self, L, R):
        """Process with multiple chorus voices."""
        out_L = np.zeros_like(L)
        out_R = np.zeros_like(R)

        # Calculate phase increment per sample
        phase_inc = 2 * np.pi * self.rate / self.sample_rate

        # Base delay in samples
        base_delay = self.base_delay_samples

        # Amplitude of delay modulation in samples
        mod_amplitude = self.depth * self.sample_rate

        # For each sample
        for i in range(len(L)):
            # Store the current sample in the buffer
            self._buffer_L[self._write_idx] = L[i]
            self._buffer_R[self._write_idx] = R[i]

            # Add dry signal
            out_L[i] += L[i] * (1.0 - self.mix)
            out_R[i] += R[i] * (1.0 - self.mix)

            # Add each chorus voice
            voice_mix = self.mix / self.voices

            for v in range(self.voices):
                # Calculate LFO output for this voice
                lfo = np.sin(self.phases[v])

                # Calculate delay samples for this voice
                delay_samples = base_delay + lfo * mod_amplitude

                # Get fractional sample index for interpolation
                read_idx_f = self._write_idx - delay_samples
                read_idx_i = int(np.floor(read_idx_f)) & self._mask
                read_idx_f_frac = read_idx_f - np.floor(read_idx_f)
                next_idx = (read_idx_i + 1) & self._mask

                # Linear interpolation
                sample_L = ((1.0 - read_idx_f_frac) * self._buffer_L[read_idx_i] +
                           read_idx_f_frac * self._buffer_L[next_idx])

                sample_R = ((1.0 - read_idx_f_frac) * self._buffer_R[read_idx_i] +
                           read_idx_f_frac * self._buffer_R[next_idx])

                # Add to output with voice mix level
                out_L[i] += sample_L * voice_mix
                out_R[i] += sample_R * voice_mix

                # Update phase for this voice
                self.phases[v] = (self.phases[v] + phase_inc) % (2 * np.pi)

            # Update write index
            self._write_idx = (self._write_idx + 1) & self._mask

        return out_L, out_R

    def clear_cache(self):
        """Clear the chorus buffers and reset indices."""
        self._buffer_L.fill(0)
        self._buffer_R.fill(0)
        self._write_idx = 0
        self.phase = 0.0
        self.phases.fill(0)
