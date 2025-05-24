"""Chorus effect module for QWERTY Synth.

A chorus effect creates a richer sound by mixing the input signal with
slightly delayed copies that have a time-varying delay, creating a
sense of multiple voices playing in unison.
"""

import numpy as np
from qwerty_synth.config import sample_rate, chorus_rate, chorus_depth, chorus_mix, chorus_voices


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
        self.rate = chorus_rate
        self.depth = chorus_depth
        self.mix = chorus_mix
        self.voices = chorus_voices

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

        # Pre-compute all phase values for the entire buffer at once
        phase_values = (self.phase + self._phase_inc * np.arange(buffer_len)) % (2 * np.pi)

        # Calculate all LFO values at once using vectorized operations
        lfo_values = np.sin(phase_values)

        # Calculate all delay values
        delay_samples = self.base_delay_samples + lfo_values * self._depth_samples

        # Pre-allocate output arrays
        out_L = np.empty_like(L, dtype=np.float32)
        out_R = np.empty_like(R, dtype=np.float32)

        # Optimized sample-by-sample processing
        for i in range(buffer_len):
            # Store the current sample in the buffer
            self._buffer_L[self._write_idx] = L[i]
            self._buffer_R[self._write_idx] = R[i]

            # Fast delay calculation with optimized math operations
            delay = delay_samples[i]
            read_pos = self._write_idx - delay
            read_idx = int(read_pos) & self._mask  # Faster than np.floor + modulo
            frac = read_pos - int(read_pos)  # Faster than separate floor operation
            next_idx = (read_idx + 1) & self._mask

            # Optimized linear interpolation with cached calculations
            inv_frac = 1.0 - frac
            sample_L = inv_frac * self._buffer_L[read_idx] + frac * self._buffer_L[next_idx]
            sample_R = inv_frac * self._buffer_R[read_idx] + frac * self._buffer_R[next_idx]

            # Direct mixing with pre-computed gains
            out_L[i] = L[i] * self._dry_gain + sample_L * self._wet_gain
            out_R[i] = R[i] * self._dry_gain + sample_R * self._wet_gain

            # Update write index
            self._write_idx = (self._write_idx + 1) & self._mask

        # Update the phase for next buffer
        self.phase = phase_values[-1]

        return out_L, out_R

    def _process_multi_voice(self, L, R):
        """Optimized processing with multiple chorus voices."""
        buffer_len = len(L)
        out_L = np.zeros_like(L, dtype=np.float32)
        out_R = np.zeros_like(R, dtype=np.float32)

        # Pre-compute constants outside the loop
        base_delay = self.base_delay_samples
        mod_amplitude = self._depth_samples

        # Add dry signal once (vectorized operation)
        out_L += L * self._dry_gain
        out_R += R * self._dry_gain

        # Process each sample
        for i in range(buffer_len):
            # Store the current sample in the buffer
            self._buffer_L[self._write_idx] = L[i]
            self._buffer_R[self._write_idx] = R[i]

            # Process each chorus voice
            for v in range(self.voices):
                # Calculate LFO output for this voice (cached phase increment)
                lfo = np.sin(self.phases[v])

                # Calculate delay samples for this voice
                delay_samples = base_delay + lfo * mod_amplitude

                # Optimized fractional sample index calculation
                read_pos = self._write_idx - delay_samples
                read_idx = int(read_pos) & self._mask
                frac = read_pos - int(read_pos)
                next_idx = (read_idx + 1) & self._mask

                # Optimized linear interpolation
                inv_frac = 1.0 - frac
                sample_L = inv_frac * self._buffer_L[read_idx] + frac * self._buffer_L[next_idx]
                sample_R = inv_frac * self._buffer_R[read_idx] + frac * self._buffer_R[next_idx]

                # Add to output with pre-computed voice mix level
                out_L[i] += sample_L * self._voice_mix
                out_R[i] += sample_R * self._voice_mix

                # Update phase for this voice (cached phase increment)
                self.phases[v] = (self.phases[v] + self._phase_inc) % (2 * np.pi)

            # Update write index
            self._write_idx = (self._write_idx + 1) & self._mask

        return out_L, out_R

    def clear_cache(self):
        """Clear the chorus buffers and reset indices efficiently."""
        # Use numpy's optimized fill operations
        self._buffer_L.fill(0.0)
        self._buffer_R.fill(0.0)
        self._write_idx = 0
        self.phase = 0.0
        if len(self.phases) > 0:
            self.phases.fill(0.0)
