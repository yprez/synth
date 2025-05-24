"""Delay/echo effect"""

import numpy as np
from numba import jit
from qwerty_synth.config import sample_rate, delay_time_ms


# Division to multiplier mapping
DIV2MULT = {
    '1/1'  : 1.0,     # Whole note
    '1/2'  : 0.5,     # Half note
    '1/4'  : 0.25,    # Quarter note
    '1/8'  : 0.125,   # Eighth note
    '1/8d' : 0.1875,  # Dotted eighth (1/8 × 1.5)
    '1/16' : 0.0625,  # Sixteenth note
    '1/16t': 0.0416667  # Triplet sixteenth (1/16 × 2/3)
}


@jit(nopython=True, cache=True, fastmath=True)
def _process_delay_interpolated(x, delayed, buffer, delay_samples_f,
                               write_idx, mask, fb):
    """JIT-compiled delay processing with interpolation."""
    buffer_len = len(x)

    for i in range(buffer_len):
        # Linear interpolation for fractional sample accuracy
        idx_f = write_idx - delay_samples_f
        idx_i = int(np.floor(idx_f)) & mask
        frac = idx_f - np.floor(idx_f)
        delayed[i] = ((1.0 - frac) * buffer[idx_i] +
                      frac * buffer[(idx_i + 1) & mask])

        # Process sample
        buffer[write_idx] = x[i] + delayed[i] * fb
        write_idx = (write_idx + 1) & mask

    return write_idx


@jit(nopython=True, cache=True, fastmath=True)
def _process_delay_simple(x, delayed, buffer, delay_samples,
                         write_idx, mask, fb):
    """JIT-compiled delay processing without interpolation."""
    buffer_len = len(x)

    for i in range(buffer_len):
        # Simple integer sample delay
        read_idx = (write_idx - delay_samples) & mask
        delayed[i] = buffer[read_idx]

        # Process sample
        buffer[write_idx] = x[i] + delayed[i] * fb
        write_idx = (write_idx + 1) & mask

    return write_idx


@jit(nopython=True, cache=True, fastmath=True)
def _process_pingpong_interpolated(L, R, delayed_L, delayed_R,
                                  buffer_L, buffer_R, delay_samples_f,
                                  write_idx_L, write_idx_R, mask, fb):
    """JIT-compiled ping-pong delay processing with interpolation."""
    buffer_len = len(L)

    for i in range(buffer_len):
        # Linear interpolation for L channel
        idx_f_L = write_idx_L - delay_samples_f
        idx_i_L = int(np.floor(idx_f_L)) & mask
        frac_L = idx_f_L - np.floor(idx_f_L)
        delayed_L[i] = ((1.0 - frac_L) * buffer_L[idx_i_L] +
                       frac_L * buffer_L[(idx_i_L + 1) & mask])

        # Linear interpolation for R channel
        idx_f_R = write_idx_R - delay_samples_f
        idx_i_R = int(np.floor(idx_f_R)) & mask
        frac_R = idx_f_R - np.floor(idx_f_R)
        delayed_R[i] = ((1.0 - frac_R) * buffer_R[idx_i_R] +
                       frac_R * buffer_R[(idx_i_R + 1) & mask])

        # Cross-feedback: L feeds R buffer and vice-versa
        buffer_L[write_idx_L] = L[i] + delayed_R[i] * fb
        buffer_R[write_idx_R] = R[i] + delayed_L[i] * fb

        write_idx_L = (write_idx_L + 1) & mask
        write_idx_R = (write_idx_R + 1) & mask

    return write_idx_L, write_idx_R


@jit(nopython=True, cache=True, fastmath=True)
def _process_pingpong_simple(L, R, delayed_L, delayed_R,
                            buffer_L, buffer_R, delay_samples,
                            write_idx_L, write_idx_R, mask, fb):
    """JIT-compiled ping-pong delay processing without interpolation."""
    buffer_len = len(L)

    for i in range(buffer_len):
        # Simple integer sample delay
        read_idx_L = (write_idx_L - delay_samples) & mask
        read_idx_R = (write_idx_R - delay_samples) & mask
        delayed_L[i] = buffer_L[read_idx_L]
        delayed_R[i] = buffer_R[read_idx_R]

        # Cross-feedback: L feeds R buffer and vice-versa
        buffer_L[write_idx_L] = L[i] + delayed_R[i] * fb
        buffer_R[write_idx_R] = R[i] + delayed_L[i] * fb

        write_idx_L = (write_idx_L + 1) & mask
        write_idx_R = (write_idx_R + 1) & mask

    return write_idx_L, write_idx_R


class Delay:
    """Delay effect processor with mono and ping-pong functionality."""

    def __init__(self, sample_rate=sample_rate, delay_ms=delay_time_ms):
        self.sample_rate = sample_rate
        self.delay_ms = delay_ms
        self.delay_samples = 0
        self.delay_samples_f = 0.0
        self._buf_len = 0
        self._buffer = None
        self._buffer_L = None
        self._buffer_R = None
        self._write_idx = 0
        self._write_idx_L = 0
        self._write_idx_R = 0
        self._mask = 0

        # Pre-allocated temporary arrays for performance
        self._temp_delayed = None
        self._temp_delayed_L = None
        self._temp_delayed_R = None
        self._max_block_size = 1024  # Initial size, will grow as needed

        # Initialize with default delay time
        self.set_time(delay_ms)

    def _resize_buffer(self, new_samples):
        """Resize buffers to accommodate the requested delay time."""
        if new_samples <= self._buf_len:
            return  # enough room already

        # Use power of 2 for buffer size for efficient modulo
        target_size = 1
        while target_size < new_samples:
            target_size *= 2

        self._buf_len = target_size
        self._mask = target_size - 1  # For efficient modulo with bitwise AND

        # Create new buffers
        self._buffer = np.zeros(self._buf_len, dtype=np.float32)
        self._buffer_L = np.zeros(self._buf_len, dtype=np.float32)
        self._buffer_R = np.zeros(self._buf_len, dtype=np.float32)

        # Reset write indices
        self._write_idx = 0
        self._write_idx_L = 0
        self._write_idx_R = 0

    def _ensure_temp_arrays(self, block_size):
        """Ensure temporary arrays are large enough for the block size."""
        if (self._temp_delayed is None or
            len(self._temp_delayed) < block_size):
            self._max_block_size = max(self._max_block_size, block_size)
            self._temp_delayed = np.empty(self._max_block_size, dtype=np.float32)
            self._temp_delayed_L = np.empty(self._max_block_size, dtype=np.float32)
            self._temp_delayed_R = np.empty(self._max_block_size, dtype=np.float32)

    def set_time(self, ms, use_interpolation=True):
        """Set delay time in milliseconds."""
        self.delay_ms = ms

        if use_interpolation:
            # Store exact fractional delay time for interpolation
            self.delay_samples_f = self.delay_ms * self.sample_rate / 1000
            self.delay_samples = int(np.floor(self.delay_samples_f))
        else:
            # Round to nearest sample for simple implementation
            self.delay_samples = round(self.delay_ms * self.sample_rate / 1000)
            self.delay_samples_f = float(self.delay_samples)

        # Ensure buffer has enough space (with headroom)
        self._resize_buffer(self.delay_samples * 2)

        # Safe time change: ensure write indices are within new delay range
        self._write_idx %= self._buf_len
        self._write_idx_L %= self._buf_len
        self._write_idx_R %= self._buf_len

    def update_delay_from_bpm(self, bpm, division):
        """Update delay time based on BPM and selected division."""
        from qwerty_synth import config

        beats_per_second = bpm / 60.0
        seconds_per_beat = 1 / beats_per_second
        seconds_per_whole_note = seconds_per_beat * 4.0
        delay_ms = seconds_per_whole_note * DIV2MULT.get(division, 0.25) * 1000

        config.delay_time_ms = delay_ms
        self.set_time(delay_ms)
        return delay_ms

    def process_block(self, x, fb, mix, use_interpolation=True):
        """Process an audio block with mono delay effect.

        Args:
            x: 1-D numpy array of input audio
            fb: Feedback amount (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
            use_interpolation: Whether to use linear interpolation

        Returns:
            y: Processed audio with delay effect
        """
        # Safety clamp feedback to prevent runaway gain
        fb = np.clip(fb, 0.0, 0.99)

        buffer_len = len(x)
        self._ensure_temp_arrays(buffer_len)

        # Use pre-allocated slice of temporary array
        delayed = self._temp_delayed[:buffer_len]

        # Pre-calculate dry component (vectorized)
        dry_component = x * (1.0 - mix)

        if use_interpolation:
            self._write_idx = _process_delay_interpolated(
                x, delayed, self._buffer, self.delay_samples_f,
                self._write_idx, self._mask, fb)
        else:
            self._write_idx = _process_delay_simple(
                x, delayed, self._buffer, self.delay_samples,
                self._write_idx, self._mask, fb)

        # Vectorized wet/dry mix
        return dry_component + delayed * mix

    def pingpong(self, L, R, mix, fb, use_interpolation=True):
        """Process stereo audio with ping-pong delay effect.

        Args:
            L: Left channel audio array
            R: Right channel audio array
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
            fb: Feedback amount (0.0 to 1.0)
            use_interpolation: Whether to use linear interpolation

        Returns:
            out_L, out_R: Processed left and right channels
        """
        # Safety clamp feedback to prevent runaway gain
        fb = np.clip(fb, 0.0, 0.99)

        buffer_len = len(L)
        self._ensure_temp_arrays(buffer_len)

        # Use pre-allocated slices of temporary arrays
        delayed_L = self._temp_delayed_L[:buffer_len]
        delayed_R = self._temp_delayed_R[:buffer_len]

        # Pre-calculate dry components (vectorized)
        dry_L = L * (1.0 - mix)
        dry_R = R * (1.0 - mix)

        if use_interpolation:
            self._write_idx_L, self._write_idx_R = _process_pingpong_interpolated(
                L, R, delayed_L, delayed_R, self._buffer_L, self._buffer_R,
                self.delay_samples_f, self._write_idx_L, self._write_idx_R,
                self._mask, fb)
        else:
            self._write_idx_L, self._write_idx_R = _process_pingpong_simple(
                L, R, delayed_L, delayed_R, self._buffer_L, self._buffer_R,
                self.delay_samples, self._write_idx_L, self._write_idx_R,
                self._mask, fb)

        # Vectorized wet/dry mix
        return dry_L + delayed_L * mix, dry_R + delayed_R * mix

    def clear_cache(self):
        """Clear the delay buffers and reset write indices."""
        if self._buffer is not None:
            self._buffer.fill(0)
        if self._buffer_L is not None:
            self._buffer_L.fill(0)
        if self._buffer_R is not None:
            self._buffer_R.fill(0)
        self._write_idx = 0
        self._write_idx_L = 0
        self._write_idx_R = 0
