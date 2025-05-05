"""Delay/echo effect"""

import numpy as np
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

        y = np.empty_like(x)
        for i, s in enumerate(x):
            if use_interpolation:
                # Linear interpolation for fractional sample accuracy
                idx_f = self._write_idx - self.delay_samples_f
                idx_i = int(np.floor(idx_f)) & self._mask
                frac = idx_f - np.floor(idx_f)
                delayed = ((1.0 - frac) * self._buffer[idx_i] +
                           frac * self._buffer[(idx_i + 1) & self._mask])
            else:
                # Simple integer sample delay
                read_idx = (self._write_idx - self.delay_samples) & self._mask
                delayed = self._buffer[read_idx]

            # Process sample
            self._buffer[self._write_idx] = s + delayed * fb
            y[i] = s * (1.0 - mix) + delayed * mix
            self._write_idx = (self._write_idx + 1) & self._mask

        return y

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

        out_L = np.empty_like(L)
        out_R = np.empty_like(R)

        for i, (sL, sR) in enumerate(zip(L, R)):
            if use_interpolation:
                # Linear interpolation for L channel
                idx_f_L = self._write_idx_L - self.delay_samples_f
                idx_i_L = int(np.floor(idx_f_L)) & self._mask
                frac_L = idx_f_L - np.floor(idx_f_L)
                dl = ((1.0 - frac_L) * self._buffer_L[idx_i_L] +
                      frac_L * self._buffer_L[(idx_i_L + 1) & self._mask])

                # Linear interpolation for R channel
                idx_f_R = self._write_idx_R - self.delay_samples_f
                idx_i_R = int(np.floor(idx_f_R)) & self._mask
                frac_R = idx_f_R - np.floor(idx_f_R)
                dr = ((1.0 - frac_R) * self._buffer_R[idx_i_R] +
                      frac_R * self._buffer_R[(idx_i_R + 1) & self._mask])
            else:
                # Simple integer sample delay
                read_idx_L = (self._write_idx_L - self.delay_samples) & self._mask
                read_idx_R = (self._write_idx_R - self.delay_samples) & self._mask
                dl = self._buffer_L[read_idx_L]
                dr = self._buffer_R[read_idx_R]

            # Dry/wet mix
            out_L[i] = sL * (1.0 - mix) + dl * mix
            out_R[i] = sR * (1.0 - mix) + dr * mix

            # Cross-feedback: L feeds R buffer and vice-versa
            self._buffer_L[self._write_idx_L] = sL + dr * fb
            self._buffer_R[self._write_idx_R] = sR + dl * fb

            self._write_idx_L = (self._write_idx_L + 1) & self._mask
            self._write_idx_R = (self._write_idx_R + 1) & self._mask

        return out_L, out_R

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


# For backward compatibility with the module-level API
_delay = Delay(sample_rate, delay_time_ms)

def set_time(ms):
    """Set delay time in milliseconds (backward compatibility)."""
    _delay.set_time(ms)

def update_delay_from_bpm():
    """Update delay time based on BPM and division (backward compatibility)."""
    from qwerty_synth import config
    return _delay.update_delay_from_bpm(config.bpm, config.delay_division)

def process_block(x, fb, mix):
    """Process an audio block with delay (backward compatibility)."""
    return _delay.process_block(x, fb, mix)

def pingpong(L, R, mix, fb):
    """Process stereo audio with ping-pong delay (backward compatibility)."""
    return _delay.pingpong(L, R, mix, fb)

def clear_cache():
    """Clear delay buffers (backward compatibility)."""
    _delay.clear_cache()
