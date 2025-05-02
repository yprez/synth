"""Delay/echo effect"""

import numpy as np
from qwerty_synth.config import sample_rate, delay_time_ms

# Allocate max 1 s buffer
_max_samples = int(1.0 * sample_rate)
_buffer = np.zeros(_max_samples, dtype=np.float32)
write_idx = 0
delay_samples = 0

def set_time(ms):
    """Set delay time in milliseconds."""
    global delay_samples
    delay_samples = int(ms * sample_rate / 1000)

# Initialize delay samples
set_time(delay_time_ms)

def process_block(x, fb, mix):
    """Process an audio block with delay effect.

    Args:
        x: 1-D numpy array of input audio
        fb: Feedback amount (0.0 to 1.0)
        mix: Wet/dry mix (0.0 = dry, 1.0 = wet)

    Returns:
        y: Processed audio with delay effect
    """
    global write_idx, _buffer, delay_samples
    y = np.empty_like(x)
    for i, s in enumerate(x):
        read_idx = (write_idx - delay_samples) % _buffer.size
        delayed = _buffer[read_idx]
        _buffer[write_idx] = s + delayed * fb   # feedback write
        y[i] = s * (1.0 - mix) + delayed * mix
        write_idx = (write_idx + 1) % _buffer.size
    return y
