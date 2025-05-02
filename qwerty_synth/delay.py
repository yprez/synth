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

def update_delay_from_bpm():
    """Update delay time based on BPM and selected division."""
    from qwerty_synth import config

    beats_per_second = config.bpm / 60.0
    seconds_per_beat = 1 / beats_per_second
    seconds_per_whole_note = seconds_per_beat * 4.0
    delay_ms = seconds_per_whole_note * DIV2MULT.get(config.delay_division, 0.25) * 1000  # seconds→ms
    config.delay_time_ms = delay_ms
    set_time(delay_ms)  # Update the actual delay time

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

def clear_cache():
    """Clear the delay buffer and reset write index."""
    global _buffer, write_idx
    _buffer.fill(0)
    write_idx = 0
