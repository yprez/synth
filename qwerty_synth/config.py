"""Configuration and global state for the QWERTY Synth."""

import numpy as np
import threading

# Audio settings
sample_rate = 44100
volume = 0.3
fade_duration = 0.01

# Visual settings
visual_buffer_size = int(sample_rate * 0.1)
waveform_buffer = np.zeros(visual_buffer_size)
buffer_lock = threading.Lock()

# Note mapping
key_note_map = {
    'a': 261.63, 'w': 277.18, 's': 293.66, 'e': 311.13,
    'd': 329.63, 'f': 349.23, 't': 369.99, 'g': 392.00,
    'y': 415.30, 'h': 440.00, 'u': 466.16, 'j': 493.88,
    'k': 523.25, 'o': 554.37, 'l': 587.33, 'p': 622.25,
    ';': 659.25, "'": 698.46
}

# Synth state
active_notes = {}
notes_lock = threading.Lock()

# Octave settings
octave_offset = 0
octave_min = -2
octave_max = 3

# Waveform settings
waveform_type = 'sine'
