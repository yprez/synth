"""Configuration and global state for the QWERTY Synth."""

import numpy as np
import threading

# Audio settings
sample_rate = 44100
volume = 0.5
fade_duration = 0.01
max_active_notes = 12  # Maximum number of simultaneous notes before oldest are released

# Visual settings
visual_buffer_size = int(sample_rate * 0.1)
waveform_buffer = np.zeros(visual_buffer_size)
unfiltered_buffer = np.zeros(visual_buffer_size)  # Buffer for storing unfiltered waveform
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

# Mono mode and portamento settings
mono_mode = False
glide_time = 0.05  # in seconds, default 50 ms
mono_pressed_keys = []  # to track currently pressed keys in mono mode

# LFO settings
lfo_rate = 5.0  # Hz
lfo_depth = 0.0  # modulation intensity (0.0-1.0)
lfo_target = 'pitch'  # options: 'pitch', 'volume', 'cutoff'
lfo_attack_time = 0.2  # seconds - time for LFO to reach full intensity
lfo_delay_time = 0.1  # seconds before LFO starts after note trigger
lfo_enabled = True  # Toggle for enabling/disabling LFO

# Delay effect settings
delay_time_ms = 350      # 10 - 1000 ms
delay_feedback = 0.35    # 0.0 - 0.95
delay_mix = 0.35         # 0.0 (dry) - 1.0 (wet)
delay_enabled = False
