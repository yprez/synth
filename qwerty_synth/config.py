"""Configuration and global state for the QWERTY Synth."""

import numpy as np
import threading

# Audio settings
sample_rate = 44100
volume = 0.5
fade_duration = 0.01
max_active_notes = 8  # Maximum number of simultaneous notes before oldest are released
blocksize = 2048  # Audio buffer size

# Visual settings
visual_buffer_size = max(int(sample_rate * 0.1), blocksize)
waveform_buffer = np.zeros(visual_buffer_size)
unfiltered_buffer = np.zeros(visual_buffer_size)  # Buffer for storing unfiltered waveform
buffer_lock = threading.Lock()

# Synth state
active_notes = {}
notes_lock = threading.Lock()

# Octave settings
octave_offset = 0
octave_min = -2
octave_max = 3

# Waveform settings
waveform_type = 'sine'

# ADSR envelope parameters for amplitude
adsr = {
    'attack': 0.01,
    'decay': 0.02,
    'sustain': 0.1,
    'release': 0.2,
}

# ADSR envelope parameters for filter cutoff
filter_adsr = {
    'attack': 0.05,
    'decay': 0.1,
    'sustain': 0.4,
    'release': 0.3,
}

# Filter envelope amount (how much the envelope affects the cutoff)
filter_env_amount = 5000  # Hz

# Filter settings
filter_cutoff = 10000  # Default cutoff frequency in Hz
filter_resonance = 0.0  # Default resonance (0.0-1.0), higher values create more pronounced peaks
filter_enabled = False  # Flag to enable/disable the filter
filter_type = 'lowpass'  # 'lowpass', 'highpass', 'bandpass', 'notch'
filter_topology = 'svf'  # 'svf' (State Variable Filter) or 'biquad'
filter_slope = 24  # Filter rolloff slope: 12 or 24 dB/octave

# Drive settings
drive_gain = 1.2        # 0.0 (clean) – 3.0 (heavy)
drive_on = False        # master toggle
drive_type = 'tanh'     # 'tanh', 'arctan', 'cubic', 'fuzz', 'asymmetric'
drive_tone = 0.0        # -1.0 (dark) to 1.0 (bright)
drive_mix = 1.0         # 0.0 (dry) to 1.0 (wet)
drive_asymmetry = 0.2   # 0.0 (symmetric) to 0.9 (asymmetric) - for asymmetric mode

# Mono mode and portamento settings
mono_mode = False
glide_time = 0.05  # in seconds, default 50 ms
mono_pressed_keys = []  # to track currently pressed keys in mono mode

# LFO settings
lfo_rate = 5.0  # Hz
lfo_depth = 0.1  # modulation intensity (0.0-1.0)
lfo_target = 'pitch'  # options: 'pitch', 'volume', 'cutoff'
lfo_attack_time = 0.2  # seconds - time for LFO to reach full intensity
lfo_delay_time = 0.1  # seconds before LFO starts after note trigger
lfo_enabled = False  # Toggle for enabling/disabling LFO

# Delay effect settings
delay_time_ms = 350      # 10 - 1000 ms
delay_feedback = 0.1     # 0.0 - 0.95
delay_mix = 0.1          # 0.0 (dry) - 1.0 (wet)
delay_enabled = False
delay_sync_enabled = True  # Whether delay time is synced to BPM
delay_pingpong = False     # Ping-pong stereo effect

# Chorus effect settings
chorus_rate = 0.5        # Hz - speed of modulation (0.1 - 10.0 Hz)
chorus_depth = 0.005     # Seconds - depth of modulation (0.001 - 0.030 s)
chorus_mix = 0.2         # 0.0 (dry) - 1.0 (wet)
chorus_voices = 1        # Number of chorus voices (1 - 4) - using 1 for CPU efficiency
chorus_enabled = False   # Toggle for enabling/disabling chorus effect

# Global tempo and sync settings
bpm = 120               # master tempo
delay_division = '1/4'  # '1/1', '1/2', '1/4', '1/8', '1/8d', '1/16', '1/16t'

# MIDI playback control
midi_paused = False
midi_playback_active = False
midi_playback_start_time = 0
midi_playback_duration = 0
midi_tempo_scale = 1.0

# Recording settings
recording_enabled = False
recorded_audio = []
recording_path = None
recording_bit_depth = 24  # 16 or 24 bit

# Arpeggiator settings
arpeggiator_enabled = False
arpeggiator_pattern = 'up'     # 'up', 'down', 'up_down', 'down_up', 'random', 'chord', 'octaves', 'order'
arpeggiator_rate = 120         # BPM when not synced
arpeggiator_gate = 0.8         # Note gate time (0.1 - 1.0)
arpeggiator_octave_range = 1   # Number of octaves to span (1-4)
arpeggiator_sync_to_bpm = True # Whether to sync to global BPM
arpeggiator_sustain_base = True  # Whether to sustain base notes while arpeggiating
arpeggiator_held_notes = set() # Currently held notes for arpeggio
