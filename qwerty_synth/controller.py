"""Programmatic API for controlling the synthesizer."""

import threading
import time

from qwerty_synth import config
from qwerty_synth.synth import Oscillator


# Counter to generate unique keys for notes in polyphonic mode
_note_counter = 0


def play_note(freq, duration=0.5, velocity=1.0):
    """
    Play a note programmatically.

    Args:
        freq: Frequency in Hz
        duration: Note duration in seconds
        velocity: Note velocity (0.0-1.0)
    """
    global _note_counter

    # In mono mode, always use 'mono' key
    # In poly mode, use unique key for each note
    if config.mono_mode:
        k = 'mono'
    else:
        # Create a unique key for each new note in poly mode
        _note_counter += 1
        k = f'program_{_note_counter}'

    with config.notes_lock:
        # If in mono mode and a note is already playing, update its target frequency
        if config.mono_mode and 'mono' in config.active_notes:
            osc = config.active_notes['mono']
            osc.target_freq = freq
            osc.key = k

            # If oscillator was released, un-release it
            if osc.released:
                osc.released = False
                osc.env_time = 0.0  # Reset envelope time to restart attack
                osc.lfo_env_time = 0.0  # Reset LFO envelope time
        else:
            # Create new oscillator instance
            osc = Oscillator(freq, config.waveform_type)
            osc.key = k

            # Store velocity with the oscillator for per-note volume control
            # This will be implemented in Oscillator.generate() in the future
            osc.velocity = velocity

            # Add the note to active notes
            config.active_notes[k] = osc

    def release():
        with config.notes_lock:
            if k in config.active_notes:
                config.active_notes[k].released = True
                config.active_notes[k].env_time = 0.0
                config.active_notes[k].lfo_env_time = 0.0

    # Schedule note release after duration
    if duration > 0:
        threading.Timer(duration, release).start()

    return osc


def midi_to_freq(midi_note):
    """
    Convert MIDI note number to frequency in Hz.

    Args:
        midi_note: MIDI note number (e.g., 69 for A4)

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def play_midi_note(midi_note, duration=0.5, velocity=1.0):
    """
    Play a note using MIDI note number.

    Args:
        midi_note: MIDI note number (e.g., 69 for A4)
        duration: Note duration in seconds
        velocity: Note velocity (0.0-1.0)
    """
    freq = midi_to_freq(midi_note)
    return play_note(freq, duration, velocity)


def play_sequence(sequence, interval=0.0):
    """
    Play a sequence of notes.

    Args:
        sequence: List of tuples (freq/midi_note, duration, velocity)
                 where velocity is optional
        interval: Additional interval between notes in seconds
    """
    def play_next(seq, index):
        if index >= len(seq):
            return

        note = seq[index]

        # Handle different formats of note specification
        if len(note) >= 2:
            freq_or_midi = note[0]
            duration = note[1]
            velocity = note[2] if len(note) > 2 else 1.0

            # Determine if it's a frequency or MIDI note number
            if isinstance(freq_or_midi, int) and freq_or_midi < 128:
                # Likely a MIDI note
                play_midi_note(freq_or_midi, duration, velocity)
            else:
                # Frequency in Hz
                play_note(freq_or_midi, duration, velocity)

            # Schedule next note
            total_delay = duration + interval
            threading.Timer(total_delay, play_next, [seq, index + 1]).start()

    # Start the sequence
    play_next(sequence, 0)
