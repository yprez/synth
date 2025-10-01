"""Programmatic API for controlling the synthesizer."""

import threading
import time
import mido

from qwerty_synth import config
from qwerty_synth.keyboard_midi import MidiEvent
from qwerty_synth.synth import Oscillator


# Counter to generate unique keys for notes in polyphonic mode
_note_counter = 0

# Keyboard event bookkeeping for translator-driven input
_keyboard_active_note_keys: dict[int, str] = {}
_keyboard_pressed_notes: list[int] = []
_keyboard_note_velocities: dict[int, float] = {}


def _velocity_to_scalar(velocity: int | float | None) -> float:
    """Convert MIDI velocity (0-127) into a 0.0-1.0 scalar."""
    if velocity is None:
        return 1.0

    if isinstance(velocity, float):
        return max(0.0, min(1.0, velocity))

    return max(0.0, min(1.0, velocity / 127.0))


def _sync_mono_pressed_notes() -> None:
    """Keep legacy mono pressed list in sync for GUI interactions."""
    config.mono_pressed_keys = list(_keyboard_pressed_notes)


def get_octave_offset() -> int:
    """Return the current octave offset in semitones."""
    with config.notes_lock:
        return config.octave_offset


def get_semitone_offset() -> int:
    """Return the current semitone transpose offset."""
    with config.notes_lock:
        return config.semitone_offset


def _get_arpeggiator_module():
    try:
        from qwerty_synth import arpeggiator as arpeggiator_module  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    return arpeggiator_module


def _clear_keyboard_state_unlocked() -> None:
    _keyboard_active_note_keys.clear()
    _keyboard_pressed_notes.clear()
    _keyboard_note_velocities.clear()
    config.mono_pressed_keys.clear()


def reset_keyboard_state() -> None:
    """Clear cached keyboard tracking structures."""
    with config.notes_lock:
        _clear_keyboard_state_unlocked()


def update_mono_mode(enabled: bool) -> None:
    """Toggle mono mode and clear state safely."""
    with config.notes_lock:
        config.mono_mode = enabled
        config.active_notes.clear()
        _clear_keyboard_state_unlocked()


def set_octave(value: int) -> None:
    """Set octave offset from a GUI control (value expressed in octaves)."""
    semitone_offset = value * 12
    with config.notes_lock:
        minimum = 12 * config.octave_min
        maximum = 12 * config.octave_max
        config.octave_offset = max(minimum, min(maximum, semitone_offset))


def set_semitone(value: int) -> None:
    """Set semitone offset from GUI control input."""
    with config.notes_lock:
        config.semitone_offset = max(config.semitone_min, min(config.semitone_max, value))


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
    # Apply global octave and semitone offsets to the MIDI note
    adjusted_midi_note = midi_note + config.octave_offset + config.semitone_offset
    freq = midi_to_freq(adjusted_midi_note)
    return play_note(freq, duration, velocity)


def play_midi_note_direct(midi_note, duration=0.5, velocity=1.0):
    """
    Play a note using MIDI note number without applying global transpose settings.
    This is used by the step sequencer and other components that should not be
    affected by the global octave and semitone transpose settings.

    Args:
        midi_note: MIDI note number (e.g., 69 for A4)
        duration: Note duration in seconds
        velocity: Note velocity (0.0-1.0)
    """
    # Play the MIDI note directly without applying global transpose settings
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


def handle_midi_message(event: MidiEvent) -> None:
    """Handle MIDI-style events emitted by the keyboard translator."""
    if event.event_type == 'note_on':
        if event.note is None:
            return
        if event.velocity <= 0:
            _handle_keyboard_note_off(event.note)
            return
        _handle_keyboard_note_on(event.note, event.velocity)
    elif event.event_type == 'note_off':
        if event.note is None:
            return
        _handle_keyboard_note_off(event.note)
    elif event.event_type == 'transpose':
        delta = int(event.payload.get('delta', 0)) if event.payload else 0
        if delta:
            apply_transpose_delta(delta)


def apply_transpose_delta(delta: int) -> bool:
    """Apply a transpose delta (in semitones) while respecting configuration bounds."""
    with config.notes_lock:
        new_offset = config.octave_offset + delta
        minimum = 12 * config.octave_min
        maximum = 12 * config.octave_max

        if not minimum <= new_offset <= maximum:
            return False

        config.octave_offset = new_offset

    direction = 'up' if delta > 0 else 'down'
    print(f'Octave {direction}: {config.octave_offset // 12:+}')

    arpeggiator_module = _get_arpeggiator_module()
    if config.arpeggiator_enabled and arpeggiator_module and arpeggiator_module.arpeggiator_instance:
        arpeggiator_module.arpeggiator_instance.clear_notes()

    return True


def _handle_keyboard_note_on(midi_note: int, velocity: int) -> None:
    velocity_scalar = _velocity_to_scalar(velocity)

    with config.notes_lock:
        if midi_note in _keyboard_pressed_notes:
            return

        _keyboard_pressed_notes.append(midi_note)
        _keyboard_note_velocities[midi_note] = velocity_scalar
        _sync_mono_pressed_notes()

        arpeggiator_module = _get_arpeggiator_module()
        if config.arpeggiator_enabled and arpeggiator_module and arpeggiator_module.arpeggiator_instance:
            arpeggiator_module.arpeggiator_instance.add_note(midi_note)
            if not config.arpeggiator_sustain_base:
                return

        freq = midi_to_freq(midi_note)

        if config.mono_mode:
            for existing_note, key_name in list(_keyboard_active_note_keys.items()):
                if key_name == 'mono':
                    _keyboard_active_note_keys.pop(existing_note, None)

            osc = config.active_notes.get('mono')

            if osc is None or osc.released:
                osc = Oscillator(freq, config.waveform_type)
                osc.key = 'mono'
                config.active_notes['mono'] = osc
            else:
                osc.target_freq = freq
                osc.lfo_env_time = 0.0

            osc.velocity = velocity_scalar
            osc.released = False
            osc.env_time = 0.0
            _keyboard_active_note_keys[midi_note] = 'mono'
        else:
            if midi_note in _keyboard_active_note_keys:
                return

            key = f'keyboard_{midi_note}'
            osc = Oscillator(freq, config.waveform_type)
            osc.key = key
            osc.velocity = velocity_scalar
            config.active_notes[key] = osc
            _keyboard_active_note_keys[midi_note] = key


def _handle_keyboard_note_off(midi_note: int) -> None:
    with config.notes_lock:
        if midi_note in _keyboard_pressed_notes:
            _keyboard_pressed_notes.remove(midi_note)
            _sync_mono_pressed_notes()

        _keyboard_note_velocities.pop(midi_note, None)

        arpeggiator_module = _get_arpeggiator_module()
        if config.arpeggiator_enabled and arpeggiator_module and arpeggiator_module.arpeggiator_instance:
            arpeggiator_module.arpeggiator_instance.remove_note(midi_note)

        if config.mono_mode:
            _keyboard_active_note_keys.pop(midi_note, None)
            if _keyboard_pressed_notes:
                next_note = _keyboard_pressed_notes[-1]
                freq = midi_to_freq(next_note)
                osc = config.active_notes.get('mono')
                if osc is not None:
                    osc.target_freq = freq
                    osc.lfo_env_time = 0.0
                    osc.velocity = _keyboard_note_velocities.get(next_note, osc.velocity)
                _keyboard_active_note_keys[next_note] = 'mono'
            else:
                osc = config.active_notes.get('mono')
                if osc is not None:
                    osc.released = True
                    osc.env_time = 0.0
                    osc.lfo_env_time = 0.0
                _keyboard_active_note_keys.pop(midi_note, None)
            return

        key = _keyboard_active_note_keys.pop(midi_note, None)
        if key and key in config.active_notes:
            osc = config.active_notes[key]
            osc.released = True
            osc.env_time = 0.0
            osc.lfo_env_time = 0.0


def play_midi_file(midi_file_path, tempo_scale=1.0):
    """
    Play a MIDI file using the synthesizer.

    Args:
        midi_file_path: Path to the MIDI file
        tempo_scale: Scale factor for tempo (1.0 = normal speed, 0.5 = half speed, etc.)
    """
    try:
        # Load the MIDI file
        midi_file = mido.MidiFile(midi_file_path)

        # Store original tempo scale for potential later adjustments
        config.midi_tempo_scale = tempo_scale

        # Estimate total duration for progress tracking
        total_duration = sum(msg.time for msg in midi_file) / tempo_scale
        config.midi_playback_duration = total_duration

        # Reset playback state
        config.midi_playback_active = True

        # Track active notes for each channel to handle note_off events
        active_notes = {}

        def play_midi_events():
            """Play MIDI events in a separate thread with improved timing."""
            # Use absolute timing instead of sleep-based timing
            start_time = time.perf_counter()
            current_time = 0
            pause_start_time = 0
            total_pause_time = 0

            try:
                for msg in midi_file:
                    # Check if playback has been stopped
                    if not config.midi_playback_active:
                        # Release any active notes
                        for channel_key in list(active_notes.keys()):
                            with config.notes_lock:
                                osc = active_notes[channel_key]
                                if osc.key in config.active_notes:
                                    config.active_notes[osc.key].released = True
                        break

                    # Handle paused state
                    while config.midi_paused and config.midi_playback_active:
                        if pause_start_time == 0:
                            pause_start_time = time.perf_counter()
                        time.sleep(0.01)  # Reduced sleep time for better responsiveness

                    # If we were paused and resumed
                    if pause_start_time > 0:
                        pause_end_time = time.perf_counter()
                        total_pause_time += (pause_end_time - pause_start_time)
                        pause_start_time = 0

                    # Check if tempo scale has changed
                    current_time_scale = 1.0 / config.midi_tempo_scale

                    # Calculate when this event should happen
                    current_time += msg.time * current_time_scale
                    # Adjust target time by subtracting pause duration
                    target_time = start_time + current_time - total_pause_time

                    # Wait until it's time to process this event - with improved precision
                    wait_time = target_time - time.perf_counter()
                    if wait_time > 0:
                        # For short waits, use busy waiting for high precision
                        if wait_time < 0.01:
                            while time.perf_counter() < target_time:
                                pass
                        else:
                            # For longer waits, sleep most of the time then busy wait
                            time.sleep(wait_time - 0.005)  # Wake up 5ms early
                            # Then busy wait for the remaining time for precision
                            while time.perf_counter() < target_time:
                                pass

                    # Handle note on events
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Convert velocity from 0-127 to 0.0-1.0
                        velocity = msg.velocity / 127.0

                        # Start the note (duration will be handled by note_off)
                        osc = play_midi_note(msg.note, 0, velocity)

                        # Store oscillator reference for this channel and note
                        channel_key = (msg.channel, msg.note)
                        active_notes[channel_key] = osc

                    # Handle note off events (or note_on with velocity 0)
                    elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                        channel_key = (msg.channel, msg.note)

                        # If we have a reference to this note's oscillator, release it
                        if channel_key in active_notes:
                            with config.notes_lock:
                                # Mark the oscillator as released to start its release envelope
                                osc = active_notes[channel_key]
                                if osc.key in config.active_notes:
                                    config.active_notes[osc.key].released = True
                                    config.active_notes[osc.key].env_time = 0.0
                                    config.active_notes[osc.key].lfo_env_time = 0.0

                            # Remove from active notes
                            del active_notes[channel_key]

                # Mark playback as complete
                config.midi_playback_active = False

            except Exception as e:
                print(f"Error during MIDI playback: {e}")
                config.midi_playback_active = False

        # Start playback in a separate thread
        threading.Thread(target=play_midi_events, daemon=True).start()

    except Exception as e:
        print(f"Error playing MIDI file: {e}")
        config.midi_playback_active = False
