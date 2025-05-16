"""Keyboard input handling for QWERTY Synth."""

import threading
from pynput import keyboard
import sounddevice as sd

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth
from qwerty_synth import controller  # Import the controller module

# Add a reference to store the GUI instance
gui_instance = None

# MIDI note mapping (instead of direct frequency mapping)
key_midi_map = {
    'a': 60,  # C4 (middle C)
    'w': 61,  # C#4
    's': 62,  # D4
    'e': 63,  # D#4
    'd': 64,  # E4
    'f': 65,  # F4
    't': 66,  # F#4
    'g': 67,  # G4
    'y': 68,  # G#4
    'h': 69,  # A4 (440Hz)
    'u': 70,  # A#4
    'j': 71,  # B4
    'k': 72,  # C5
    'o': 73,  # C#5
    'l': 74,  # D5
    'p': 75,  # D#5
    ';': 76,  # E5
    "'": 77,  # F5
}


def on_press(key):
    """Handle key press events."""
    global gui_instance

    try:
        k = key.char.lower()

        if k in key_midi_map:
            midi_note = key_midi_map[k] + config.octave_offset
            freq = controller.midi_to_freq(midi_note)

            with config.notes_lock:
                # Track the key press for mono mode
                if k not in config.mono_pressed_keys:
                    config.mono_pressed_keys.append(k)

                if config.mono_mode:
                    # In mono mode, create or update oscillator directly
                    if 'mono' not in config.active_notes or config.active_notes['mono'].released:
                        # No current note or released note - create new oscillator
                        osc = synth.Oscillator(freq, config.waveform_type)
                        osc.key = 'mono'
                        config.active_notes['mono'] = osc
                    else:
                        # Update existing oscillator's target frequency
                        config.active_notes['mono'].target_freq = freq
                else:
                    # Polyphonic mode
                    if k not in config.active_notes:
                        osc = synth.Oscillator(freq, config.waveform_type)
                        osc.key = k
                        config.active_notes[k] = osc

        elif k == 'z' and config.octave_offset > 12 * config.octave_min:
            config.octave_offset -= 12
            print(f'Octave down: {config.octave_offset // 12:+}')
            # Update the GUI octave display if GUI instance exists
            if gui_instance is not None and gui_instance.running:
                gui_instance.octave_label.setText(f"{config.octave_offset // 12:+d}")
                # Update slider without triggering signals
                gui_instance.octave_slider.blockSignals(True)
                gui_instance.octave_slider.setValue(config.octave_offset // 12)
                gui_instance.octave_slider.blockSignals(False)
        elif k == 'x' and config.octave_offset < 12 * config.octave_max:
            config.octave_offset += 12
            print(f'Octave up: {config.octave_offset // 12:+}')
            # Update the GUI octave display if GUI instance exists
            if gui_instance is not None and gui_instance.running:
                gui_instance.octave_label.setText(f"{config.octave_offset // 12:+d}")
                # Update slider without triggering signals
                gui_instance.octave_slider.blockSignals(True)
                gui_instance.octave_slider.setValue(config.octave_offset // 12)
                gui_instance.octave_slider.blockSignals(False)

        elif k == '1':
            config.waveform_type = 'sine'
            print('Waveform: sine')
        elif k == '2':
            config.waveform_type = 'square'
            print('Waveform: square')
        elif k == '3':
            config.waveform_type = 'triangle'
            print('Waveform: triangle')
        elif k == '4':
            config.waveform_type = 'sawtooth'
            print('Waveform: sawtooth')

        step = adsr.get_adsr_parameter_steps()

        if k == '5':
            config.adsr['attack'] += step['attack']
            print(f"Attack: {config.adsr['attack']:.2f}s")
        elif k == '6':
            config.adsr['attack'] = max(0.001, config.adsr['attack'] - step['attack'])
            print(f"Attack: {config.adsr['attack']:.2f}s")
        elif k == '7':
            config.adsr['decay'] += step['decay']
            print(f"Decay: {config.adsr['decay']:.2f}s")
        elif k == '8':
            config.adsr['decay'] = max(0.01, config.adsr['decay'] - step['decay'])
            print(f"Decay: {config.adsr['decay']:.2f}s")
        elif k == '9':
            config.adsr['sustain'] = min(1.0, config.adsr['sustain'] + step['sustain'])
            print(f"Sustain: {config.adsr['sustain']:.2f}")
        elif k == '0':
            config.adsr['sustain'] = max(0.0, config.adsr['sustain'] - step['sustain'])
            print(f"Sustain: {config.adsr['sustain']:.2f}")
        elif k == '-':
            config.adsr['release'] += step['release']
            print(f"Release: {config.adsr['release']:.2f}s")
        elif k == '=':
            config.adsr['release'] = max(0.01, config.adsr['release'] - step['release'])
            print(f"Release: {config.adsr['release']:.2f}s")

        # Volume control
        elif k == '[':
            config.volume = max(0.0, config.volume - 0.05)
            print(f"Volume: {config.volume:.2f}")
        elif k == ']':
            config.volume = min(1.0, config.volume + 0.05)
            print(f"Volume: {config.volume:.2f}")

        adsr.update_adsr_curve()

    except AttributeError:
        if key == keyboard.Key.esc:
            print('Exiting...')
            sd.stop()

            # Close the GUI if it exists
            if gui_instance is not None:
                gui_instance.close()

            return False


def on_release(key):
    """Handle key release events."""
    try:
        k = key.char.lower()

        with config.notes_lock:
            # Remove key from mono_pressed_keys list if it exists
            if k in config.mono_pressed_keys:
                config.mono_pressed_keys.remove(k)

            if k in config.active_notes or (config.mono_mode and 'mono' in config.active_notes):
                if not config.mono_mode:
                    # Regular polyphonic mode - release the specific note
                    if k in config.active_notes:
                        config.active_notes[k].released = True
                        config.active_notes[k].env_time = 0.0
                        config.active_notes[k].lfo_env_time = 0.0  # Reset LFO envelope time
                else:
                    # Mono mode - handle differently
                    if not config.mono_pressed_keys:
                        # No keys pressed - release the mono oscillator
                        if 'mono' in config.active_notes:
                            config.active_notes['mono'].released = True
                            config.active_notes['mono'].env_time = 0.0
                            config.active_notes['mono'].lfo_env_time = 0.0  # Reset LFO envelope time
                    elif 'mono' in config.active_notes:
                        # Some keys still pressed - switch to the last pressed key
                        last_key = config.mono_pressed_keys[-1]
                        midi_note = key_midi_map[last_key] + config.octave_offset
                        freq = controller.midi_to_freq(midi_note)

                        # Update oscillator target frequency for glide to new note
                        osc = config.active_notes['mono']
                        osc.target_freq = freq
                        osc.key = last_key
                        osc.lfo_env_time = 0.0  # Reset LFO envelope time when switching notes in mono mode

    except AttributeError:
        pass


def run_keyboard_listener():
    """Start the keyboard listener thread."""
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def start_keyboard_input():
    """Start the keyboard input handling in a separate thread."""
    thread = threading.Thread(target=run_keyboard_listener, daemon=True)
    thread.start()
    return thread
