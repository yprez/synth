"""Keyboard input handling for QWERTY Synth."""

import threading
from pynput import keyboard
import sounddevice as sd
import matplotlib.pyplot as plt

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth


def on_press(key):
    """Handle key press events."""
    try:
        k = key.char.lower()

        if k in config.key_note_map:
            base_freq = config.key_note_map[k]
            freq = base_freq * (2 ** (config.octave_offset / 12))
            with config.notes_lock:
                if k not in config.active_notes:
                    osc = synth.Oscillator(freq, config.waveform_type)
                    config.active_notes[k] = osc

        elif k == 'z' and config.octave_offset > 12 * config.octave_min:
            config.octave_offset -= 12
            print(f'Octave down: {config.octave_offset // 12:+}')
        elif k == 'x' and config.octave_offset < 12 * config.octave_max:
            config.octave_offset += 12
            print(f'Octave up: {config.octave_offset // 12:+}')

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
            adsr.adsr['attack'] += step['attack']
            print(f"Attack: {adsr.adsr['attack']:.2f}s")
        elif k == '6':
            adsr.adsr['attack'] = max(0.001, adsr.adsr['attack'] - step['attack'])
            print(f"Attack: {adsr.adsr['attack']:.2f}s")
        elif k == '7':
            adsr.adsr['decay'] += step['decay']
            print(f"Decay: {adsr.adsr['decay']:.2f}s")
        elif k == '8':
            adsr.adsr['decay'] = max(0.01, adsr.adsr['decay'] - step['decay'])
            print(f"Decay: {adsr.adsr['decay']:.2f}s")
        elif k == '9':
            adsr.adsr['sustain'] = min(1.0, adsr.adsr['sustain'] + step['sustain'])
            print(f"Sustain: {adsr.adsr['sustain']:.2f}")
        elif k == '0':
            adsr.adsr['sustain'] = max(0.0, adsr.adsr['sustain'] - step['sustain'])
            print(f"Sustain: {adsr.adsr['sustain']:.2f}")
        elif k == '-':
            adsr.adsr['release'] += step['release']
            print(f"Release: {adsr.adsr['release']:.2f}s")
        elif k == '=':
            adsr.adsr['release'] = max(0.01, adsr.adsr['release'] - step['release'])
            print(f"Release: {adsr.adsr['release']:.2f}s")

        adsr.update_adsr_curve()

    except AttributeError:
        if key == keyboard.Key.esc:
            print('Exiting...')
            sd.stop()
            plt.close('all')
            return False


def on_release(key):
    """Handle key release events."""
    try:
        k = key.char.lower()
        with config.notes_lock:
            if k in config.active_notes:
                config.active_notes[k].released = True
                config.active_notes[k].env_time = 0.0
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
