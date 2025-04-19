"""Keyboard input handling for QWERTY Synth."""

import threading
from pynput import keyboard
import sounddevice as sd

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth

# Add a reference to store the GUI instance
gui_instance = None


def on_press(key):
    """Handle key press events."""
    global gui_instance

    try:
        k = key.char.lower()

        if k in config.key_note_map:
            base_freq = config.key_note_map[k]
            freq = base_freq * (2 ** (config.octave_offset / 12))

            with config.notes_lock:
                # Track the key press for mono mode
                if k not in config.mono_pressed_keys:
                    config.mono_pressed_keys.append(k)

                if config.mono_mode:
                    # In mono mode, we only keep one oscillator
                    if 'mono' in config.active_notes:
                        # Update existing oscillator's target frequency for glide
                        osc = config.active_notes['mono']
                        osc.target_freq = freq
                        osc.key = k

                        # If oscillator was released, un-release it
                        if osc.released:
                            osc.released = False
                            osc.env_time = 0.0  # Reset envelope time to restart attack
                            osc.lfo_env_time = 0.0  # Reset LFO envelope time to restart delay/attack
                    else:
                        # Create a new oscillator for the mono voice
                        osc = synth.Oscillator(freq, config.waveform_type)
                        osc.key = k
                        config.active_notes['mono'] = osc
                else:
                    # Polyphonic mode - normal behavior
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

        # Volume control
        elif k == '[':
            config.volume = max(0.0, config.volume - 0.05)
            print(f"Volume: {config.volume:.2f}")
        elif k == ']':
            config.volume = min(1.0, config.volume + 0.05)
            print(f"Volume: {config.volume:.2f}")

        # Toggle mono mode
        elif k == 'm':
            config.mono_mode = not config.mono_mode
            print(f"Mono mode: {'ON' if config.mono_mode else 'OFF'}")
            with config.notes_lock:
                config.active_notes.clear()  # Clear notes to prevent stuck notes
                config.mono_pressed_keys.clear()  # Clear pressed keys

            # Update the GUI checkbox if GUI instance exists
            if gui_instance is not None and gui_instance.running:
                gui_instance.mono_checkbox.setChecked(config.mono_mode)

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
                        base_freq = config.key_note_map[last_key]
                        freq = base_freq * (2 ** (config.octave_offset / 12))

                        # Update oscillator target frequency for glide to new note
                        osc = config.active_notes['mono']
                        osc.target_freq = freq
                        osc.key = last_key

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
