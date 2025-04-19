"""QWERTY Synth - A simple software synthesizer controlled by your keyboard."""

from qwerty_synth import gui_qt as gui


def main():
    """Start the QWERTY Synth application."""
    print('QWERTY Synth: Play keys A–K, W,E,T,Y,U,O,P etc.')
    print('Use Z / X to shift octave down / up')
    print('Use 1–4 to switch waveform: 1=sine, 2=square, 3=triangle, 4=sawtooth')
    print('Use 5/6 for Attack, 7/8 for Decay, 9/0 for Sustain, -/= for Release')
    print('Use [/] to decrease/increase volume')
    print('Press ESC to quit')
    print('Starting GUI...')

    # Start the GUI (which handles audio stream and keyboard input)
    gui.start_gui()


if __name__ == "__main__":
    main()
