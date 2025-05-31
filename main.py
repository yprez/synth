"""QWERTY Synth - A simple software synthesizer controlled by your keyboard."""

from qwerty_synth import gui_qt as gui


def main():
    """Start the QWERTY Synth application."""
    print('QWERTY Synth: Play keys A-K, W,E,T,Y,U,O,P etc.')
    print('Use Z / X to shift octave down / up')
    print('Press ESC to quit')
    print('Starting GUI...')

    # Start the GUI (which handles audio stream and keyboard input)
    gui.start_gui()


if __name__ == "__main__":
    main()
