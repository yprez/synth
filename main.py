"""QWERTY Synth - A simple software synthesizer controlled by your keyboard."""

from qwerty_synth import synth
from qwerty_synth import input
from qwerty_synth import plot


def main():
    """Start the QWERTY Synth application."""
    print('QWERTY Synth: Play keys A–K, W,E,T,Y,U,O,P etc.')
    print('Use Z / X to shift octave down / up')
    print('Use 1–4 to switch waveform: 1=sine, 2=square, 3=triangle, 4=sawtooth')
    print('Use 5/6 for Attack, 7/8 for Decay, 9/0 for Sustain, -/= for Release')
    print('Press ESC to quit')

    # Create and start audio stream
    stream = synth.create_audio_stream()
    stream.start()

    # Start keyboard input handling
    input.start_keyboard_input()

    # Start visualization (blocking call)
    plot.plot_waveform()


if __name__ == "__main__":
    main()
