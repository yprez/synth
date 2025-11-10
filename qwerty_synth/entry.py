"""Entry point for launching the QWERTY Synth GUI."""

import argparse

from qwerty_synth import gui_qt as gui


def main():
    """Parse command-line arguments and start the synthesizer GUI."""
    parser = argparse.ArgumentParser(description='QWERTY Synth - Software synthesizer')
    parser.add_argument('--midi', help='MIDI file to load on startup')
    parser.add_argument('--play', action='store_true', help='Automatically play the MIDI file')
    parser.add_argument('--patch', help='Patch name to load on startup')
    args = parser.parse_args()

    print('QWERTY Synth: Play keys A-K, W,E,T,Y,U,O,P etc.')
    print('Use Z / X to shift octave down / up')
    print('Press ESC to quit')

    if args.midi:
        print(f'MIDI file will be loaded: {args.midi}')
        if args.play:
            print('Auto-play enabled')

    if args.patch:
        print(f'Patch will be loaded: {args.patch}')

    print('Starting GUI...')

    gui.start_gui(midi_file=args.midi, auto_play=args.play, patch_name=args.patch)


if __name__ == "__main__":
    main()
