#!/usr/bin/env python3
"""
Script to play MIDI files with qwerty_synth.
Usage: python -m qwerty_synth.play_midi midi_file.mid [--tempo TEMPO] [--volume VOLUME]
"""

import sys
import time
import argparse
from qwerty_synth.controller import play_midi_file
from qwerty_synth.synth import create_audio_stream
from qwerty_synth import config


def main():
    """Run the MIDI file player."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Play MIDI files using qwerty_synth synthesizer",
        prog="python -m qwerty_synth.play_midi"
    )

    # Add arguments
    parser.add_argument(
        "midi_file",
        help="Path to the MIDI file to play"
    )
    parser.add_argument(
        "--tempo", "-t",
        type=float,
        default=1.0,
        help="Tempo scale factor (default: 1.0, half speed: 0.5, double speed: 2.0)"
    )
    parser.add_argument(
        "--volume", "-v",
        type=float,
        default=0.5,
        help="Output volume (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--waveform", "-w",
        choices=["sine", "square", "triangle", "sawtooth"],
        default=None,
        help="Set the oscillator waveform type"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set global volume
    config.volume = max(0.0, min(1.0, args.volume))

    # If waveform is specified, set it
    if args.waveform:
        config.waveform_type = args.waveform

    # Create and start audio stream
    stream = create_audio_stream()
    stream.start()

    print(f"Playing MIDI file: {args.midi_file}")
    print(f"Settings: tempo={args.tempo}, volume={args.volume}, waveform={config.waveform_type}")

    # Play the MIDI file
    play_midi_file(args.midi_file, args.tempo)

    # Wait while playback happens (in another thread)
    try:
        print("Press Ctrl+C to stop playback...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping playback...")
    finally:
        # Close the audio stream
        stream.stop()
        stream.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
