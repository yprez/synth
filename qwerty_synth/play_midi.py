#!/usr/bin/env python3
"""
Script to play MIDI files with qwerty_synth.
Usage: python -m qwerty_synth.play_midi midi_file.mid [--tempo TEMPO] [--volume VOLUME]
"""

import sys
import time
import argparse
from qwerty_synth.controller import play_midi_file, midi_file_to_sequence
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
        "--info", "-i",
        action="store_true",
        help="Display information about the MIDI file without playing it"
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

    # If info mode, just display MIDI file information
    if args.info:
        try:
            sequence = midi_file_to_sequence(args.midi_file)
            print(f"MIDI File: {args.midi_file}")
            print(f"Total notes: {len(sequence)}")
            if sequence:
                min_note = min(note for note, _, _ in sequence)
                max_note = max(note for note, _, _ in sequence)
                total_duration = sum(duration for _, duration, _ in sequence)
                print(f"Note range: {min_note} to {max_note}")
                print(f"Total duration: {total_duration:.2f} seconds")
            return 0
        except Exception as e:
            print(f"Error getting MIDI file info: {e}")
            return 1

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
