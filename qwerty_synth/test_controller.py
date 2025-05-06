#!/usr/bin/env python3
"""Test script for the programmatic note API."""

import time
import signal
import threading
import sys

from qwerty_synth import synth
from qwerty_synth import config
from qwerty_synth.controller import play_note, play_midi_note, play_sequence


def main():
    """Run a demonstration of the programmatic API."""
    print("Starting QWERTY Synth Controller Test")

    # Create and start audio stream
    stream = synth.create_audio_stream()
    stream.start()

    # Set up signal handling
    def signal_handler(sig, frame):
        print("\nStopping...")
        stream.stop()
        stream.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Store original mono mode setting
        original_mono_mode = config.mono_mode

        # Ensure we're in poly mode for most examples
        if config.mono_mode:
            print("Disabling mono mode for examples (will restore original setting at the end)")
            config.mono_mode = False

        # Example 1: Play a single note (A4 = 440Hz)
        print("Playing A4 (440Hz) for 1 second...")
        play_note(440, 1.0)
        time.sleep(1.5)  # Wait for the note to complete (duration + a bit more)

        # Example 2: Play a MIDI note (C4 = MIDI note 60)
        print("Playing C4 (MIDI note 60) for 1 second...")
        play_midi_note(60, 1.0)
        time.sleep(1.5)

        # Example 3: Play a sequence of MIDI notes (C major scale)
        print("Playing C major scale...")
        scale = [(60 + i, 0.2) for i in [0, 2, 4, 5, 7, 9, 11, 12]]
        play_sequence(scale, interval=0.1)

        # Wait for the sequence to complete
        # Calculate total duration: sum of all note durations + intervals
        total_duration = sum(note[1] for note in scale) + (len(scale) - 1) * 0.05
        time.sleep(total_duration + 0.5)  # Add a small buffer

        # Example 4: Play a basic chord (C major)
        print("Playing C major chord...")
        # Start all notes at the same time
        play_midi_note(60, 2.0)  # C
        play_midi_note(64, 2.0)  # E
        play_midi_note(67, 2.0)  # G
        time.sleep(2.5)

        # Example 5: Play a more expressive chord with different velocities
        print("Playing C major chord with different velocities for each note...")
        play_midi_note(48, 3.0, 0.7)  # C3 - medium velocity (bass)
        play_midi_note(60, 3.0, 0.5)  # C4 - softer
        play_midi_note(64, 3.0, 0.6)  # E4 - medium
        play_midi_note(67, 3.0, 0.9)  # G4 - louder (to emphasize)
        play_midi_note(72, 3.0, 0.8)  # C5 - medium-loud
        time.sleep(3.5)

        # Example 6: Play a short melody with different velocities
        print("Playing a short melody with different velocities...")
        melody = [
            (69, 0.2, 1.0),    # A4
            (67, 0.2, 0.8),    # G4
            (65, 0.2, 0.6),    # F4
            (64, 0.2, 0.4),    # E4
            (62, 0.2, 0.2),    # D4
            (60, 0.6, 0.1),    # C4
        ]
        play_sequence(melody, interval=0.1)

        # Wait for the melody to complete
        total_duration = sum(note[1] for note in melody) + (len(melody) - 1) * 0.1
        time.sleep(total_duration + 0.5)

        # Example 7: Arpeggiated chord with overlapping notes
        print("\nPlaying an arpeggiated chord with overlapping notes...")
        play_midi_note(60, 2.0, 0.8)  # C4
        time.sleep(0.2)
        play_midi_note(64, 1.8, 0.8)  # E4
        time.sleep(0.2)
        play_midi_note(67, 1.6, 0.8)  # G4
        time.sleep(0.2)
        play_midi_note(72, 1.4, 0.9)  # C5
        time.sleep(2.0)

        # Example 8: Demo of mono vs. poly mode
        print("\nDemonstrating mono mode vs. poly mode:")

        # Test in poly mode
        config.mono_mode = False
        print("  Poly mode: Playing three consecutive notes (they should overlap)")
        play_midi_note(60, 1.0)  # C4
        time.sleep(0.3)
        play_midi_note(64, 7.0)  # E4
        time.sleep(0.3)
        play_midi_note(67, 5.0)  # G4
        time.sleep(1.5)

        # Test in mono mode
        config.mono_mode = True
        print("  Mono mode: Playing three consecutive notes (with glide between them)")
        play_midi_note(60, 1.0)  # C4
        time.sleep(0.3)
        play_midi_note(64, 1.0)  # E4
        time.sleep(0.3)
        play_midi_note(67, 1.0)  # G4
        time.sleep(1.5)

        # Restore original mono mode setting
        config.mono_mode = original_mono_mode
        print(f"Mono mode restored to: {'ON' if config.mono_mode else 'OFF'}")

        print("\nDemo completed! Press Ctrl+C to exit.")

        # Keep the script running until manually terminated
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Clean up
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
