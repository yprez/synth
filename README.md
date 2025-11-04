# QWERTY Synth

> **Note:** This is a toy project I created to get better at AI assisted coding and is not intended for public use.

A minimalist real-time synthesizer built in Python using the keyboard as a piano.

## Features

- Play notes using keys A-K, W, E, T, Y, U, O, P, etc.
- Octave shift with `Z` (down) / `X` (up)
- MIDI controller input support for hardware MIDI devices
- GUI with additional features:
  - Filter controls (cutoff, resonance, envelope)
  - Drive effect for soft-clipping distortion
  - Delay effect with tempo sync and ping-pong mode
  - Reverb effect with configurable room size and mix
  - Chorus effect with configurable voices and modulation
  - LFO for creating vibrato, tremolo, or filter wobble effects
  - Arpeggiator for automatic note patterns
  - Step sequencer with customizable scales and patterns
  - Audio recording to WAV files (16-bit or 24-bit)
  - MIDI file browser and playback controls
- Real-time plots:
  - Waveform view
  - Frequency spectrum
  - ADSR envelope curve

## Requirements

- Python 3.10+
- Linux (tested on Ubuntu)

## Installation

Install dependencies using `uv`:

```bash
uv sync
```

## Running

Run the synth with:

```bash
uv run python main.py
```

Then press keys on the keyboard to play or use the GUI controls. Press ESC to quit.

## Testing

Run the test suite to verify everything is working correctly:

```bash
uv run pytest
```

## Usage tips

* You can tweak the envelope (ADSR) in real time while playing notes.
* Your system should support low-latency audio (e.g. ALSA on Linux).
* Ensure the keyboard layout matches QWERTY for accurate note mapping.
* The step sequencer allows you to create patterns in various musical scales.
* Use mono mode and portamento for lead sounds with smooth note transitions.


## TODO

* Additional waveform types (FM synthesis, wavetables)
* More filter types (comb, formant filters)

## Demo video
https://github.com/user-attachments/assets/6c35c888-a61f-4219-97a1-2ab8da18e066

