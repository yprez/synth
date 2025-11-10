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

First, install the required system package:

```bash
sudo apt install portaudio19-dev
```

Then install Python dependencies using `uv`:

```bash
uv sync
```

## Running

Run the synth with:

```bash
uv run python main.py
```

Then press keys on the keyboard to play or use the GUI controls. Press ESC to quit.

## Command line options

```bash
uv run python main.py [OPTIONS]
```

- `--midi FILE` - Load MIDI file on startup
- `--play` - Auto-play the loaded MIDI file (requires `--midi`)
- `--patch NAME` - Load a saved patch preset

Example:
```bash
uv run python main.py --midi song.mid --play --patch "Bass"
```

## Input methods

The synth supports three distinct ways to create sound:

- **QWERTY Keyboard**: Play notes using computer keys A-K, W, E, T, Y, U, O, P, etc. (default)
- **MIDI Controller**: Connect external MIDI keyboard/controller hardware (enable in MIDI Input tab)
- **MIDI File Playback**: Play pre-recorded .mid files like a music player (use MIDI Player tab or `--midi` flag)

Note: MIDI controller input and MIDI file playback are separate features that can be used independently.

## Testing

Run the test suite to verify everything is working correctly:

```bash
PYNPUT_BACKEND=dummy QT_QPA_PLATFORM=offscreen uv run pytest
```

The environment variables enable headless mode for running tests without a display or X11 server.

## Usage tips

* You can tweak the envelope (ADSR) in real time while playing notes.
* Your system should support low-latency audio (e.g. ALSA on Linux).
* Ensure the keyboard layout matches QWERTY for accurate note mapping.
* The step sequencer allows you to create patterns in various musical scales.
* Use mono mode and portamento for lead sounds with smooth note transitions.


## TODO

* Auto-save/restore session on exit
* Step sequencer improvements (full screen mode, save/load patterns, longer patterns, more editing options)
* MIDI file loop function
* More command line options
* Additional waveform types (FM synthesis, wavetables)
* More filter types (comb, formant filters)
* MIDI input enhancements (sustain pedal, pitch bend, MIDI learn, etc.)

## Demo video
https://github.com/user-attachments/assets/6c35c888-a61f-4219-97a1-2ab8da18e066

