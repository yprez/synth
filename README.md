# QWERTY Synth

A minimalist real-time synthesizer built in Python using your keyboard as a piano. It includes:

- Polyphony (multiple notes at once)
- Waveform switching (sine, square, triangle, sawtooth)
- Octave shifting
- ADSR envelope control (with live tweaking)
- Multi-mode filter (low-pass, high-pass, band-pass, notch) with cutoff and resonance control
- Drive/distortion effect for adding warmth and grit
- Tempo-synced stereo delay effect
- LFO modulation for pitch, volume, and filter cutoff
- Step sequencer with scale and rhythm controls
- Live waveform, frequency spectrum, and ADSR curve visualization
- Patch management for saving and loading sound presets
- MIDI file playback with tempo control and progress tracking
- Audio recording to WAV files (16-bit or 24-bit)
- Chorus effect for creating richer sounds
- A graphical user interface

## Features

- Play notes using keys A–K, W, E, T, Y, U, O, P, etc.
- Octave shift with `Z` (down) / `X` (up)
- GUI with additional features:
  - Filter controls (cutoff, resonance, envelope)
  - Drive effect for soft-clipping distortion
  - Delay effect with tempo sync and ping-pong mode
  - LFO for creating vibrato, tremolo, or filter wobble effects
  - Step sequencer with customizable scales and patterns
  - Audio recording to WAV files (16-bit or 24-bit)
  - Chorus effect with configurable voices and modulation
  - MIDI file browser and playback controls
- Real-time plots:
  - Waveform view
  - Frequency spectrum
  - ADSR envelope curve

## Requirements

- Python 3.10+
- Linux (tested on Ubuntu)

### Install dependencies using `uv`:

```bash
uv sync
```

## Testing

Run the test suite to verify everything is working correctly:

```bash
uv run pytest
```

## Running

Run the synth with:

```bash
uv run python main.py
```

Then press keys on your keyboard to play or use the GUI controls.

### Exiting

* Press ESC to quit the program and close all windows gracefully.


### Notes

* You can tweak the envelope (ADSR) in real time while playing notes.
* Your system should support low-latency audio (e.g. ALSA on Linux).
* Ensure your keyboard layout matches QWERTY for accurate note mapping.
* The step sequencer allows you to create patterns in various musical scales.
* Use mono mode and portamento for lead sounds with smooth note transitions.


### TODO

* Real-time MIDI input support (hardware MIDI devices)
* Additional waveform types (FM synthesis, wavetables)
* More filter types (comb, formant filters)
* Reverb effect integration
