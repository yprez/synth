# QWERTY Synth

A minimalist real-time synthesizer built in Python using your keyboard as a piano. It includes:

- Polyphony (multiple notes at once)
- Waveform switching (sine, square, triangle, sawtooth)
- Octave shifting
- ADSR envelope control (with live tweaking)
- Low-pass filter with cutoff and resonance control
- Drive/distortion effect for adding warmth and grit
- Tempo-synced stereo delay effect
- LFO modulation for pitch, volume, and filter cutoff
- Step sequencer with scale and rhythm controls
- Live waveform, frequency spectrum, and ADSR curve visualization
- A graphical user interface

## Features

- Play notes using keys A–K, W, E, T, Y, U, O, P, etc.
- Octave shift with `Z` (down) / `X` (up)
- Switch waveform:
  - `1`: Sine
  - `2`: Square
  - `3`: Triangle
  - `4`: Sawtooth
- Adjust ADSR envelope live:
  - `5` / `6`: Attack increase/decrease
  - `7` / `8`: Decay increase/decrease
  - `9` / `0`: Sustain increase/decrease
  - `-` / `=`: Release increase/decrease
- Volume control:
  - `[`: Decrease volume
  - `]`: Increase volume
- GUI with additional features:
  - Filter controls (cutoff, resonance, envelope)
  - Drive effect for soft-clipping distortion
  - Delay effect with tempo sync and ping-pong mode
  - LFO for creating vibrato, tremolo, or filter wobble effects
  - Step sequencer with customizable scales and patterns
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

#### Or using pip:

```bash
pip install numpy sounddevice pynput matplotlib pyqt5 pyqtgraph
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

* MIDI input support
* Save/load patches
* Audio recording to WAV
