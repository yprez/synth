# QWERTY Synth

A minimalist real-time synthesizer built in Python using your QWERTY keyboard as a piano. It includes:

- Polyphony (multiple notes at once)
 Waveform switching (sine, square, triangle, sawtooth)
- Octave shifting
- ADSR envelope control (with live tweaking)
- Live waveform, frequency spectrum, and ADSR curve visualization

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
pip install numpy sounddevice pynput matplotlib
```

## Running

Run the synth with:

```bash
uv run python main.py
```

Then press keys on your keyboard to play.

### Exiting

* Press ESC to quit the program and close all windows gracefully.


### Notes

* You can tweak the envelope (ADSR) in real time while playing notes.
* Your system should support low-latency audio (e.g. ALSA on Linux).
* Ensure your keyboard layout matches QWERTY for accurate note mapping.


### TODO

* MIDI input support
* Save/load patches
* GUI for envelope and waveform controls
* Audio recording to WAV
