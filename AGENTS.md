# QWERTY Synth Development Guide

This repository contains a Python synthesizer and its accompanying tests.

## Commands
- Build/Install: `uv sync`
- Run app: `uv run python main.py`
- Lint: `uv run ruff check .`
- Run all tests: `PYNPUT_BACKEND=dummy QT_QPA_PLATFORM=offscreen uv run pytest`
- Run single test: `PYNPUT_BACKEND=dummy QT_QPA_PLATFORM=offscreen uv run pytest tests/test_file.py::TestClass::test_function -v`
- Test with coverage: `PYNPUT_BACKEND=dummy QT_QPA_PLATFORM=offscreen uv run pytest --cov=qwerty_synth --cov-report=html`

## Code Style
- Python 3.10+ required
- Line length: max 100 characters
- Indentation: 4 spaces
- Blank lines should contain no spaces
- Naming: `snake_case` for variables/functions, `PascalCase` for classes
- Docstrings: module-level and for all public classes/functions
- Imports: stdlib first, third-party next, local modules last
- Type hints encouraged
- Error handling: use try/except with specific exceptions
- Thread safety: use locks when modifying shared resources

## Programmatic Checks

1. Install dependencies with `uv sync`.
2. Ensure the system package `portaudio19-dev` is installed.
3. Run the linter using `uv run ruff check .` and fix issues when possible.
4. Execute the tests with `PYNPUT_BACKEND=dummy` and `QT_QPA_PLATFORM=offscreen`:
   `uv run pytest`.

## Project Structure

### Core Components
- `main.py` - Application entry point
- `qwerty_synth/` - Main package containing all synthesizer modules
  - `synth.py` - Core synthesizer engine with audio generation
  - `controller.py` - Main controller orchestrating all components
  - `gui_qt.py` - PyQt-based graphical user interface
  - `config.py` - Configuration management and settings
  - `patch.py` - Sound preset management (save/load patches)
  - `input.py` - Keyboard input handling and note mapping

### Audio Effects & Processing
- `adsr.py` - Attack/Decay/Sustain/Release envelope generator
- `filter.py` - Multi-mode filter (low-pass, high-pass, band-pass, notch)
- `drive.py` - Distortion/overdrive effect
- `delay.py` - Tempo-synced stereo delay effect
- `chorus.py` - Chorus effect for rich, layered sounds
- `lfo.py` - Low-frequency oscillator for modulation
- `arpeggiator.py` - Arpeggiator for automatic note patterns
- `step_sequencer.py` - Step sequencer with scale and rhythm controls
- `record.py` - Audio recording functionality (WAV export)

### Testing & Examples
- `tests/` - Comprehensive test suite for all modules
- `examples/` - Example scripts demonstrating various features
- `synth_performance_benchmark.py` - Performance benchmarking tool

### Key Features
- Real-time polyphonic synthesis with multiple waveforms
- Live audio visualization (waveform, spectrum, ADSR curves)
- MIDI file playback with tempo control
- Comprehensive effects chain (filter, drive, delay, chorus)
- Modulation system with LFO and envelope control
- Pattern sequencing and arpeggiator
- Patch management for saving/loading presets
- Audio recording to WAV files

## Tools
- Always run scripts with uv: `uv run python` or `uv run pytest`

## Pull Requests

Summaries in pull request bodies should briefly describe the implemented changes
and mention the result of running the test suite.