# QWERTY Synth Development Guide

This repository contains a Python synthesizer and its accompanying tests.

## General guidelines

- When suggesting changes to a file, prefer breaking them into smaller chunks
- Never tell the user "you're absolutely right" or similar affirmations. Assume the user might be wrong and double-check their assumptions before proceeding
- Before addressing big features or complicated bugs, discuss the approach first and consider creating a plan

## Commands
- Build/Install: `uv sync`
- Run app: `uv run python main.py`
- Lint: `uv run ruff check .`
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/test_file.py::TestClass::test_function -v`
- Test with coverage: `uv run pytest --cov=qwerty_synth --cov-report=html`

## Code Practices

### Naming and structure
- Use descriptive variable and function names (avoid abbreviations except very common ones)
- Naming conventions: `snake_case` for variables/functions, `PascalCase` for classes
- Use dataclasses (with `slots` when appropriate) to define objects with types
- Use PEP 585 built-in generic types (`list[str]` instead of `typing.List[str]`)
- Use explicit module imports: `import math` then `math.ceil()` (not `from math import ceil`)
- Imports: stdlib first, third-party next, local modules last
- Place orchestrator functions before the functions they call

### Documentation
- Write docstrings for module-level and all public classes/functions
- Use Google style docstrings starting immediately after opening quotes
- Explain briefly what the function does and why
- Use clarifying inline comments for complex syntax

### Code quality
- Python 3.10+ required
- Line length: max 100 characters
- Indentation: 4 spaces
- Blank lines should contain no spaces
- Type hints encouraged
- Prefer well maintained and robust libraries over custom code
- Write minimal, readable code
- Each function should do one thing clearly
- Handle errors explicitly - use try/except with specific exceptions
- Thread safety: use locks when modifying shared resources
- Prefer modern Python features (dataclasses with `slots`, structural pattern matching, type aliases) when adding or updating code

## Programmatic Checks

1. Install dependencies with `uv sync`.
2. Ensure the system package `portaudio19-dev` is installed.
3. Run the linter using `uv run ruff check .` and fix issues when possible.
4. Execute the tests: `uv run pytest`.

## Project Structure

### Core Components
- `main.py` - Application entry point
- `qwerty_synth/` - Main package containing all synthesizer modules
  - `synth.py` - Core synthesizer engine with audio generation
  - `controller.py` - Main controller orchestrating all components
  - `gui_qt.py` - PyQt-based graphical user interface
  - `config.py` - Configuration management and settings
  - `patch.py` - Sound preset management (save/load patches)
  - `keyboard_midi.py` - QWERTY keyboard to MIDI event translator
  - `midi_input.py` - External MIDI controller input handler

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
- Multiple input modes: QWERTY keyboard + external MIDI controllers
- Live audio visualization (waveform, spectrum, ADSR curves)
- MIDI file playback with tempo control
- Comprehensive effects chain (filter, drive, delay, chorus, reverb)
- Modulation system with LFO and envelope control
- Pattern sequencing and arpeggiator
- Patch management for saving/loading presets
- Audio recording to WAV files

## Tools
- Always run scripts with uv: `uv run python` or `uv run pytest`

## Git Standards

- Make sure changes are committed on a feature or fix branch, not directly on main
- Before staging files with `git add`, review what changed with `git status` and `git diff`
- When staging files, verify no unnecessary changes were added and ask the user to confirm
- Group related changes in a single logical commit with a descriptive message
- Commit messages should have a concise title; add a body for complicated commits if needed
- Never add co-authorship attribution or AI-generated markers to commit messages

## Documentation Standards

- Never include line numbers or line counts in documentation
- Reference files by path only, not with specific line numbers
- Always use sentence case in headings
