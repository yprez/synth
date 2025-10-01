# Keyboard-to-MIDI Translator Plan

## Overview
- Build a dedicated module that listens to QWERTY keyboard events and emits MIDI-style note messages so downstream components can treat it like any other MIDI instrument.
- Decouple GUI and controller code from the current ad-hoc keyboard model to simplify future input modes (real MIDI controllers, automation, etc.).

## Design Goals
- Keep keyboard-to-note mapping explicit and easily swappable.
- Emit canonical MIDI `note_on`/`note_off` events with velocity and timestamp metadata.
- Maintain compatibility with mono/poly modes, arpeggiator, transpose, and global config locks.
- Provide a clean integration layer so the rest of the synth only handles MIDI events.

## Proposed Architecture
- `qwerty_synth/keyboard_midi.py` will host the new functionality.
- Core pieces:
  - `KeyboardMidiTranslator`: wraps `pynput.keyboard.Listener`, handles key state, octave/semitone shifts, and maps QWERTY keys to MIDI note numbers.
  - `MidiEvent`: lightweight dataclass bundling `type` (`note_on`/`note_off`), `note`, `velocity`, `channel`, and `timestamp`.
  - `MidiEventDispatcher`: pluggable callback (default: controller integration) invoked sequentially from translator threads while guarding shared state with `config.notes_lock`.
- Translator responsibilities:
  - Maintain pressed key set to avoid duplicate `note_on` spam and to support mono note priority logic before handing notes to the controller.
  - Emit transpose or mode-change commands by publishing control callbacks instead of directly mutating `config` (e.g., dedicated dispatcher hook or new `controller.apply_transpose_delta`).
  - Allow velocity overrides (initially constant) and expose hooks for future features (e.g., velocity from key press duration).
- Event flow: QWERTY key → translator produces MIDI event → dispatcher maps to `controller.handle_midi_message` → controller converts to oscillator lifecycle operations in `synth`.

## Integration Strategy
- Controller layer:
  - Add `handle_midi_message(message: MidiEvent)` to centralize `note_on`/`note_off` handling, reuse existing helper `play_midi_note` for note-on logic, and add a symmetric release path that operates via oscillator keys rather than raw characters.
  - Consolidate mono-mode tracking inside controller (possibly using a queue or `MonoVoiceManager`) so the translator no longer manipulates `config.active_notes` directly.
- GUI / application bootstrap:
  - Replace imports of `qwerty_synth.input` with the new translator.
  - Update GUI toggles (octave buttons, arpeggiator, mono mode) to call controller helpers or translator setters rather than mutating module globals.
  - Ensure controller provides functions the GUI can call when the user changes transpose so translator state stays in sync.
- Configuration adjustments:
  - Move `mono_pressed_keys` and any keyboard-specific artifacts into translator-owned state.
  - Keep `octave_offset`, `semitone_offset`, and locks in `config`, but expose controller helpers that the translator uses to read/write the values under lock.

## Legacy Keyboard Model Removal
- Delete `qwerty_synth/input.py` once the translator is integrated and all call sites are updated.
- Remove or rewrite tests in `tests/test_input.py` so they target the new translator module and controller message handling instead of direct oscillator creation.
- Strip unused globals (`gui_instance`, keyboard print statements) and ensure exit behavior (Esc key) is handled via dispatcher (e.g., produce a `system_exit` control event routed by the main loop).
- Verify no residual imports reference the old keyboard model (search in `main.py`, `controller.py`, `gui_qt.py`, examples, and tests).

## Testing & Validation
- Unit tests for translator:
  - Key-to-MIDI mapping integrity, including octave/semitone shifts and boundary checks.
  - Mono/poly behavior via injected mock dispatcher capturing emitted events.
  - Arpeggiator enablement path ensures appropriate events (possibly beat-tracked) are pushed.
- Integration tests:
  - Controller `handle_midi_message` generates/release oscillators correctly.
  - GUI smoke test ensuring translator bootstrap occurs without raising and exits cleanly when requested.
- Manual verification checklist:
  - Run synth and confirm QWERTY input still plays notes.
  - Confirm mono glide and octave switches function.
  - Validate arpeggiator receives held notes.

## Open Assumptions
- `pynput` remains our keyboard listener and is acceptable for the new module.
- We will keep using `mido` message semantics (note numbers 0–127, velocity 0–127 scaled internally) without introducing a third-party virtual MIDI device.
- Mono voice priority can stay "last pressed" for now; no need for configurable priority schemes during this refactor.
- Escape-to-exit behavior can be mediated through the dispatcher without additional UI prompts.
- Tests can be refactored in place without large fixture overhauls (current mocking strategy stays workable).

## Current Status
- ✅ Implemented `qwerty_synth/keyboard_midi.py` with `KeyboardMidiTranslator`, `MidiEvent`, transpose controls, system-exit signaling, duplicate key suppression, and safe config access.
- ✅ Added `tests/test_keyboard_midi.py` covering note-on/off emission, octave shifts, transpose limits, escape handling, MIDI range clamping, and listener lifecycle behaviour.

## Next Implementation Steps
- **Next Step — Integrate translator with controller and application entry points:**
  Build `controller.handle_midi_message` to translate `note_on`/`note_off` events into oscillator lifecycle calls (respecting mono/poly rules and arpeggiator hooks), wire transpose events through a controller helper that adjusts `config` under lock, update `main.py` and `gui_qt.py` to instantiate the translator with the controller dispatcher, and ensure the app shutdown path responds to the `system_exit` event.
- **Following Step — Update GUI/config pathways and automated tests:**
  Refactor GUI controls (octave buttons, mono toggles, arpeggiator state) to call the new controller helpers instead of mutating globals, migrate or replace `tests/test_input.py` to exercise the controller-facing API, and add integration coverage confirming the translator-controller chain behaves end-to-end.
- **Subsequent Step — Remove the legacy keyboard input module.**
