"""Keyboard-to-MIDI translator for the QWERTY Synth."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pynput import keyboard

from qwerty_synth import config

LOGGER = logging.getLogger(__name__)

DEFAULT_VELOCITY = 100
DEFAULT_CHANNEL = 0

MidiEventType = Literal['note_on', 'note_off', 'transpose', 'system_exit']
MidiPayload = dict[str, int | float | str]

# Default QWERTY-to-MIDI mapping (C-major layout).
DEFAULT_KEY_MIDI_MAP: dict[str, int] = {
    'a': 60,  # C4 (middle C)
    'w': 61,  # C#4
    's': 62,  # D4
    'e': 63,  # D#4
    'd': 64,  # E4
    'f': 65,  # F4
    't': 66,  # F#4
    'g': 67,  # G4
    'y': 68,  # G#4
    'h': 69,  # A4 (440Hz)
    'u': 70,  # A#4
    'j': 71,  # B4
    'k': 72,  # C5
    'o': 73,  # C#5
    'l': 74,  # D5
    'p': 75,  # D#5
    ';': 76,  # E5
    "'": 77,  # F5
}

CONTROL_KEY_OCTAVE_DOWN = 'z'
CONTROL_KEY_OCTAVE_UP = 'x'


@dataclass(frozen=True, slots=True)
class MidiEvent:
    """High-level MIDI-like event emitted by the keyboard translator."""

    event_type: MidiEventType
    note: int | None = None
    velocity: int = DEFAULT_VELOCITY
    channel: int = DEFAULT_CHANNEL
    timestamp: float = field(default_factory=time.time)
    payload: MidiPayload = field(default_factory=dict)


class MidiEventDispatcher(Protocol):
    """Protocol describing the callback invoked for each emitted event."""

    def __call__(self, event: MidiEvent) -> None:  # pragma: no cover - protocol signature
        ...


class KeyboardMidiTranslator:
    """Translate QWERTY keyboard events into MIDI-style messages."""

    def __init__(
        self,
        dispatcher: MidiEventDispatcher,
        *,
        velocity: int = DEFAULT_VELOCITY,
        channel: int = DEFAULT_CHANNEL,
        key_midi_map: dict[str, int] | None = None,
        listener_cls: Callable[..., keyboard.Listener] = keyboard.Listener,
    ) -> None:
        if not callable(dispatcher):
            raise TypeError('dispatcher must be callable')
        if not 0 <= velocity <= 127:
            raise ValueError('velocity must be within MIDI range 0-127')
        if not 0 <= channel <= 15:
            raise ValueError('channel must be within MIDI range 0-15')

        self._dispatcher = dispatcher
        self._velocity = velocity
        self._channel = channel
        self._key_midi_map = dict(key_midi_map) if key_midi_map is not None else dict(DEFAULT_KEY_MIDI_MAP)
        self._listener_cls = listener_cls

        self._listener: keyboard.Listener | None = None
        self._lock = threading.Lock()
        self._active_notes: dict[str, int] = {}
        self._held_controls: set[str] = set()

    def start(self) -> keyboard.Listener:
        """Start listening for keyboard input in a background thread."""
        if self._listener is not None:
            return self._listener

        listener = self._listener_cls(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        self._listener = listener
        return listener

    def stop(self) -> None:
        """Stop the keyboard listener and clear pending state."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

        with self._lock:
            self._active_notes.clear()
            self._held_controls.clear()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.esc:
            self._dispatch(MidiEvent(event_type='system_exit', velocity=0))
            return

        if (char := self._safe_char(key)) is None:
            return

        match char:
            case _ if char == CONTROL_KEY_OCTAVE_DOWN:
                self._handle_control_press(char, semitone_delta=-12)
            case _ if char == CONTROL_KEY_OCTAVE_UP:
                self._handle_control_press(char, semitone_delta=12)
            case _ if char in self._key_midi_map:
                self._handle_note_press(char)
            case _:
                return

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.esc:
            return

        if (char := self._safe_char(key)) is None:
            return

        match char:
            case _ if char in {CONTROL_KEY_OCTAVE_DOWN, CONTROL_KEY_OCTAVE_UP}:
                self._handle_control_release(char)
            case _:
                self._handle_note_release(char)

    def _handle_note_press(self, key_char: str) -> None:
        with self._lock:
            if key_char in self._active_notes:
                return
            if (note := self._compute_midi_note(key_char)) is None:
                return
            self._active_notes[key_char] = note

        self._dispatch(
            MidiEvent(
                event_type='note_on',
                note=note,
                velocity=self._velocity,
                channel=self._channel,
            )
        )

    def _handle_note_release(self, key_char: str) -> None:
        with self._lock:
            note = self._active_notes.pop(key_char, None)

        if note is not None:
            self._dispatch(
                MidiEvent(
                    event_type='note_off',
                    note=note,
                    velocity=0,
                    channel=self._channel,
                )
            )

    def _handle_control_press(self, control_char: str, *, semitone_delta: int) -> None:
        with self._lock:
            if control_char in self._held_controls:
                return
            self._held_controls.add(control_char)

        if not self._transpose_within_limits(semitone_delta):
            return

        payload: MidiPayload = {'delta': semitone_delta, 'source': 'keyboard'}
        self._dispatch(
            MidiEvent(
                event_type='transpose',
                velocity=0,
                channel=self._channel,
                payload=payload,
            )
        )

    def _handle_control_release(self, control_char: str) -> None:
        with self._lock:
            self._held_controls.discard(control_char)

    def _compute_midi_note(self, key_char: str) -> int | None:
        base_note = self._key_midi_map.get(key_char)
        if base_note is None:
            return None

        with config.notes_lock:
            octave_offset = config.octave_offset
            semitone_offset = config.semitone_offset

        midi_note = base_note + octave_offset + semitone_offset
        return max(0, min(127, midi_note))

    def _transpose_within_limits(self, semitone_delta: int) -> bool:
        with config.notes_lock:
            new_offset = config.octave_offset + semitone_delta
            minimum = 12 * config.octave_min
            maximum = 12 * config.octave_max

        return minimum <= new_offset <= maximum

    def _dispatch(self, event: MidiEvent) -> None:
        try:
            self._dispatcher(event)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception('Unhandled exception while dispatching MIDI event: %s', event)

    @staticmethod
    def _safe_char(key: keyboard.Key | keyboard.KeyCode) -> str | None:
        try:
            char = key.char
        except AttributeError:
            return None

        return char.lower() if char else None


__all__ = [
    'DEFAULT_KEY_MIDI_MAP',
    'KeyboardMidiTranslator',
    'MidiEvent',
    'MidiEventDispatcher',
]
