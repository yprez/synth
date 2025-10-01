"""Keyboard-to-MIDI translator for the QWERTY Synth."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol, Set

from pynput import keyboard

from qwerty_synth import config

LOGGER = logging.getLogger(__name__)

DEFAULT_VELOCITY = 100
DEFAULT_CHANNEL = 0

# Default QWERTY-to-MIDI mapping (C-major layout).
DEFAULT_KEY_MIDI_MAP: Dict[str, int] = {
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


@dataclass(frozen=True)
class MidiEvent:
    """High-level MIDI-like event emitted by the keyboard translator."""

    event_type: str
    note: Optional[int] = None
    velocity: int = DEFAULT_VELOCITY
    channel: int = DEFAULT_CHANNEL
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, int | float | str] = field(default_factory=dict)


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
        key_midi_map: Optional[Dict[str, int]] = None,
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

        self._listener: Optional[keyboard.Listener] = None
        self._lock = threading.Lock()
        self._active_notes: Dict[str, int] = {}
        self._held_controls: Set[str] = set()

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

        char = self._safe_char(key)
        if char is None:
            return

        if char == CONTROL_KEY_OCTAVE_DOWN:
            self._handle_control_press(char, semitone_delta=-12)
            return

        if char == CONTROL_KEY_OCTAVE_UP:
            self._handle_control_press(char, semitone_delta=12)
            return

        if char not in self._key_midi_map:
            return

        self._handle_note_press(char)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.esc:
            return

        char = self._safe_char(key)
        if char is None:
            return

        if char in (CONTROL_KEY_OCTAVE_DOWN, CONTROL_KEY_OCTAVE_UP):
            self._handle_control_release(char)
            return

        self._handle_note_release(char)

    def _handle_note_press(self, key_char: str) -> None:
        dispatch_note: Optional[int] = None
        with self._lock:
            if key_char in self._active_notes:
                return

            note = self._compute_midi_note(key_char)
            if note is None:
                return

            self._active_notes[key_char] = note
            dispatch_note = note

        if dispatch_note is not None:
            self._dispatch(
                MidiEvent(
                    event_type='note_on',
                    note=dispatch_note,
                    velocity=self._velocity,
                    channel=self._channel,
                )
            )

    def _handle_note_release(self, key_char: str) -> None:
        dispatch_note: Optional[int] = None
        with self._lock:
            dispatch_note = self._active_notes.pop(key_char, None)

        if dispatch_note is not None:
            self._dispatch(
                MidiEvent(
                    event_type='note_off',
                    note=dispatch_note,
                    velocity=0,
                    channel=self._channel,
                )
            )

    def _handle_control_press(self, control_char: str, *, semitone_delta: int) -> None:
        should_dispatch = False
        with self._lock:
            if control_char in self._held_controls:
                return
            self._held_controls.add(control_char)
            should_dispatch = True

        if not should_dispatch:
            return

        if not self._transpose_within_limits(semitone_delta):
            return

        payload = {'delta': semitone_delta, 'source': 'keyboard'}
        self._dispatch(
            MidiEvent(
                event_type='transpose',
                note=None,
                velocity=0,
                channel=self._channel,
                payload=payload,
            )
        )

    def _handle_control_release(self, control_char: str) -> None:
        with self._lock:
            self._held_controls.discard(control_char)

    def _compute_midi_note(self, key_char: str) -> Optional[int]:
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

        if new_offset < minimum or new_offset > maximum:
            return False
        return True

    def _dispatch(self, event: MidiEvent) -> None:
        try:
            self._dispatcher(event)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception('Unhandled exception while dispatching MIDI event: %s', event)

    @staticmethod
    def _safe_char(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
        try:
            char = key.char
        except AttributeError:
            return None

        if char is None:
            return None

        return char.lower()


__all__ = [
    'DEFAULT_KEY_MIDI_MAP',
    'KeyboardMidiTranslator',
    'MidiEvent',
    'MidiEventDispatcher',
]
