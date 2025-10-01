"""Tests for the keyboard-to-MIDI translator module."""

import pytest
from pynput.keyboard import Key, KeyCode

from qwerty_synth import config
from qwerty_synth.keyboard_midi import (
    DEFAULT_KEY_MIDI_MAP,
    KeyboardMidiTranslator,
    MidiEvent,
)


@pytest.fixture(autouse=True)
def reset_config_state():
    """Reset shared config state before each test."""
    config.octave_offset = 0
    config.semitone_offset = 0
    config.octave_min = -2
    config.octave_max = 3
    yield


def test_note_on_off_dispatch():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append, velocity=96, channel=2)

    translator._on_press(KeyCode.from_char('a'))
    translator._on_release(KeyCode.from_char('a'))

    assert [event.event_type for event in events] == ['note_on', 'note_off']
    assert events[0].note == DEFAULT_KEY_MIDI_MAP['a']
    assert events[1].note == DEFAULT_KEY_MIDI_MAP['a']
    assert events[0].velocity == 96
    assert events[0].channel == 2
    assert events[1].velocity == 0


def test_repeated_keydown_does_not_duplicate_events():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)

    key = KeyCode.from_char('s')
    translator._on_press(key)
    translator._on_press(key)
    translator._on_release(key)

    assert [event.event_type for event in events] == ['note_on', 'note_off']


def test_note_off_ignored_if_not_pressed():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)

    translator._on_release(KeyCode.from_char('d'))

    assert events == []


def test_octave_offset_applied_to_note():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)
    config.octave_offset = 12  # +1 octave

    translator._on_press(KeyCode.from_char('a'))
    translator._on_release(KeyCode.from_char('a'))

    assert events[0].note == DEFAULT_KEY_MIDI_MAP['a'] + 12
    assert events[1].note == DEFAULT_KEY_MIDI_MAP['a'] + 12


def test_control_key_transpose_dispatches_event_within_limits():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)

    translator._on_press(KeyCode.from_char('x'))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == 'transpose'
    assert event.payload['delta'] == 12
    assert event.payload['source'] == 'keyboard'


def test_control_key_respects_octave_limits():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)
    config.octave_offset = 12 * config.octave_max

    translator._on_press(KeyCode.from_char('x'))

    assert events == []


def test_control_key_release_clears_hold_state():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)

    key = KeyCode.from_char('z')
    translator._on_press(key)
    translator._on_release(key)
    translator._on_press(key)

    assert len(events) == 2  # Two valid transpose events from two distinct presses


def test_escape_key_emits_system_exit():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)

    translator._on_press(Key.esc)

    assert len(events) == 1
    assert events[0].event_type == 'system_exit'


def test_note_values_clamped_to_midi_range():
    events: list[MidiEvent] = []
    translator = KeyboardMidiTranslator(dispatcher=events.append)
    config.octave_offset = 100  # intentionally large to exceed MIDI range

    translator._on_press(KeyCode.from_char('k'))  # Highest mapped note
    translator._on_release(KeyCode.from_char('k'))

    assert events[0].note == 127
    assert events[1].note == 127


def test_custom_listener_is_respected():
    events: list[MidiEvent] = []

    class FakeListener:
        def __init__(self, *, on_press, on_release):
            self.on_press = on_press
            self.on_release = on_release
            self.started = False

        def start(self):
            self.started = True
            return self

        def stop(self):
            self.started = False

    translator = KeyboardMidiTranslator(
        dispatcher=events.append,
        listener_cls=FakeListener,
    )

    listener = translator.start()
    assert isinstance(listener, FakeListener)
    assert listener.started is True

    # Ensure start is idempotent
    assert translator.start() is listener

    translator.stop()
    assert listener.started is False
