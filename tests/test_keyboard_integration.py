"""Integration tests covering keyboard translator to controller flows."""

import pytest

# Skip entire module if sounddevice/PortAudio is not available
try:
    import sounddevice
except OSError:
    pytest.skip("PortAudio library not found", allow_module_level=True)

from unittest.mock import Mock

from pynput.keyboard import Key, KeyCode

from qwerty_synth import config, controller
from qwerty_synth.keyboard_midi import KeyboardMidiTranslator


@pytest.fixture(autouse=True)
def reset_keyboard_state():
    controller.reset_keyboard_state()
    yield
    controller.reset_keyboard_state()


class DummyListener:
    """Minimal listener stub to avoid threading during tests."""

    def __init__(self, *, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.started = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.started = False


def make_translator(dispatcher):
    return KeyboardMidiTranslator(dispatcher=dispatcher, listener_cls=DummyListener)


def test_translator_note_flow_polyphonic():
    translator = make_translator(controller.handle_midi_message)
    translator.start()

    translator._on_press(KeyCode.from_char('a'))
    assert 'keyboard_60' in config.active_notes

    translator._on_release(KeyCode.from_char('a'))
    assert config.active_notes['keyboard_60'].released is True

    translator.stop()


def test_translator_respects_transpose_controls():
    translator = make_translator(controller.handle_midi_message)
    translator.start()

    translator._on_press(KeyCode.from_char('x'))
    translator._on_release(KeyCode.from_char('x'))
    assert controller.get_octave_offset() == 12

    translator._on_press(KeyCode.from_char('z'))
    translator._on_release(KeyCode.from_char('z'))
    assert controller.get_octave_offset() == 0

    translator.stop()


def test_system_exit_event_dispatch():
    dispatcher = Mock()
    translator = make_translator(dispatcher)
    translator.start()

    translator._on_press(Key.esc)

    event = dispatcher.call_args[0][0]
    assert event.event_type == 'system_exit'
    assert event.velocity == 0

    translator.stop()
