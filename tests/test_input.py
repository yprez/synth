"""Integration tests covering keyboard events routed through the controller."""

from types import SimpleNamespace
from unittest.mock import Mock, patch, call

import pytest

from qwerty_synth import config, controller
from qwerty_synth.keyboard_midi import MidiEvent


@pytest.fixture(autouse=True)
def reset_controller_state():
    """Ensure controller state is reset before and after each test."""
    controller.reset_keyboard_state()
    yield
    controller.reset_keyboard_state()


def make_event(event_type: str, note: int | None = None, velocity: int = 100, payload=None) -> MidiEvent:
    """Helper to build MIDI events with sensible defaults."""
    return MidiEvent(event_type=event_type, note=note, velocity=velocity, payload=payload or {})


@patch('qwerty_synth.controller.Oscillator')
@patch('qwerty_synth.controller.midi_to_freq', return_value=440.0)
def test_handle_midi_message_polyphonic_spawns_and_releases(mock_midi_to_freq, mock_oscillator):
    """Polyphonic mode should create an oscillator and release it on note off."""
    osc = Mock()
    osc.released = False
    osc.env_time = None
    osc.lfo_env_time = None
    mock_oscillator.return_value = osc

    controller.handle_midi_message(make_event('note_on', note=60, velocity=100))

    mock_midi_to_freq.assert_called_once_with(60)
    assert 'keyboard_60' in config.active_notes
    assert config.active_notes['keyboard_60'] is osc
    assert pytest.approx(osc.velocity, rel=1e-4) == 100 / 127.0

    controller.handle_midi_message(make_event('note_off', note=60, velocity=0))

    assert osc.released is True
    assert osc.env_time == 0.0
    assert osc.lfo_env_time == 0.0


@patch('qwerty_synth.controller.Oscillator')
@patch('qwerty_synth.controller.midi_to_freq')
def test_handle_midi_message_mono_maintains_last_pressed(mock_midi_to_freq, mock_oscillator):
    """Mono mode should glide to the last pressed note and release when all keys lift."""
    config.mono_mode = True
    mock_midi_to_freq.side_effect = [440.0, 554.37, 440.0]

    osc = Mock()
    osc.released = False
    osc.env_time = 0.0
    osc.lfo_env_time = 0.0
    osc.velocity = 0.0
    mock_oscillator.return_value = osc

    controller.handle_midi_message(make_event('note_on', note=60, velocity=100))
    assert config.mono_pressed_keys == [60]
    assert config.active_notes['mono'] is osc

    controller.handle_midi_message(make_event('note_on', note=64, velocity=90))
    assert config.mono_pressed_keys == [60, 64]
    assert osc.target_freq == 554.37
    assert osc.lfo_env_time == 0.0

    controller.handle_midi_message(make_event('note_off', note=64))
    assert config.mono_pressed_keys == [60]
    assert osc.target_freq == 440.0

    controller.handle_midi_message(make_event('note_off', note=60))
    assert config.mono_pressed_keys == []
    assert osc.released is True


@patch('qwerty_synth.controller.Oscillator')
@patch('qwerty_synth.controller._get_arpeggiator_module')
def test_handle_midi_message_arpeggiator_gate(mock_get_arpeggiator_module, mock_oscillator):
    """When sustain base is disabled, arpeggiator events should not spawn oscillators."""
    config.arpeggiator_enabled = True
    config.arpeggiator_sustain_base = False

    arp_instance = Mock()
    mock_get_arpeggiator_module.return_value = SimpleNamespace(arpeggiator_instance=arp_instance)

    controller.handle_midi_message(make_event('note_on', note=60, velocity=80))

    arp_instance.add_note.assert_called_once_with(60)
    mock_oscillator.assert_not_called()
    assert config.active_notes == {}

    controller.handle_midi_message(make_event('note_off', note=60))
    arp_instance.remove_note.assert_called_once_with(60)


def test_apply_transpose_delta_prints_and_bounds():
    """Transpose helper should respect configuration bounds and provide user feedback."""
    config.octave_min = -1
    config.octave_max = 1

    with patch('builtins.print') as mock_print:
        assert controller.apply_transpose_delta(12) is True
        assert controller.apply_transpose_delta(-12) is True
        assert controller.apply_transpose_delta(-12) is True
        assert mock_print.call_args_list == [
            call('Octave up: +1'),
            call('Octave down: +0'),
            call('Octave down: -1'),
        ]

    with patch('builtins.print') as mock_print:
        assert controller.apply_transpose_delta(-12) is False
        mock_print.assert_not_called()


@patch('qwerty_synth.controller.Oscillator')
@patch('qwerty_synth.controller.midi_to_freq', return_value=440.0)
def test_reset_keyboard_state_clears_tracking(mock_midi_to_freq, mock_oscillator):
    """Resetting the controller should drop pressed-note bookkeeping."""
    config.mono_mode = True
    osc = Mock()
    osc.released = False
    mock_oscillator.return_value = osc

    controller.handle_midi_message(make_event('note_on', note=60))
    assert config.mono_pressed_keys == [60]

    controller.reset_keyboard_state()
    assert config.mono_pressed_keys == []
    controller.handle_midi_message(make_event('note_on', note=60))
    mock_midi_to_freq.assert_called_with(60)


def test_handle_midi_message_unknown_event_noop():
    """Unknown event types should not raise errors."""
    controller.handle_midi_message(make_event('pitch_bend', note=None))
    # Simply ensure no exceptions and state remains untouched
    assert config.active_notes == {}
