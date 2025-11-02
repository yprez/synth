"""Integration test for MIDI controller → controller flow."""

from unittest.mock import Mock, MagicMock, patch
import pytest

from qwerty_synth import config, controller
from qwerty_synth.midi_input import MidiPortTranslator


@pytest.fixture(autouse=True)
def reset_state():
    """Reset controller state before each test."""
    controller.reset_keyboard_state()
    config.mono_mode = False
    config.active_notes.clear()
    yield
    controller.reset_keyboard_state()


def test_midi_note_on_creates_oscillator():
    """Test that MIDI note_on from controller creates an oscillator."""
    events = []
    
    def capture_dispatcher(event):
        events.append(event)
        controller.handle_midi_message(event)
    
    mock_port = MagicMock()
    mock_port.__iter__ = Mock(return_value=iter([]))
    
    translator = MidiPortTranslator(dispatcher=capture_dispatcher, port_name='Test Port')
    
    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Test Port']):
        with patch('qwerty_synth.midi_input.mido.open_input', return_value=mock_port):
            translator.start()
            
            # Simulate a MIDI message
            midi_msg = Mock()
            midi_msg.type = 'note_on'
            midi_msg.note = 60
            midi_msg.velocity = 100
            midi_msg.channel = 0
            
            event = translator._translate_message(midi_msg)
            assert event is not None
            
            # Dispatch it
            controller.handle_midi_message(event)
            
            # Check that oscillator was created
            assert 'keyboard_60' in config.active_notes
            assert config.active_notes['keyboard_60'].velocity == pytest.approx(100 / 127.0, rel=1e-4)
            
            translator.stop()


def test_midi_note_off_releases_oscillator():
    """Test that MIDI note_off releases the oscillator."""
    events = []
    
    def capture_dispatcher(event):
        events.append(event)
        controller.handle_midi_message(event)
    
    mock_port = MagicMock()
    mock_port.__iter__ = Mock(return_value=iter([]))
    
    translator = MidiPortTranslator(dispatcher=capture_dispatcher, port_name='Test Port')
    
    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Test Port']):
        with patch('qwerty_synth.midi_input.mido.open_input', return_value=mock_port):
            translator.start()
            
            # Note on
            msg_on = Mock()
            msg_on.type = 'note_on'
            msg_on.note = 60
            msg_on.velocity = 100
            msg_on.channel = 0
            event_on = translator._translate_message(msg_on)
            controller.handle_midi_message(event_on)
            
            assert 'keyboard_60' in config.active_notes
            assert not config.active_notes['keyboard_60'].released
            
            # Note off
            msg_off = Mock()
            msg_off.type = 'note_off'
            msg_off.note = 60
            msg_off.velocity = 0
            msg_off.channel = 0
            event_off = translator._translate_message(msg_off)
            controller.handle_midi_message(event_off)
            
            assert config.active_notes['keyboard_60'].released is True
            
            translator.stop()


def test_midi_mono_mode_behavior():
    """Test that MIDI input respects mono mode."""
    config.mono_mode = True
    
    def dispatcher(event):
        controller.handle_midi_message(event)
    
    mock_port = MagicMock()
    mock_port.__iter__ = Mock(return_value=iter([]))
    
    translator = MidiPortTranslator(dispatcher=dispatcher, port_name='Test Port')
    
    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Test Port']):
        with patch('qwerty_synth.midi_input.mido.open_input', return_value=mock_port):
            translator.start()
            
            # First note
            msg1 = Mock()
            msg1.type = 'note_on'
            msg1.note = 60
            msg1.velocity = 100
            msg1.channel = 0
            event1 = translator._translate_message(msg1)
            controller.handle_midi_message(event1)
            
            assert 'mono' in config.active_notes
            first_osc = config.active_notes['mono']
            
            # Second note (should update the same oscillator)
            msg2 = Mock()
            msg2.type = 'note_on'
            msg2.note = 64
            msg2.velocity = 110
            msg2.channel = 0
            event2 = translator._translate_message(msg2)
            controller.handle_midi_message(event2)
            
            assert 'mono' in config.active_notes
            assert config.active_notes['mono'] is first_osc  # Same oscillator
            assert config.active_notes['mono'].target_freq != first_osc.freq  # But frequency changed
            
            translator.stop()
