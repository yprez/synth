"""Tests for the MIDI controller input translator module."""

from unittest.mock import Mock, patch, MagicMock
import pytest

from qwerty_synth.midi_input import MidiPortTranslator, list_midi_ports


@pytest.fixture
def mock_mido_message():
    """Create a mock mido Message."""
    def _make_message(msg_type, note=60, velocity=100, channel=0):
        msg = Mock()
        msg.type = msg_type
        msg.note = note
        msg.velocity = velocity
        msg.channel = channel
        return msg
    return _make_message


def test_list_midi_ports_success():
    """Test listing available MIDI ports."""
    with patch('qwerty_synth.midi_input.mido.get_input_names', return_value=['Port 1', 'Port 2']):
        ports = list_midi_ports()
        assert ports == ['Port 1', 'Port 2']


def test_list_midi_ports_failure():
    """Test listing MIDI ports when enumeration fails."""
    with patch('qwerty_synth.midi_input.mido.get_input_names', side_effect=OSError('No backend')):
        ports = list_midi_ports()
        assert ports == []


def test_translator_init_requires_callable_dispatcher():
    """Test that translator requires a callable dispatcher."""
    with pytest.raises(TypeError, match='dispatcher must be callable'):
        MidiPortTranslator(dispatcher='not-callable')


def test_translator_start_no_ports_available():
    """Test starting translator when no MIDI ports are available."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=[]):
        result = translator.start()
        assert result is False


def test_translator_start_auto_select_first_port():
    """Test auto-selecting first port when none specified."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    mock_port = MagicMock()
    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Port 1', 'Port 2']):
        with patch('qwerty_synth.midi_input.mido.open_input', return_value=mock_port) as mock_open:
            result = translator.start()
            assert result is True
            mock_open.assert_called_once_with('Port 1')


def test_translator_start_specific_port():
    """Test starting translator with specific port name."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher, port_name='Port 2')

    mock_port = MagicMock()
    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Port 1', 'Port 2']):
        with patch('qwerty_synth.midi_input.mido.open_input', return_value=mock_port) as mock_open:
            result = translator.start()
            assert result is True
            mock_open.assert_called_once_with('Port 2')


def test_translator_start_port_not_found():
    """Test starting translator with non-existent port."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher, port_name='Nonexistent')

    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Port 1']):
        result = translator.start()
        assert result is False


def test_translator_start_port_open_failure():
    """Test handling of port open failure."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Port 1']):
        with patch('qwerty_synth.midi_input.mido.open_input', side_effect=OSError('Device busy')):
            result = translator.start()
            assert result is False


def test_translator_stop_closes_port():
    """Test that stop closes the MIDI port."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    mock_port = MagicMock()
    with patch('qwerty_synth.midi_input.list_midi_ports', return_value=['Port 1']):
        with patch('qwerty_synth.midi_input.mido.open_input', return_value=mock_port):
            translator.start()
            translator.stop()
            mock_port.close.assert_called_once()


def test_translate_note_on_message(mock_mido_message):
    """Test translating a note_on MIDI message."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    msg = mock_mido_message('note_on', note=60, velocity=100, channel=0)
    event = translator._translate_message(msg)

    assert event is not None
    assert event.event_type == 'note_on'
    assert event.note == 60
    assert event.velocity == 100
    assert event.channel == 0


def test_translate_note_on_zero_velocity_as_note_off(mock_mido_message):
    """Test that note_on with velocity 0 becomes note_off."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    msg = mock_mido_message('note_on', note=60, velocity=0, channel=0)
    event = translator._translate_message(msg)

    assert event is not None
    assert event.event_type == 'note_off'
    assert event.note == 60
    assert event.velocity == 0


def test_translate_note_off_message(mock_mido_message):
    """Test translating a note_off MIDI message."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    msg = mock_mido_message('note_off', note=60, velocity=0, channel=0)
    event = translator._translate_message(msg)

    assert event is not None
    assert event.event_type == 'note_off'
    assert event.note == 60
    assert event.velocity == 0


def test_translate_unsupported_message(mock_mido_message):
    """Test that unsupported message types return None."""
    dispatcher = Mock()
    translator = MidiPortTranslator(dispatcher=dispatcher)

    msg = Mock()
    msg.type = 'control_change'
    event = translator._translate_message(msg)

    assert event is None


def test_dispatcher_exception_handling(mock_mido_message):
    """Test that dispatcher exceptions are caught and logged."""
    def failing_dispatcher(event):
        raise ValueError('Dispatcher error')

    translator = MidiPortTranslator(dispatcher=failing_dispatcher)
    msg = mock_mido_message('note_on', note=60, velocity=100, channel=0)
    event = translator._translate_message(msg)

    # Should not raise, just log
    translator._dispatch(event)
