"""Tests for arpeggiator functionality."""

import pytest
from unittest.mock import Mock, patch, call
from PyQt5.QtWidgets import QApplication

from qwerty_synth import config
from qwerty_synth.arpeggiator import Arpeggiator


@pytest.fixture
def app():
    """Create QApplication instance for testing Qt components."""
    try:
        app = QApplication([])
    except RuntimeError:
        # QApplication already exists
        app = QApplication.instance()
    return app


@pytest.fixture
def arpeggiator(app):
    """Create an arpeggiator instance for testing."""
    arp = Arpeggiator()
    # Enable test mode to bypass Qt threading for unit tests
    arp._test_mode = True
    return arp


class TestArpeggiatorInit:
    """Test arpeggiator initialization."""

    def test_initialization_from_config(self, arpeggiator):
        """Test that arpeggiator initializes with values from config."""
        # Set some config values
        config.arpeggiator_enabled = True
        config.arpeggiator_pattern = 'down'
        config.arpeggiator_rate = 140
        config.arpeggiator_gate = 0.6
        config.arpeggiator_octave_range = 2
        config.arpeggiator_sync_to_bpm = False

        # Create new arpeggiator
        arp = Arpeggiator()

        assert arp.enabled == True
        assert arp.pattern == 'down'
        assert arp.rate == 140
        assert arp.gate == 0.6
        assert arp.octave_range == 2
        assert arp.sync_to_bpm == False

    def test_initial_state(self, arpeggiator):
        """Test initial state of arpeggiator."""
        assert arpeggiator.held_notes == set()
        assert arpeggiator.note_order == []
        assert arpeggiator.current_sequence == []
        assert arpeggiator.sequence_position == 0
        assert arpeggiator.is_running == False


class TestArpeggiatorControls:
    """Test arpeggiator control functions."""

    def test_toggle_enabled(self, arpeggiator):
        """Test enabling/disabling arpeggiator."""
        # Initially disabled
        assert arpeggiator.enabled == False
        assert config.arpeggiator_enabled == False

        # Enable
        arpeggiator.toggle_enabled(True)
        assert arpeggiator.enabled == True
        assert config.arpeggiator_enabled == True

        # Disable
        arpeggiator.toggle_enabled(False)
        assert arpeggiator.enabled == False
        assert config.arpeggiator_enabled == False

    def test_update_pattern(self, arpeggiator):
        """Test updating arpeggio pattern."""
        arpeggiator.update_pattern('Down')
        assert arpeggiator.pattern == 'down'
        assert config.arpeggiator_pattern == 'down'

        arpeggiator.update_pattern('Up/Down')
        assert arpeggiator.pattern == 'up_down'
        assert config.arpeggiator_pattern == 'up_down'

    def test_update_rate(self, arpeggiator):
        """Test updating arpeggio rate."""
        arpeggiator.update_rate(150)
        assert arpeggiator.rate == 150
        assert config.arpeggiator_rate == 150

    def test_update_gate(self, arpeggiator):
        """Test updating gate time."""
        arpeggiator.update_gate(70)  # 70% = 0.7
        assert arpeggiator.gate == 0.7
        assert config.arpeggiator_gate == 0.7

    def test_update_octave_range(self, arpeggiator):
        """Test updating octave range."""
        arpeggiator.update_octave_range(3)
        assert arpeggiator.octave_range == 3
        assert config.arpeggiator_octave_range == 3

    def test_toggle_sync(self, arpeggiator):
        """Test BPM sync toggle."""
        arpeggiator.toggle_sync(False)
        assert arpeggiator.sync_to_bpm == False
        assert config.arpeggiator_sync_to_bpm == False

        arpeggiator.toggle_sync(True)
        assert arpeggiator.sync_to_bpm == True
        assert config.arpeggiator_sync_to_bpm == True

    def test_toggle_sustain_base(self, arpeggiator):
        """Test sustain base toggle."""
        arpeggiator.toggle_sustain_base(False)
        assert arpeggiator.sustain_base == False
        assert config.arpeggiator_sustain_base == False

        arpeggiator.toggle_sustain_base(True)
        assert arpeggiator.sustain_base == True
        assert config.arpeggiator_sustain_base == True


class TestArpeggiatorNoteManagement:
    """Test note addition and removal."""

    def test_add_note(self, arpeggiator):
        """Test adding notes to arpeggiator."""
        arpeggiator.add_note(60)  # C4
        assert 60 in arpeggiator.held_notes
        assert 60 in arpeggiator.note_order
        assert len(arpeggiator.current_sequence) > 0

        arpeggiator.add_note(64)  # E4
        assert 64 in arpeggiator.held_notes
        assert 64 in arpeggiator.note_order
        assert len(arpeggiator.held_notes) == 2

    def test_remove_note(self, arpeggiator):
        """Test removing notes from arpeggiator."""
        # Add some notes first
        arpeggiator.add_note(60)
        arpeggiator.add_note(64)

        # Remove one note
        arpeggiator.remove_note(60)
        assert 60 not in arpeggiator.held_notes
        assert 60 not in arpeggiator.note_order
        assert 64 in arpeggiator.held_notes

        # Remove last note
        arpeggiator.remove_note(64)
        assert len(arpeggiator.held_notes) == 0
        assert len(arpeggiator.note_order) == 0

    def test_duplicate_note_add(self, arpeggiator):
        """Test adding the same note twice."""
        arpeggiator.add_note(60)
        arpeggiator.add_note(60)  # Add same note again

        # Should only appear once
        assert len(arpeggiator.held_notes) == 1
        assert arpeggiator.note_order.count(60) == 1


class TestArpeggiatorSequences:
    """Test arpeggio sequence generation."""

    def test_up_pattern(self, arpeggiator):
        """Test ascending arpeggio pattern."""
        arpeggiator.pattern = 'up'
        arpeggiator.octave_range = 1
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4
        arpeggiator.add_note(67)  # G4

        expected = [60, 64, 67]  # C, E, G
        assert arpeggiator.current_sequence == expected

    def test_down_pattern(self, arpeggiator):
        """Test descending arpeggio pattern."""
        arpeggiator.pattern = 'down'
        arpeggiator.octave_range = 1
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4
        arpeggiator.add_note(67)  # G4

        expected = [67, 64, 60]  # G, E, C
        assert arpeggiator.current_sequence == expected

    def test_up_down_pattern(self, arpeggiator):
        """Test up/down arpeggio pattern."""
        arpeggiator.pattern = 'up_down'
        arpeggiator.octave_range = 1
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4

        expected = [60, 64, 60]  # C, E, C
        assert arpeggiator.current_sequence == expected

    def test_chord_pattern(self, arpeggiator):
        """Test chord pattern (all notes at once)."""
        arpeggiator.pattern = 'chord'
        arpeggiator.octave_range = 1
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4
        arpeggiator.add_note(67)  # G4

        # Chord pattern stores all notes as a single "chord" element
        assert len(arpeggiator.current_sequence) == 1
        assert set(arpeggiator.current_sequence[0]) == {60, 64, 67}

    def test_octaves_pattern(self, arpeggiator):
        """Test octaves pattern."""
        arpeggiator.pattern = 'octaves'
        arpeggiator.octave_range = 2
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4

        expected = [60, 72, 64, 76]  # C4, C5, E4, E5
        assert arpeggiator.current_sequence == expected

    def test_order_pattern(self, arpeggiator):
        """Test order played pattern."""
        arpeggiator.pattern = 'order'
        arpeggiator.octave_range = 1

        # Add notes in specific order
        arpeggiator.add_note(67)  # G4 first
        arpeggiator.add_note(60)  # C4 second
        arpeggiator.add_note(64)  # E4 third

        expected = [67, 60, 64]  # Order they were added
        assert arpeggiator.current_sequence == expected

    def test_multiple_octaves(self, arpeggiator):
        """Test multiple octave ranges."""
        arpeggiator.pattern = 'up'
        arpeggiator.octave_range = 2
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4

        expected = [60, 64, 72, 76]  # C4, E4, C5, E5
        assert arpeggiator.current_sequence == expected


class TestArpeggiatorTiming:
    """Test arpeggiator timing functions."""

    def test_update_timing_sync_mode(self, arpeggiator):
        """Test timing update in sync mode."""
        arpeggiator.sync_to_bpm = True
        config.bpm = 120

        # Mock the timer to check if interval is calculated correctly
        with patch.object(arpeggiator.step_timer, 'setInterval') as mock_set_interval:
            arpeggiator.is_running = True
            arpeggiator.update_timing()

            # 60000 / 120 / 4 = 125ms for 16th notes at 120 BPM
            mock_set_interval.assert_called_once_with(125)

    def test_update_timing_free_mode(self, arpeggiator):
        """Test timing update in free-running mode."""
        arpeggiator.sync_to_bpm = False
        arpeggiator.rate = 140

        with patch.object(arpeggiator.step_timer, 'setInterval') as mock_set_interval:
            arpeggiator.is_running = True
            arpeggiator.update_timing()

            # 60000 / 140 / 4 = 107ms for 16th notes at 140 BPM
            expected_interval = int(60000 / 140 / 4)
            mock_set_interval.assert_called_once_with(expected_interval)


class TestArpeggiatorPlayback:
    """Test arpeggiator playback functionality."""

    @patch('qwerty_synth.arpeggiator.play_midi_note_direct')
    def test_advance_arpeggio_up_pattern(self, mock_play_midi, arpeggiator):
        """Test advancing through up pattern."""
        arpeggiator.enabled = True
        arpeggiator.pattern = 'up'
        arpeggiator.octave_range = 1
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4

        # Advance through sequence
        arpeggiator.advance_arpeggio()
        mock_play_midi.assert_called_once()
        # First call should be C4
        assert mock_play_midi.call_args[0][0] == 60

        # Advance again
        mock_play_midi.reset_mock()
        arpeggiator.advance_arpeggio()
        # Second call should be E4
        assert mock_play_midi.call_args[0][0] == 64

    @patch('qwerty_synth.arpeggiator.play_midi_note_direct')
    def test_advance_arpeggio_chord_pattern(self, mock_play_midi, arpeggiator):
        """Test advancing through chord pattern."""
        arpeggiator.enabled = True
        arpeggiator.pattern = 'chord'
        arpeggiator.octave_range = 1
        arpeggiator.add_note(60)  # C4
        arpeggiator.add_note(64)  # E4
        arpeggiator.add_note(67)  # G4

        arpeggiator.advance_arpeggio()

        # Should play all three notes
        assert mock_play_midi.call_count == 3
        played_notes = [call[0][0] for call in mock_play_midi.call_args_list]
        assert set(played_notes) == {60, 64, 67}

    def test_start_stop_arpeggio(self, arpeggiator):
        """Test starting and stopping arpeggio."""
        arpeggiator.enabled = True
        arpeggiator.add_note(60)

        # Start arpeggio
        arpeggiator.start_arpeggio()
        assert arpeggiator.is_running == True

        # Stop arpeggio
        arpeggiator.stop_arpeggio()
        assert arpeggiator.is_running == False


class TestArpeggiatorUtilityFunctions:
    """Test utility functions."""

    def test_midi_to_note_name(self, arpeggiator):
        """Test MIDI note to name conversion."""
        assert arpeggiator.midi_to_note_name(60) == "C4"
        assert arpeggiator.midi_to_note_name(69) == "A4"
        assert arpeggiator.midi_to_note_name(72) == "C5"
        assert arpeggiator.midi_to_note_name(49) == "C#3"

    def test_clear_notes(self, arpeggiator):
        """Test clearing all notes."""
        arpeggiator.add_note(60)
        arpeggiator.add_note(64)
        arpeggiator.start_arpeggio()

        arpeggiator.clear_notes()

        assert len(arpeggiator.held_notes) == 0
        assert len(arpeggiator.note_order) == 0
        assert len(arpeggiator.current_sequence) == 0
        assert arpeggiator.is_running == False

    def test_sync_bpm_changed(self, arpeggiator):
        """Test BPM sync update."""
        arpeggiator.sync_to_bpm = True

        with patch.object(arpeggiator, 'update_timing') as mock_update_timing:
            arpeggiator.sync_bpm_changed(130)
            mock_update_timing.assert_called_once()

    def test_stop_cleanup(self, arpeggiator):
        """Test stop and cleanup."""
        arpeggiator.add_note(60)
        arpeggiator.start_arpeggio()

        arpeggiator.stop()

        assert arpeggiator.is_running == False
        assert len(arpeggiator.held_notes) == 0
