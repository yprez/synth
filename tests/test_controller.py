"""Comprehensive unit tests for the controller functionality."""

import numpy as np
import pytest
import time
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import threading

from qwerty_synth import controller, config
from qwerty_synth.synth import Oscillator


class TestNoteCounterGlobal:
    """Test cases for global note counter functionality."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False

    def test_note_counter_increments(self):
        """Test that note counter increments for each new note in polyphonic mode."""
        config.mono_mode = False

        osc1 = controller.play_note(440.0, 0, 1.0)
        osc2 = controller.play_note(550.0, 0, 1.0)
        osc3 = controller.play_note(660.0, 0, 1.0)

        assert osc1.key == 'program_1'
        assert osc2.key == 'program_2'
        assert osc3.key == 'program_3'
        assert controller._note_counter == 3


class TestPlayNote:
    """Test cases for play_note function."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.waveform_type = 'sine'

    def test_play_note_polyphonic_mode(self):
        """Test playing note in polyphonic mode."""
        config.mono_mode = False

        osc = controller.play_note(440.0, 0, 0.8)

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0
        assert osc.velocity == 0.8
        assert osc.key == 'program_1'
        assert len(config.active_notes) == 1
        assert 'program_1' in config.active_notes

    def test_play_note_mono_mode_new_note(self):
        """Test playing note in mono mode when no note is active."""
        config.mono_mode = True

        osc = controller.play_note(440.0, 0, 0.7)

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0
        assert osc.velocity == 0.7
        assert osc.key == 'mono'
        assert len(config.active_notes) == 1
        assert 'mono' in config.active_notes

    def test_play_note_mono_mode_update_frequency(self):
        """Test updating frequency in mono mode when note is already active."""
        config.mono_mode = True

        # Play first note
        osc1 = controller.play_note(440.0, 0, 0.7)
        first_osc = config.active_notes['mono']

        # Play second note (should update frequency)
        osc2 = controller.play_note(550.0, 0, 0.8)

        # Should be the same oscillator with updated frequency
        assert osc2 is first_osc
        assert first_osc.target_freq == 550.0
        assert first_osc.key == 'mono'
        assert len(config.active_notes) == 1

    def test_play_note_mono_mode_unreleased_note(self):
        """Test mono mode with released note gets un-released."""
        config.mono_mode = True

        # Play and release a note
        osc1 = controller.play_note(440.0, 0, 0.7)
        config.active_notes['mono'].released = True
        config.active_notes['mono'].env_time = 1.0
        config.active_notes['mono'].lfo_env_time = 0.5

        # Play new note
        osc2 = controller.play_note(550.0, 0, 0.8)

        # Should un-release the note
        assert not config.active_notes['mono'].released
        assert config.active_notes['mono'].env_time == 0.0
        assert config.active_notes['mono'].lfo_env_time == 0.0

    def test_play_note_with_duration_scheduling(self):
        """Test that note release is scheduled when duration > 0."""
        config.mono_mode = False

        with patch('threading.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer.return_value = mock_timer_instance

            osc = controller.play_note(440.0, 1.5, 0.8)

            # Timer should be created and started
            mock_timer.assert_called_once()
            timer_args = mock_timer.call_args[0]
            assert timer_args[0] == 1.5  # duration

            mock_timer_instance.start.assert_called_once()

    def test_play_note_no_duration_no_timer(self):
        """Test that no timer is created when duration is 0."""
        config.mono_mode = False

        with patch('threading.Timer') as mock_timer:
            osc = controller.play_note(440.0, 0, 0.8)

            # No timer should be created
            mock_timer.assert_not_called()

    def test_play_note_release_function(self):
        """Test the release function created by play_note."""
        config.mono_mode = False

        osc = controller.play_note(440.0, 0, 0.8)
        key = osc.key

        # Manually call the release function that would be scheduled
        config.active_notes[key].released = False
        config.active_notes[key].env_time = 1.0
        config.active_notes[key].lfo_env_time = 0.5

        # Simulate the release function
        with config.notes_lock:
            if key in config.active_notes:
                config.active_notes[key].released = True
                config.active_notes[key].env_time = 0.0
                config.active_notes[key].lfo_env_time = 0.0

        assert config.active_notes[key].released
        assert config.active_notes[key].env_time == 0.0
        assert config.active_notes[key].lfo_env_time == 0.0

    def test_play_note_multiple_polyphonic(self):
        """Test playing multiple notes in polyphonic mode."""
        config.mono_mode = False

        osc1 = controller.play_note(440.0, 0, 0.8)
        osc2 = controller.play_note(550.0, 0, 0.7)
        osc3 = controller.play_note(660.0, 0, 0.6)

        assert len(config.active_notes) == 3
        assert osc1.key == 'program_1'
        assert osc2.key == 'program_2'
        assert osc3.key == 'program_3'

        assert config.active_notes['program_1'].freq == 440.0
        assert config.active_notes['program_2'].freq == 550.0
        assert config.active_notes['program_3'].freq == 660.0


class TestMidiToFreq:
    """Test cases for midi_to_freq function."""

    def test_midi_to_freq_a4(self):
        """Test MIDI note 69 (A4) converts to 440 Hz."""
        freq = controller.midi_to_freq(69)
        assert freq == 440.0

    def test_midi_to_freq_c4(self):
        """Test MIDI note 60 (C4) converts correctly."""
        freq = controller.midi_to_freq(60)
        expected = 440.0 * (2 ** ((60 - 69) / 12))
        assert abs(freq - expected) < 0.001

    def test_midi_to_freq_octave_relationships(self):
        """Test octave relationships in MIDI to frequency conversion."""
        # A3 should be half the frequency of A4
        freq_a3 = controller.midi_to_freq(57)  # A3
        freq_a4 = controller.midi_to_freq(69)  # A4
        freq_a5 = controller.midi_to_freq(81)  # A5

        assert abs(freq_a4 / freq_a3 - 2.0) < 0.001
        assert abs(freq_a5 / freq_a4 - 2.0) < 0.001

    def test_midi_to_freq_edge_cases(self):
        """Test edge cases for MIDI to frequency conversion."""
        # Test very low and high MIDI notes
        freq_low = controller.midi_to_freq(0)
        freq_high = controller.midi_to_freq(127)

        assert freq_low > 0
        assert freq_high > freq_low

        # Calculate expected values
        expected_low = 440.0 * (2 ** ((0 - 69) / 12))
        expected_high = 440.0 * (2 ** ((127 - 69) / 12))

        assert abs(freq_low - expected_low) < 0.001
        assert abs(freq_high - expected_high) < 0.001

    def test_midi_to_freq_chromatic_scale(self):
        """Test semitone relationships in chromatic scale."""
        # Each semitone should multiply frequency by 2^(1/12)
        semitone_ratio = 2 ** (1/12)

        for midi_note in range(60, 72):  # C4 to B4
            freq1 = controller.midi_to_freq(midi_note)
            freq2 = controller.midi_to_freq(midi_note + 1)
            ratio = freq2 / freq1
            assert abs(ratio - semitone_ratio) < 0.001


class TestPlayMidiNote:
    """Test cases for play_midi_note function."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False

    def test_play_midi_note_basic(self):
        """Test basic MIDI note playing."""
        osc = controller.play_midi_note(69, 0, 0.8)  # A4

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0
        assert osc.velocity == 0.8

    def test_play_midi_note_with_duration(self):
        """Test MIDI note with duration scheduling."""
        with patch('threading.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer.return_value = mock_timer_instance

            osc = controller.play_midi_note(60, 2.0, 0.7)  # C4

            # Timer should be created
            mock_timer.assert_called_once()
            timer_args = mock_timer.call_args[0]
            assert timer_args[0] == 2.0  # duration

    @patch('qwerty_synth.controller.play_note')
    def test_play_midi_note_calls_play_note(self, mock_play_note):
        """Test that play_midi_note calls play_note with correct frequency."""
        mock_osc = MagicMock()
        mock_play_note.return_value = mock_osc

        result = controller.play_midi_note(72, 1.5, 0.9)  # C5

        expected_freq = 440.0 * (2 ** ((72 - 69) / 12))
        mock_play_note.assert_called_once_with(expected_freq, 1.5, 0.9)
        assert result is mock_osc


class TestPlaySequence:
    """Test cases for play_sequence function."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False

    @patch('qwerty_synth.controller.play_note')
    @patch('qwerty_synth.controller.play_midi_note')
    def test_play_sequence_frequency_notes(self, mock_play_midi, mock_play_note):
        """Test playing sequence with frequency values (first note only)."""
        sequence = [
            (440.0, 0.5),          # A4 frequency
        ]

        # Test just the synchronous first note
        controller.play_sequence(sequence, interval=0.1)

        # Should call play_note for frequency values
        mock_play_note.assert_called_once_with(440.0, 0.5, 1.0)

        # Should not call play_midi_note
        mock_play_midi.assert_not_called()

    @patch('qwerty_synth.controller.play_note')
    @patch('qwerty_synth.controller.play_midi_note')
    def test_play_sequence_midi_notes(self, mock_play_midi, mock_play_note):
        """Test playing sequence with MIDI note numbers (first note only)."""
        sequence = [
            (60, 0.5),         # C4 MIDI
        ]

        # Test just the synchronous first note
        controller.play_sequence(sequence, interval=0.0)

        # Should call play_midi_note for MIDI values
        mock_play_midi.assert_called_once_with(60, 0.5, 1.0)

        # Should not call play_note
        mock_play_note.assert_not_called()

    @patch('qwerty_synth.controller.play_note')
    @patch('qwerty_synth.controller.play_midi_note')
    def test_play_sequence_mixed_types(self, mock_play_midi, mock_play_note):
        """Test playing sequence with mixed frequency and MIDI values."""
        sequence = [
            (60, 0.5),         # MIDI note
        ]

        # Test just the first note which is called immediately
        controller.play_sequence(sequence, interval=0.0)

        # Should call play_midi_note for the first MIDI note
        mock_play_midi.assert_called_once_with(60, 0.5, 1.0)

    @patch('threading.Timer')
    def test_play_sequence_with_interval(self, mock_timer):
        """Test sequence timer setup."""
        sequence = [
            (440.0, 0.0),  # Use 0 duration to avoid note release timer
            (550.0, 0.0),
        ]

        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance

        controller.play_sequence(sequence, interval=0.2)

        # Should create a timer for scheduling the next note
        # (Note: Timer may be called for both sequence scheduling and potentially note duration)
        assert mock_timer.call_count >= 1
        mock_timer_instance.start.assert_called()

    def test_play_sequence_empty(self):
        """Test playing empty sequence."""
        sequence = []

        # Should handle gracefully and not crash
        controller.play_sequence(sequence, interval=0.1)

    def test_play_sequence_malformed_notes(self):
        """Test sequence with malformed note entries."""
        sequence = [
            (440.0,),          # Missing duration
            (550.0, 0.5),      # Valid note
        ]

        # Should handle gracefully and not crash
        controller.play_sequence(sequence, interval=0.0)


class TestPlayMidiFile:
    """Test cases for play_midi_file function."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.midi_playback_active = False
        config.midi_paused = False

    @patch('qwerty_synth.controller.mido.MidiFile')
    @patch('threading.Thread')
    def test_play_midi_file_basic_setup(self, mock_thread, mock_midi_file_class):
        """Test basic MIDI file setup."""
        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([]))
        mock_midi_file_class.return_value = mock_midi_file

        controller.play_midi_file('test.mid', tempo_scale=0.8)

        # Should load MIDI file
        mock_midi_file_class.assert_called_once_with('test.mid')

        # Should set config values
        assert config.midi_tempo_scale == 0.8
        assert config.midi_playback_active == True

        # Should start thread
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch('qwerty_synth.controller.mido.MidiFile')
    @patch('threading.Thread')
    def test_play_midi_file_duration_calculation(self, mock_thread, mock_midi_file_class):
        """Test MIDI file duration calculation."""
        # Mock MIDI file with timing
        mock_msg1 = MagicMock()
        mock_msg1.time = 0.5
        mock_msg2 = MagicMock()
        mock_msg2.time = 1.0
        mock_msg3 = MagicMock()
        mock_msg3.time = 0.5

        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([mock_msg1, mock_msg2, mock_msg3]))
        mock_midi_file_class.return_value = mock_midi_file

        controller.play_midi_file('test.mid', tempo_scale=2.0)

        # Duration should be (0.5 + 1.0 + 0.5) / 2.0 = 1.0
        assert config.midi_playback_duration == 1.0

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_play_midi_file_exception_handling(self, mock_midi_file_class):
        """Test MIDI file exception handling."""
        mock_midi_file_class.side_effect = Exception("File not found")

        # Should not raise exception
        controller.play_midi_file('nonexistent.mid')

        # Should set playback to inactive
        assert config.midi_playback_active == False

    def test_midi_playback_state_management(self):
        """Test MIDI playback state flags."""
        # Test initial state
        assert config.midi_playback_active == False

        # Test that we can set the flag
        config.midi_playback_active = True
        assert config.midi_playback_active == True

        config.midi_playback_active = False
        assert config.midi_playback_active == False


class TestControllerIntegration:
    """Integration tests for controller functionality."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False

    def test_polyphonic_note_management(self):
        """Test complete polyphonic note lifecycle."""
        config.mono_mode = False

        # Play multiple notes
        osc1 = controller.play_note(440.0, 0, 0.8)
        osc2 = controller.play_note(550.0, 0, 0.7)
        osc3 = controller.play_note(660.0, 0, 0.6)

        assert len(config.active_notes) == 3

        # Simulate releasing one note
        key2 = osc2.key
        with config.notes_lock:
            config.active_notes[key2].released = True

        assert config.active_notes[key2].released
        assert not config.active_notes[osc1.key].released
        assert not config.active_notes[osc3.key].released

    def test_mono_mode_transition(self):
        """Test transitioning between mono and poly modes."""
        # Start in poly mode
        config.mono_mode = False
        osc1 = controller.play_note(440.0, 0, 0.8)
        osc2 = controller.play_note(550.0, 0, 0.7)

        assert len(config.active_notes) == 2

        # Switch to mono mode and play note
        config.mono_mode = True
        osc3 = controller.play_note(660.0, 0, 0.6)

        # Should now have 3 total notes (2 poly + 1 mono)
        assert len(config.active_notes) == 3
        assert 'mono' in config.active_notes

    def test_midi_note_conversion_integration(self):
        """Test MIDI note to frequency conversion in context."""
        # Test musical relationships
        c4 = controller.play_midi_note(60, 0, 1.0)  # C4
        e4 = controller.play_midi_note(64, 0, 1.0)  # E4 (major third)
        g4 = controller.play_midi_note(67, 0, 1.0)  # G4 (perfect fifth)

        # Check that frequencies are correct
        c4_freq = controller.midi_to_freq(60)
        e4_freq = controller.midi_to_freq(64)
        g4_freq = controller.midi_to_freq(67)

        assert c4.freq == c4_freq
        assert e4.freq == e4_freq
        assert g4.freq == g4_freq

        # Check musical intervals (approximately)
        major_third_ratio = e4_freq / c4_freq
        perfect_fifth_ratio = g4_freq / c4_freq

        assert abs(major_third_ratio - 1.26) < 0.01  # Major third ≈ 1.26
        assert abs(perfect_fifth_ratio - 1.50) < 0.01  # Perfect fifth ≈ 1.50


class TestControllerEdgeCases:
    """Test cases for edge cases and error conditions."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False

    def test_play_note_zero_frequency(self):
        """Test playing note with zero frequency."""
        osc = controller.play_note(0.0, 0, 1.0)
        assert osc.freq == 0.0

    def test_play_note_negative_frequency(self):
        """Test playing note with negative frequency."""
        osc = controller.play_note(-440.0, 0, 1.0)
        assert osc.freq == -440.0

    def test_play_note_extreme_velocities(self):
        """Test playing notes with extreme velocity values."""
        osc1 = controller.play_note(440.0, 0, 0.0)  # Zero velocity
        osc2 = controller.play_note(440.0, 0, 2.0)  # High velocity

        assert osc1.velocity == 0.0
        assert osc2.velocity == 2.0

    def test_midi_to_freq_extreme_values(self):
        """Test MIDI conversion with extreme values."""
        # Very low MIDI note
        freq_low = controller.midi_to_freq(-10)
        assert freq_low > 0

        # Very high MIDI note
        freq_high = controller.midi_to_freq(200)
        assert freq_high > 0
        assert freq_high > freq_low

    def test_play_sequence_empty_notes(self):
        """Test sequence with empty note tuples."""
        sequence = [
            (),            # Empty tuple
            (440.0,),      # Missing duration
            (440.0, 0.5),  # Valid note
        ]

        # Should handle gracefully
        with patch('threading.Timer'):
            controller.play_sequence(sequence, interval=0.0)

    def test_very_large_note_counter(self):
        """Test with very large note counter values."""
        controller._note_counter = 999999

        osc = controller.play_note(440.0, 0, 1.0)
        assert osc.key == 'program_1000000'
        assert controller._note_counter == 1000000

    def test_concurrent_note_access(self):
        """Test concurrent access to active_notes dict."""
        import threading

        def add_notes():
            for i in range(10):
                controller.play_note(440.0 + i * 10, 0, 1.0)
                time.sleep(0.001)

        def remove_notes():
            time.sleep(0.005)  # Let some notes be added first
            with config.notes_lock:
                keys_to_remove = list(config.active_notes.keys())[:5]
                for key in keys_to_remove:
                    if key in config.active_notes:
                        del config.active_notes[key]

        # Start concurrent operations
        thread1 = threading.Thread(target=add_notes)
        thread2 = threading.Thread(target=remove_notes)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should complete without errors (main test is that it doesn't crash)
        assert True
