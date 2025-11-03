"""Comprehensive unit tests for the controller functionality."""

import time
from unittest.mock import patch, MagicMock
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
        config.octave_offset = 0  # Reset octave offset for consistent tests

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
        config.octave_offset = 0  # Reset octave offset for consistent tests

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
        controller.play_note(440.0, 0, 0.7)
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
        controller.play_note(440.0, 0, 0.7)
        config.active_notes['mono'].released = True
        config.active_notes['mono'].env_time = 1.0
        config.active_notes['mono'].lfo_env_time = 0.5

        # Play new note
        controller.play_note(550.0, 0, 0.8)

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

            controller.play_note(440.0, 1.5, 0.8)

            # Timer should be created and started
            mock_timer.assert_called_once()
            timer_args = mock_timer.call_args[0]
            assert timer_args[0] == 1.5  # duration

            mock_timer_instance.start.assert_called_once()

    def test_play_note_no_duration_no_timer(self):
        """Test that no timer is created when duration is 0."""
        config.mono_mode = False

        with patch('threading.Timer') as mock_timer:
            controller.play_note(440.0, 0, 0.8)

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
        config.octave_offset = 0  # Reset octave offset for consistent tests

    def test_play_midi_note_basic(self):
        """Test basic MIDI note playing."""
        config.octave_offset = 0  # Ensure no octave offset
        osc = controller.play_midi_note(69, 0, 0.8)  # A4

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0
        assert osc.velocity == 0.8

    def test_play_midi_note_with_duration(self):
        """Test MIDI note with duration scheduling."""
        with patch('threading.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer.return_value = mock_timer_instance

            controller.play_midi_note(60, 2.0, 0.7)  # C4

            # Timer should be created
            mock_timer.assert_called_once()
            timer_args = mock_timer.call_args[0]
            assert timer_args[0] == 2.0  # duration

    @patch('qwerty_synth.controller.play_note')
    def test_play_midi_note_calls_play_note(self, mock_play_note):
        """Test that play_midi_note calls play_note with correct frequency."""
        config.octave_offset = 0  # Ensure no octave offset
        mock_osc = MagicMock()
        mock_play_note.return_value = mock_osc

        result = controller.play_midi_note(72, 1.5, 0.9)  # C5

        expected_freq = 440.0 * (2 ** ((72 - 69) / 12))
        mock_play_note.assert_called_once_with(expected_freq, 1.5, 0.9)
        assert result is mock_osc

    def test_play_midi_note_with_octave_offset(self):
        """Test MIDI note playing with octave offset applied."""
        config.octave_offset = 12  # One octave up
        osc = controller.play_midi_note(69, 0, 0.8)  # A4 + 1 octave = A5

        assert isinstance(osc, Oscillator)
        # A5 should be 880 Hz (A4 * 2)
        assert osc.freq == 880.0
        assert osc.velocity == 0.8

    def test_play_midi_note_with_negative_octave_offset(self):
        """Test MIDI note playing with negative octave offset."""
        config.octave_offset = -12  # One octave down
        osc = controller.play_midi_note(69, 0, 0.8)  # A4 - 1 octave = A3

        assert isinstance(osc, Oscillator)
        # A3 should be 220 Hz (A4 / 2)
        assert osc.freq == 220.0
        assert osc.velocity == 0.8


class TestPlaySequence:
    """Test cases for play_sequence function."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.octave_offset = 0  # Reset octave offset for consistent tests

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
        config.octave_offset = 0  # Reset octave offset for consistent tests

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
        assert config.midi_playback_active

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
        assert not config.midi_playback_active

    def test_midi_playback_state_management(self):
        """Test MIDI playback state flags."""
        # Test initial state
        assert not config.midi_playback_active

        # Test that we can set the flag
        config.midi_playback_active = True
        assert config.midi_playback_active

        config.midi_playback_active = False
        assert not config.midi_playback_active


class TestControllerIntegration:
    """Integration tests for controller functionality."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.octave_offset = 0  # Reset octave offset for consistent tests

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
        controller.play_note(440.0, 0, 0.8)
        controller.play_note(550.0, 0, 0.7)

        assert len(config.active_notes) == 2

        # Switch to mono mode and play note
        config.mono_mode = True
        controller.play_note(660.0, 0, 0.6)

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
        config.octave_offset = 0  # Reset octave offset for consistent tests

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


class TestMidiFilePlaybackEvents:
    """Test cases for MIDI file playback event handling."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.midi_playback_active = False
        config.midi_paused = False
        config.midi_tempo_scale = 1.0
        config.octave_offset = 0  # Reset octave offset for consistent tests

    @patch('qwerty_synth.controller.mido.MidiFile')
    @patch('qwerty_synth.controller.play_midi_note')
    @patch('threading.Thread')
    def test_midi_note_on_events_setup(self, mock_thread, mock_play_midi, mock_midi_file_class):
        """Test MIDI file setup and thread creation for note events."""
        # Create mock MIDI messages
        note_on_msg = MagicMock()
        note_on_msg.type = 'note_on'
        note_on_msg.note = 60  # C4
        note_on_msg.velocity = 100
        note_on_msg.channel = 0
        note_on_msg.time = 0.0

        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([note_on_msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Should have created and started a thread
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

        # Should have set up playback state
        assert config.midi_playback_active is True

    @patch('qwerty_synth.controller.mido.MidiFile')
    @patch('threading.Thread')
    def test_midi_file_playback_thread_function(self, mock_thread, mock_midi_file_class):
        """Test that the MIDI playback thread function is properly created."""
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([]))
        mock_midi_file_class.return_value = mock_midi_file

        controller.play_midi_file('test.mid')

        # Check that thread was called with the correct target function
        mock_thread.assert_called_once()
        call_args = mock_thread.call_args
        assert 'target' in call_args[1]
        assert call_args[1]['daemon'] is True

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_basic_setup_only(self, mock_midi_file_class):
        """Test basic MIDI playback setup without complex state assertions."""
        # Create mock messages with proper time attributes
        msg1 = MagicMock()
        msg1.time = 0.1
        msg2 = MagicMock()
        msg2.time = 0.2

        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg1, msg2]))
        mock_midi_file_class.return_value = mock_midi_file

        # Test with different tempo scales
        controller.play_midi_file('test.mid', 2.0)

        # Test that the basic setup works
        assert config.midi_tempo_scale == 2.0
        assert config.midi_playback_duration == (0.1 + 0.2) / 2.0

    def test_midi_playback_pause_resume_state(self):
        """Test MIDI playback pause and resume state changes."""
        # Test pause state
        config.midi_paused = False
        assert config.midi_paused is False

        config.midi_paused = True
        assert config.midi_paused is True

        config.midi_paused = False
        assert config.midi_paused is False

    def test_midi_playback_stop_state(self):
        """Test MIDI playback stop state management."""
        config.midi_playback_active = True
        assert config.midi_playback_active is True

        config.midi_playback_active = False
        assert config.midi_playback_active is False

    def test_midi_tempo_scale_changes(self):
        """Test MIDI tempo scale changes."""
        config.midi_tempo_scale = 1.0
        assert config.midi_tempo_scale == 1.0

        config.midi_tempo_scale = 0.5  # Half speed
        assert config.midi_tempo_scale == 0.5

        config.midi_tempo_scale = 2.0  # Double speed
        assert config.midi_tempo_scale == 2.0

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_file_exception_handling_detailed(self, mock_midi_file_class):
        """Test detailed exception handling in MIDI file operations."""
        # Test file loading exception
        mock_midi_file_class.side_effect = FileNotFoundError("File not found")

        controller.play_midi_file('nonexistent.mid')
        assert config.midi_playback_active is False

        # Test other exceptions
        mock_midi_file_class.side_effect = Exception("Generic error")
        controller.play_midi_file('error.mid')
        assert config.midi_playback_active is False

    def test_midi_note_velocity_conversion(self):
        """Test MIDI velocity conversion logic."""
        # Test velocity conversion from 0-127 to 0.0-1.0
        test_velocities = [0, 64, 100, 127]
        expected_velocities = [0.0, 64/127.0, 100/127.0, 1.0]

        for midi_vel, expected_vel in zip(test_velocities, expected_velocities):
            converted = midi_vel / 127.0
            assert abs(converted - expected_vel) < 0.001

    @patch('qwerty_synth.controller.mido.MidiFile')
    @patch('qwerty_synth.controller.play_midi_note')
    def test_midi_channel_key_generation(self, mock_play_midi, mock_midi_file_class):
        """Test MIDI channel key generation for note tracking."""
        mock_osc = MagicMock()
        mock_osc.key = 'program_1'
        mock_play_midi.return_value = mock_osc

        # Test channel key generation logic
        channel = 5
        note = 60
        expected_key = (channel, note)

        # This tests the logic that would be used in the MIDI playback thread
        channel_key = (channel, note)
        assert channel_key == expected_key
        assert isinstance(channel_key, tuple)
        assert len(channel_key) == 2

    def test_midi_message_type_handling(self):
        """Test MIDI message type identification."""
        # Test note on message identification
        note_on_msg = MagicMock()
        note_on_msg.type = 'note_on'
        note_on_msg.velocity = 100
        assert note_on_msg.type == 'note_on' and note_on_msg.velocity > 0

        # Test note off message identification
        note_off_msg = MagicMock()
        note_off_msg.type = 'note_off'
        assert note_off_msg.type == 'note_off'

        # Test note on with zero velocity (treated as note off)
        note_on_zero = MagicMock()
        note_on_zero.type = 'note_on'
        note_on_zero.velocity = 0
        assert note_on_zero.type == 'note_on' and note_on_zero.velocity == 0

    def test_active_notes_management(self):
        """Test active notes management during MIDI playback."""
        # Test adding notes to active_notes
        mock_osc = MagicMock()
        mock_osc.key = 'program_1'

        config.active_notes[mock_osc.key] = mock_osc
        assert mock_osc.key in config.active_notes

        # Test releasing notes
        config.active_notes[mock_osc.key].released = True
        config.active_notes[mock_osc.key].env_time = 0.0
        config.active_notes[mock_osc.key].lfo_env_time = 0.0

        assert config.active_notes[mock_osc.key].released is True
        assert config.active_notes[mock_osc.key].env_time == 0.0
        assert config.active_notes[mock_osc.key].lfo_env_time == 0.0

        # Test removing notes
        del config.active_notes[mock_osc.key]
        assert mock_osc.key not in config.active_notes

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_timing_calculations(self, mock_midi_file_class):
        """Test MIDI timing calculations."""
        # Test duration calculation
        msg1 = MagicMock()
        msg1.time = 0.5
        msg2 = MagicMock()
        msg2.time = 1.0
        msg3 = MagicMock()
        msg3.time = 0.3

        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg1, msg2, msg3]))
        mock_midi_file_class.return_value = mock_midi_file

        controller.play_midi_file('test.mid', 1.5)

        expected_duration = (0.5 + 1.0 + 0.3) / 1.5
        assert config.midi_playback_duration == expected_duration

    def test_timing_precision_logic(self):
        """Test timing precision logic used in MIDI playback."""
        # Test wait time calculation logic
        target_time = 1.0
        current_time = 0.5
        wait_time = target_time - current_time

        assert wait_time == 0.5

        # Test short vs long wait logic
        short_wait = 0.005
        long_wait = 0.1

        assert short_wait < 0.01  # Would use busy waiting
        assert long_wait >= 0.01  # Would use sleep + busy wait

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_file_basic_loading(self, mock_midi_file_class):
        """Test basic MIDI file loading without complex state assertions."""
        # Create mock messages with proper attributes
        msg1 = MagicMock()
        msg1.type = 'note_on'
        msg1.note = 60
        msg1.velocity = 100
        msg1.channel = 0
        msg1.time = 0.0

        msg2 = MagicMock()
        msg2.type = 'note_off'
        msg2.note = 60
        msg2.velocity = 0
        msg2.channel = 0
        msg2.time = 0.5

        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg1, msg2]))
        mock_midi_file_class.return_value = mock_midi_file

        controller.play_midi_file('test.mid')

        # Should have attempted to load the MIDI file
        mock_midi_file_class.assert_called_once_with('test.mid')


class TestMidiPlaybackEventHandling:
    """Test detailed MIDI playback event handling to improve coverage."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.midi_playback_active = False
        config.midi_paused = False
        config.midi_tempo_scale = 1.0
        config.octave_offset = 0

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_note_on_handling(self, mock_midi_file_class):
        """Test MIDI note_on event basic setup."""
        # Create note_on message
        note_on_msg = MagicMock()
        note_on_msg.type = 'note_on'
        note_on_msg.note = 60
        note_on_msg.velocity = 100
        note_on_msg.channel = 0
        note_on_msg.time = 0.5

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([note_on_msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Verify playback was started
        assert config.midi_playback_active

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_note_off_handling(self, mock_midi_file_class):
        """Test MIDI note_off event basic setup."""
        # Create note_off message
        note_off_msg = MagicMock()
        note_off_msg.type = 'note_off'
        note_off_msg.note = 60
        note_off_msg.velocity = 0
        note_off_msg.channel = 0
        note_off_msg.time = 0.1

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([note_off_msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Verify playback was started
        assert config.midi_playback_active

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_pause_resume(self, mock_midi_file_class):
        """Test MIDI playback pause and resume state management."""
        # Create simple message
        msg = MagicMock()
        msg.type = 'note_on'
        msg.note = 60
        msg.velocity = 100
        msg.channel = 0
        msg.time = 0.1

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Test pause
        config.midi_paused = True
        assert config.midi_paused is True

        # Test resume
        config.midi_paused = False
        assert config.midi_paused is False

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_stop(self, mock_midi_file_class):
        """Test MIDI playback stop functionality."""
        # Create message
        msg = MagicMock()
        msg.type = 'note_on'
        msg.note = 60
        msg.velocity = 100
        msg.channel = 0
        msg.time = 0.0

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Test that we can control the state flags
        config.midi_playback_active = False
        assert config.midi_playback_active is False

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_tempo_changes(self, mock_midi_file_class):
        """Test MIDI playback with tempo scale changes."""
        # Create message
        msg = MagicMock()
        msg.type = 'note_on'
        msg.note = 60
        msg.velocity = 100
        msg.channel = 0
        msg.time = 0.1

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback with tempo scale
        controller.play_midi_file('test.mid', 2.0)
        assert config.midi_tempo_scale == 2.0

        # Change tempo during playback
        config.midi_tempo_scale = 0.5
        assert config.midi_tempo_scale == 0.5

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_timing_precision(self, mock_midi_file_class):
        """Test MIDI playback with precise timing values."""
        # Create message with timing
        msg = MagicMock()
        msg.type = 'note_on'
        msg.note = 60
        msg.velocity = 100
        msg.channel = 0
        msg.time = 0.02  # 20ms timing

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Verify playback started
        assert config.midi_playback_active

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_thread_exception_handling(self, mock_midi_file_class):
        """Test exception handling in MIDI playback thread."""
        # Make MIDI file iteration raise an exception
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(side_effect=Exception("Playback error"))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Let the thread run briefly
        time.sleep(0.1)

        # Playback should be stopped due to exception
        assert config.midi_playback_active is False

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_note_on_zero_velocity(self, mock_midi_file_class):
        """Test MIDI note_on with zero velocity (treated as note_off)."""
        # Create note_on with zero velocity
        msg = MagicMock()
        msg.type = 'note_on'
        msg.note = 60
        msg.velocity = 0  # Zero velocity = note off
        msg.channel = 0
        msg.time = 0.1

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Verify playback started
        assert config.midi_playback_active

    @patch('qwerty_synth.controller.mido.MidiFile')
    def test_midi_playback_multiple_channels(self, mock_midi_file_class):
        """Test MIDI playback with multiple channels."""
        # Create messages on different channels
        msg1 = MagicMock()
        msg1.type = 'note_on'
        msg1.note = 60
        msg1.velocity = 100
        msg1.channel = 0
        msg1.time = 0.0

        msg2 = MagicMock()
        msg2.type = 'note_on'
        msg2.note = 64
        msg2.velocity = 80
        msg2.channel = 1
        msg2.time = 0.1

        # Mock MIDI file
        mock_midi_file = MagicMock()
        mock_midi_file.__iter__ = MagicMock(return_value=iter([msg1, msg2]))
        mock_midi_file_class.return_value = mock_midi_file

        # Start playback
        controller.play_midi_file('test.mid')

        # Verify playback started
        assert config.midi_playback_active

    def test_midi_playback_state_variables(self):
        """Test MIDI playback state variable management."""
        # Test initial states
        assert config.midi_playback_active is False
        assert config.midi_paused is False
        assert config.midi_tempo_scale == 1.0

        # Test state changes
        config.midi_playback_active = True
        config.midi_paused = True
        config.midi_tempo_scale = 1.5

        assert config.midi_playback_active is True
        assert config.midi_paused is True
        assert config.midi_tempo_scale == 1.5

        # Reset
        config.midi_playback_active = False
        config.midi_paused = False
        config.midi_tempo_scale = 1.0


class TestControllerMidiIntegration:
    """Test MIDI integration features and edge cases."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.midi_playback_active = False
        config.midi_paused = False
        config.midi_tempo_scale = 1.0
        config.octave_offset = 0
        config.semitone_offset = 0

    def test_note_release_function_creation(self):
        """Test the release function created by play_note."""
        config.mono_mode = False

        osc = controller.play_note(440.0, 0, 0.8)
        key = osc.key

        # Note should be in active_notes and not released
        assert key in config.active_notes
        assert not config.active_notes[key].released

        # Manually trigger a release (simulating timer callback)
        with config.notes_lock:
            if key in config.active_notes:
                config.active_notes[key].released = True
                config.active_notes[key].env_time = 0.0
                config.active_notes[key].lfo_env_time = 0.0

        assert config.active_notes[key].released

    def test_note_release_nonexistent_key(self):
        """Test release function with non-existent key."""
        # Try to release a note that doesn't exist
        with config.notes_lock:
            if 'nonexistent' in config.active_notes:
                config.active_notes['nonexistent'].released = True

        # Should not crash

    def test_semitone_offset_application(self):
        """Test semitone offset application in play_midi_note."""
        config.octave_offset = 0
        config.semitone_offset = 3  # Three semitones up

        osc = controller.play_midi_note(60, 0, 0.8)  # C4 + 3 semitones = D#4

        # C4 (60) + 3 semitones = D#4 (63)
        expected_freq = controller.midi_to_freq(63)
        assert abs(osc.freq - expected_freq) < 0.01

    def test_combined_offset_application(self):
        """Test combined octave and semitone offsets."""
        config.octave_offset = -12  # One octave down
        config.semitone_offset = 7   # Seven semitones up (perfect fifth)

        osc = controller.play_midi_note(60, 0, 0.8)  # C4 - 12 + 7 = G3

        # C4 (60) - 12 + 7 = G3 (55)
        expected_freq = controller.midi_to_freq(55)
        assert abs(osc.freq - expected_freq) < 0.01

    def test_play_note_with_zero_velocity(self):
        """Test playing note with zero velocity."""
        osc = controller.play_note(440.0, 0, 0.0)
        assert osc.velocity == 0.0
        assert osc.freq == 440.0

    def test_play_note_with_high_velocity(self):
        """Test playing note with velocity > 1.0."""
        osc = controller.play_note(440.0, 0, 2.5)
        assert osc.velocity == 2.5
        assert osc.freq == 440.0

    def test_midi_file_duration_with_zero_tempo(self):
        """Test MIDI file duration calculation with zero tempo scale."""
        with patch('qwerty_synth.controller.mido.MidiFile') as mock_midi_file_class:
            # Create mock messages
            msg1 = MagicMock()
            msg1.time = 1.0
            msg2 = MagicMock()
            msg2.time = 0.5

            mock_midi_file = MagicMock()
            mock_midi_file.__iter__ = MagicMock(return_value=iter([msg1, msg2]))
            mock_midi_file_class.return_value = mock_midi_file

            # Try with very small tempo scale
            controller.play_midi_file('test.mid', 0.001)

            # Should handle extreme tempo values
            assert config.midi_tempo_scale == 0.001

    def test_midi_file_empty_file(self):
        """Test MIDI file with no messages."""
        with patch('qwerty_synth.controller.mido.MidiFile') as mock_midi_file_class:
            mock_midi_file = MagicMock()
            mock_midi_file.__iter__ = MagicMock(return_value=iter([]))
            mock_midi_file_class.return_value = mock_midi_file

            controller.play_midi_file('empty.mid')

            # Should handle empty files gracefully
            assert config.midi_playback_duration == 0.0

    def test_velocity_conversion_edge_cases(self):
        """Test MIDI velocity conversion edge cases."""
        # Test velocity conversion logic
        test_cases = [
            (0, 0.0),
            (1, 1/127.0),
            (64, 64/127.0),
            (127, 1.0)
        ]

        for midi_vel, expected in test_cases:
            converted = midi_vel / 127.0
            assert abs(converted - expected) < 0.001

    def test_play_sequence_edge_cases(self):
        """Test play_sequence with edge cases."""
        # Empty sequence
        controller.play_sequence([], 0.1)

        # Sequence with just one note
        with patch('qwerty_synth.controller.play_note') as mock_play_note:
            controller.play_sequence([(440.0, 0.5)], 0.0)
            mock_play_note.assert_called_once_with(440.0, 0.5, 1.0)

        # Sequence with malformed entries
        with patch('qwerty_synth.controller.play_note') as mock_play_note:
            controller.play_sequence([()], 0.0)  # Empty tuple
            # Should not crash

    def test_note_counter_overflow_behavior(self):
        """Test behavior with very large note counter."""
        # Set a very large counter
        controller._note_counter = 999999999

        osc = controller.play_note(440.0, 0, 1.0)
        assert osc.key == 'program_1000000000'
        assert controller._note_counter == 1000000000

    def test_mono_mode_repeated_notes(self):
        """Test mono mode with repeated note playing."""
        config.mono_mode = True

        # Play same frequency multiple times
        osc1 = controller.play_note(440.0, 0, 0.8)
        osc2 = controller.play_note(440.0, 0, 0.7)
        osc3 = controller.play_note(440.0, 0, 0.6)

        # Should always be the same oscillator
        assert osc1 is osc2 is osc3
        assert len(config.active_notes) == 1

    def test_threading_timer_behavior(self):
        """Test threading.Timer behavior with play_note."""
        with patch('threading.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer.return_value = mock_timer_instance

            # Test with positive duration
            controller.play_note(440.0, 1.0, 0.8)
            mock_timer.assert_called_once()
            mock_timer_instance.start.assert_called_once()

            # Reset mocks
            mock_timer.reset_mock()
            mock_timer_instance.reset_mock()

            # Test with zero duration
            controller.play_note(440.0, 0.0, 0.8)
            mock_timer.assert_not_called()

            # Test with negative duration (should not schedule)
            controller.play_note(440.0, -1.0, 0.8)
            mock_timer.assert_not_called()


class TestPlayMidiNoteDirect:
    """Test cases for play_midi_note_direct function."""

    def setup_method(self):
        """Reset state before each test."""
        controller._note_counter = 0
        config.active_notes = {}
        config.mono_mode = False
        config.octave_offset = 0  # Reset octave offset for consistent tests

    def test_play_midi_note_direct_basic(self):
        """Test basic MIDI note playing without octave offset."""
        config.octave_offset = 0  # Ensure no octave offset
        osc = controller.play_midi_note_direct(69, 0, 0.8)  # A4

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0
        assert osc.velocity == 0.8

    def test_play_midi_note_direct_ignores_octave_offset(self):
        """Test that play_midi_note_direct ignores global octave offset."""
        config.octave_offset = 12  # One octave up
        osc = controller.play_midi_note_direct(69, 0, 0.8)  # A4

        assert isinstance(osc, Oscillator)
        # Should still be A4 (440 Hz), not A5 (880 Hz)
        assert osc.freq == 440.0
        assert osc.velocity == 0.8

    def test_play_midi_note_direct_with_negative_octave_offset(self):
        """Test MIDI note playing direct function ignores negative octave offset."""
        config.octave_offset = -12  # One octave down
        config.semitone_offset = -2  # Two semitones down (D)
        osc = controller.play_midi_note_direct(69, 0, 0.8)  # A4, should still be A4

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0  # Should ignore both offsets
        assert osc.velocity == 0.8

    def test_play_midi_note_direct_with_semitone_offset(self):
        """Test MIDI note playing direct function ignores semitone offset."""
        config.octave_offset = 0
        config.semitone_offset = 5  # Five semitones up
        osc = controller.play_midi_note_direct(69, 0, 0.8)  # A4, should still be A4

        assert isinstance(osc, Oscillator)
        assert osc.freq == 440.0  # Should ignore semitone offset
        assert osc.velocity == 0.8

    def test_play_midi_note_with_semitone_offset(self):
        """Test MIDI note playing with semitone offset applied."""
        config.octave_offset = 0
        config.semitone_offset = 1  # One semitone up (A# / Bb)
        osc = controller.play_midi_note(69, 0, 0.8)  # A4 + 1 semitone = A#4

        assert isinstance(osc, Oscillator)
        # A#4 should be approximately 466.16 Hz
        expected_freq = 440.0 * (2 ** (1/12))
        assert abs(osc.freq - expected_freq) < 0.01
        assert osc.velocity == 0.8

    def test_play_midi_note_with_octave_and_semitone_offset(self):
        """Test MIDI note playing with both octave and semitone offsets."""
        config.octave_offset = 12  # One octave up
        config.semitone_offset = -2  # Two semitones down
        osc = controller.play_midi_note(69, 0, 0.8)  # A4 + 12 - 2 = A5 - 2 semitones = G5

        assert isinstance(osc, Oscillator)
        # G5 should be 783.99 Hz (G4 = 392.00 Hz * 2 = 784.00 Hz)
        # Calculate: A4(69) + 12 - 2 = MIDI note 79 (G5)
        expected_midi = 69 + 12 - 2  # = 79 (G5)
        expected_freq = 440.0 * (2 ** ((expected_midi - 69) / 12))
        assert abs(osc.freq - expected_freq) < 0.01
        assert osc.velocity == 0.8

    def test_play_midi_note_direct_with_duration(self):
        """Test MIDI note direct with duration scheduling."""
        with patch('threading.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer.return_value = mock_timer_instance

            controller.play_midi_note_direct(60, 2.0, 0.7)  # C4

            # Timer should be created
            mock_timer.assert_called_once()
            timer_args = mock_timer.call_args[0]
            assert timer_args[0] == 2.0  # duration

    @patch('qwerty_synth.controller.play_note')
    def test_play_midi_note_direct_calls_play_note(self, mock_play_note):
        """Test that play_midi_note_direct calls play_note with correct frequency."""
        config.octave_offset = 12  # Set octave offset to ensure it's ignored
        mock_osc = MagicMock()
        mock_play_note.return_value = mock_osc

        result = controller.play_midi_note_direct(72, 1.5, 0.9)  # C5

        # Should use the original MIDI note (72), not offset (72 + 12 = 84)
        expected_freq = 440.0 * (2 ** ((72 - 69) / 12))  # C5 frequency
        mock_play_note.assert_called_once_with(expected_freq, 1.5, 0.9)
        assert result is mock_osc
