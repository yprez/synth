"""Comprehensive unit tests for the recording functionality."""

import numpy as np
import pytest
from unittest.mock import patch
from unittest.mock import MagicMock

from qwerty_synth import record


class TestRecordingState:
    """Test cases for recording state management."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    def test_initial_state(self):
        """Test initial recording state."""
        assert not record.is_recording()
        assert record.get_recording_time() == 0
        assert record.current_recording_path is None

    def test_is_recording_true_when_enabled(self):
        """Test is_recording returns True when recording enabled."""
        record.recording_enabled = True
        assert record.is_recording()

    def test_is_recording_false_when_disabled(self):
        """Test is_recording returns False when recording disabled."""
        record.recording_enabled = False
        assert not record.is_recording()


class TestStartRecording:
    """Test cases for start_recording function."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    def test_start_recording_with_custom_path(self):
        """Test starting recording with custom output path."""
        custom_path = "/tmp/test_recording.wav"

        result_path = record.start_recording(custom_path)

        assert record.is_recording()
        assert result_path == custom_path
        assert record.current_recording_path == custom_path
        assert record.recorded_audio == []

    @patch('qwerty_synth.record.datetime')
    @patch('qwerty_synth.record.Path')
    def test_start_recording_auto_path_generation(self, mock_path_class, mock_datetime):
        """Test automatic path generation when no path provided."""
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        # Mock Path operations more precisely
        mock_home = MagicMock()
        mock_path_class.home.return_value = mock_home
        mock_directory = MagicMock()
        mock_home.__truediv__ = MagicMock(return_value=mock_directory)
        mock_directory.exists.return_value = False
        mock_directory.mkdir = MagicMock()

        # Mock the final path construction
        mock_final_path = MagicMock()
        mock_final_path.__str__ = MagicMock(return_value="/home/user/qwerty_synth_recordings/qwerty_synth_20240101_120000.wav")
        mock_directory.__truediv__ = MagicMock(return_value=mock_final_path)

        result_path = record.start_recording()

        assert record.is_recording()
        assert "qwerty_synth_20240101_120000.wav" in str(result_path)
        mock_directory.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('qwerty_synth.record.Path')
    def test_start_recording_directory_exists(self, mock_path_class):
        """Test starting recording when directory already exists."""
        # Mock Path operations
        mock_home = mock_path_class.home.return_value
        mock_directory = mock_home / "qwerty_synth_recordings"
        mock_directory.exists.return_value = True

        record.start_recording()

        # mkdir should not be called if directory exists
        mock_directory.mkdir.assert_not_called()

    def test_start_recording_clears_previous_audio(self):
        """Test that starting recording clears previous audio buffer."""
        # Add some dummy audio first
        record.recorded_audio = [np.array([[1, 2], [3, 4]])]

        record.start_recording("/tmp/test.wav")

        assert record.recorded_audio == []

    def test_multiple_start_calls(self):
        """Test multiple calls to start_recording."""
        path1 = record.start_recording("/tmp/test1.wav")
        path2 = record.start_recording("/tmp/test2.wav")

        assert path1 == "/tmp/test1.wav"
        assert path2 == "/tmp/test2.wav"
        assert record.current_recording_path == "/tmp/test2.wav"


class TestStopRecording:
    """Test cases for stop_recording function."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    @patch('qwerty_synth.record.sf.write')
    def test_stop_recording_with_audio(self, mock_sf_write):
        """Test stopping recording with audio data."""
        # Set up recording state
        record.recording_enabled = True
        record.current_recording_path = "/tmp/test.wav"

        # Add some test audio data
        audio1 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        audio2 = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        record.recorded_audio = [audio1, audio2]

        result_path = record.stop_recording(sample_rate=48000, bit_depth=16)

        # Check state after stopping
        assert not record.is_recording()
        assert record.recorded_audio == []
        assert record.current_recording_path is None
        assert result_path == "/tmp/test.wav"

        # Check that sf.write was called correctly
        mock_sf_write.assert_called_once()
        call_args = mock_sf_write.call_args
        assert call_args[0][0] == "/tmp/test.wav"  # filename
        assert call_args[0][2] == 48000  # sample_rate
        assert call_args[1]['subtype'] == 'PCM_16'  # bit_depth

        # Check that audio was properly concatenated
        expected_audio = np.vstack([audio1, audio2])
        np.testing.assert_array_equal(call_args[0][1], expected_audio)

    @patch('qwerty_synth.record.sf.write')
    def test_stop_recording_24bit(self, mock_sf_write):
        """Test stopping recording with 24-bit depth."""
        record.recording_enabled = True
        record.current_recording_path = "/tmp/test.wav"
        record.recorded_audio = [np.array([[0.1, 0.2]])]

        record.stop_recording(bit_depth=24)

        call_args = mock_sf_write.call_args
        assert call_args[1]['subtype'] == 'PCM_24'

    def test_stop_recording_not_enabled(self):
        """Test stopping recording when not recording."""
        record.recording_enabled = False

        result = record.stop_recording()

        assert result is None

    def test_stop_recording_no_audio(self):
        """Test stopping recording with no audio data."""
        record.recording_enabled = True
        record.recorded_audio = []

        result = record.stop_recording()

        assert result is None

    @patch('qwerty_synth.record.sf.write')
    def test_stop_recording_sf_write_exception(self, mock_sf_write):
        """Test handling of sf.write exceptions."""
        mock_sf_write.side_effect = Exception("Write failed")

        record.recording_enabled = True
        record.current_recording_path = "/tmp/test.wav"
        record.recorded_audio = [np.array([[0.1, 0.2]])]

        # Should raise the exception
        with pytest.raises(Exception, match="Write failed"):
            record.stop_recording()


class TestAddAudioBlock:
    """Test cases for add_audio_block function."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    def test_add_audio_block_when_recording(self):
        """Test adding audio blocks when recording is enabled."""
        record.recording_enabled = True

        audio_block = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        record.add_audio_block(audio_block)

        assert len(record.recorded_audio) == 1
        np.testing.assert_array_equal(record.recorded_audio[0], audio_block)

        # Verify it's a copy, not the same object
        assert record.recorded_audio[0] is not audio_block

    def test_add_audio_block_when_not_recording(self):
        """Test adding audio blocks when recording is disabled."""
        record.recording_enabled = False

        audio_block = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        record.add_audio_block(audio_block)

        assert len(record.recorded_audio) == 0

    def test_add_multiple_audio_blocks(self):
        """Test adding multiple audio blocks."""
        record.recording_enabled = True

        audio1 = np.array([[0.1, 0.2]], dtype=np.float32)
        audio2 = np.array([[0.3, 0.4]], dtype=np.float32)
        audio3 = np.array([[0.5, 0.6]], dtype=np.float32)

        record.add_audio_block(audio1)
        record.add_audio_block(audio2)
        record.add_audio_block(audio3)

        assert len(record.recorded_audio) == 3
        np.testing.assert_array_equal(record.recorded_audio[0], audio1)
        np.testing.assert_array_equal(record.recorded_audio[1], audio2)
        np.testing.assert_array_equal(record.recorded_audio[2], audio3)

    def test_add_audio_block_different_shapes(self):
        """Test adding audio blocks with different shapes."""
        record.recording_enabled = True

        # Different frame counts but same channel count
        audio1 = np.array([[0.1, 0.2]], dtype=np.float32)  # 1 frame
        audio2 = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)  # 2 frames

        record.add_audio_block(audio1)
        record.add_audio_block(audio2)

        assert len(record.recorded_audio) == 2
        assert record.recorded_audio[0].shape == (1, 2)
        assert record.recorded_audio[1].shape == (2, 2)

    def test_add_audio_block_copies_data(self):
        """Test that add_audio_block makes copies of the data."""
        record.recording_enabled = True

        original_audio = np.array([[0.1, 0.2]], dtype=np.float32)
        record.add_audio_block(original_audio)

        # Modify original
        original_audio[0, 0] = 999.0

        # Recorded audio should be unchanged
        assert record.recorded_audio[0][0, 0] == 0.1


class TestGetRecordingTime:
    """Test cases for get_recording_time function."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    def test_get_recording_time_no_audio(self):
        """Test getting recording time with no audio."""
        assert record.get_recording_time() == 0

    def test_get_recording_time_with_audio(self):
        """Test getting recording time with audio data."""
        # Simulate 44.1kHz audio
        # 1 second = 44100 frames
        # 0.5 seconds = 22050 frames

        # Add blocks totaling 1.5 seconds of audio
        audio1 = np.zeros((44100, 2))  # 1 second
        audio2 = np.zeros((22050, 2))  # 0.5 seconds

        record.recorded_audio = [audio1, audio2]

        recording_time = record.get_recording_time()
        assert recording_time == 1.5

    def test_get_recording_time_single_block(self):
        """Test getting recording time with single audio block."""
        # 2205 frames = 0.05 seconds at 44.1kHz
        audio = np.zeros((2205, 2))
        record.recorded_audio = [audio]

        recording_time = record.get_recording_time()
        assert abs(recording_time - 0.05) < 0.001  # Allow small floating point error

    def test_get_recording_time_empty_blocks(self):
        """Test getting recording time with empty blocks."""
        # Empty blocks should contribute 0 time
        audio1 = np.zeros((0, 2))
        audio2 = np.zeros((44100, 2))  # 1 second
        audio3 = np.zeros((0, 2))

        record.recorded_audio = [audio1, audio2, audio3]

        recording_time = record.get_recording_time()
        assert recording_time == 1.0


class TestRecordingIntegration:
    """Integration tests for the recording workflow."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    def test_full_recording_workflow(self):
        """Test complete recording workflow."""
        # Start recording
        path = record.start_recording("/tmp/integration_test.wav")
        assert record.is_recording()
        assert record.get_recording_time() == 0

        # Add some audio
        audio1 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        audio2 = np.array([[0.5, 0.6]], dtype=np.float32)

        record.add_audio_block(audio1)
        assert record.get_recording_time() > 0

        record.add_audio_block(audio2)

        # Stop recording (mock file writing)
        with patch('qwerty_synth.record.sf.write') as mock_write:
            result_path = record.stop_recording()

        assert result_path == path
        assert not record.is_recording()
        assert record.get_recording_time() == 0
        mock_write.assert_called_once()

    def test_recording_state_after_stop(self):
        """Test that recording state is properly reset after stopping."""
        # Start and add audio
        record.start_recording("/tmp/test.wav")
        record.add_audio_block(np.array([[0.1, 0.2]]))

        # Stop recording
        with patch('qwerty_synth.record.sf.write'):
            record.stop_recording()

        # State should be reset
        assert not record.is_recording()
        assert record.recorded_audio == []
        assert record.current_recording_path is None
        assert record.get_recording_time() == 0

    def test_restart_recording_after_stop(self):
        """Test starting a new recording after stopping previous one."""
        # First recording
        with patch('qwerty_synth.record.sf.write'):
            record.start_recording("/tmp/test1.wav")
            record.add_audio_block(np.array([[0.1, 0.2]]))
            record.stop_recording()

        # Second recording
        path2 = record.start_recording("/tmp/test2.wav")
        assert record.is_recording()
        assert record.current_recording_path == "/tmp/test2.wav"
        assert record.recorded_audio == []

    def test_add_audio_during_recording_cycle(self):
        """Test adding audio during start/stop cycle."""
        record.start_recording("/tmp/test.wav")

        # Add audio blocks of different sizes
        for i in range(5):
            frames = (i + 1) * 100  # 100, 200, 300, 400, 500 frames
            audio = np.random.randn(frames, 2).astype(np.float32) * 0.1
            record.add_audio_block(audio)

        # Total frames should be 100+200+300+400+500 = 1500
        expected_time = 1500 / 44100
        actual_time = record.get_recording_time()
        assert abs(actual_time - expected_time) < 0.001

        with patch('qwerty_synth.record.sf.write'):
            record.stop_recording()


class TestRecordingEdgeCases:
    """Test cases for edge cases and error conditions."""

    def setup_method(self):
        """Reset recording state before each test."""
        record.recording_enabled = False
        record.recorded_audio = []
        record.current_recording_path = None

    def test_stop_without_start(self):
        """Test stopping recording without starting."""
        result = record.stop_recording()
        assert result is None

    def test_multiple_stops(self):
        """Test calling stop multiple times."""
        record.start_recording("/tmp/test.wav")
        # Add some audio data so the first stop will succeed
        record.add_audio_block(np.array([[0.1, 0.2]]))

        with patch('qwerty_synth.record.sf.write') as mock_write:
            result1 = record.stop_recording()
            result2 = record.stop_recording()

        assert result1 == "/tmp/test.wav"
        assert result2 is None
        assert mock_write.call_count == 1

    def test_add_audio_with_none(self):
        """Test adding None as audio block."""
        record.recording_enabled = True

        # This should handle gracefully or raise appropriate error
        try:
            record.add_audio_block(None)
            # If it doesn't raise an error, check state
            assert len(record.recorded_audio) <= 1
        except (AttributeError, TypeError):
            # It's acceptable to raise an error for None input
            pass

    def test_very_large_audio_blocks(self):
        """Test handling very large audio blocks."""
        record.recording_enabled = True

        # Large audio block (1 million frames)
        large_audio = np.zeros((1000000, 2), dtype=np.float32)
        record.add_audio_block(large_audio)

        assert len(record.recorded_audio) == 1
        assert record.recorded_audio[0].shape == (1000000, 2)

        # Recording time should be approximately 22.7 seconds
        time = record.get_recording_time()
        expected_time = 1000000 / 44100
        assert abs(time - expected_time) < 0.1

    def test_empty_audio_blocks(self):
        """Test handling empty audio blocks."""
        record.recording_enabled = True

        empty_audio = np.zeros((0, 2), dtype=np.float32)
        record.add_audio_block(empty_audio)

        assert len(record.recorded_audio) == 1
        assert record.recorded_audio[0].shape == (0, 2)
        assert record.get_recording_time() == 0
