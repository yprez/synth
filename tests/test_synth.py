"""Comprehensive unit tests for the core synthesizer functionality."""

import pytest

# Skip entire module if sounddevice/PortAudio is not available
try:
    import sounddevice
except OSError:
    pytest.skip("PortAudio library not found", allow_module_level=True)

import numpy as np
from unittest.mock import patch, Mock
import threading
import time

from qwerty_synth import config
from qwerty_synth.synth import (
    Oscillator,
    audio_callback,
    create_audio_stream,
    update_chorus_params,
    update_delay_params,
    mark_chorus_params_changed,
    mark_delay_params_changed
)


class TestOscillator:
    """Test cases for the Oscillator class."""

    def test_oscillator_initialization(self):
        """Test oscillator initialization with various parameters."""
        osc = Oscillator(440.0, 'sine')

        assert osc.freq == 440.0
        assert osc.target_freq == 440.0
        assert osc.waveform == 'sine'
        assert osc.phase == 0.0
        assert not osc.done
        assert osc.env_time == 0.0
        assert osc.filter_env_time == 0.0
        assert not osc.released
        assert osc.last_env_level == 0.0
        assert osc.last_filter_env_level == 0.0
        assert osc.key is None
        assert osc.velocity == 1.0

    def test_oscillator_waveforms(self, audio_helper):
        """Test all supported waveform types."""
        frames = 1024

        for waveform in ['sine', 'square', 'triangle', 'sawtooth']:
            osc = Oscillator(440.0, waveform)
            audio, filter_env = osc.generate(frames)

            assert len(audio) == frames
            assert len(filter_env) == frames
            assert audio_helper.has_signal(audio)
            assert not audio_helper.is_clipped(audio)

    def test_oscillator_frequency_accuracy(self, audio_helper):
        """Test that generated frequencies match expected values."""
        frames = 4096  # Longer for better frequency resolution
        test_frequencies = [220.0, 440.0, 880.0, 1760.0]

        for freq in test_frequencies:
            osc = Oscillator(freq, 'sine')
            audio, _ = osc.generate(frames)

            dominant_freq = audio_helper.dominant_frequency(audio, config.sample_rate)
            # Allow for some tolerance due to windowing effects and envelope modulation
            assert abs(dominant_freq - freq) < 6.0

    def test_oscillator_phase_continuity(self):
        """Test that phase is continuous across multiple generate calls."""
        osc = Oscillator(440.0, 'sine')
        frames = 100

        # Generate first block
        audio1, _ = osc.generate(frames)
        audio1[-1]

        # Generate second block
        audio2, _ = osc.generate(frames)
        first_sample2 = audio2[0]

        # Check phase continuity (adjacent samples should be similar)
        # Calculate expected next sample based on frequency and phase
        phase_increment = 2 * np.pi * osc.freq / config.sample_rate
        expected_next = np.sin(osc.phase - phase_increment)  # Previous phase

        # Allow for reasonable tolerance including envelope effects
        assert abs(first_sample2 - expected_next) < 0.12

    def test_oscillator_envelope_attack(self):
        """Test ADSR attack phase."""
        config.adsr['attack'] = 0.1  # 100ms attack
        config.adsr['decay'] = 0.0
        config.adsr['sustain'] = 1.0
        config.adsr['release'] = 0.1

        osc = Oscillator(440.0, 'sine')
        frames = int(0.05 * config.sample_rate)  # 50ms worth of samples

        audio, _ = osc.generate(frames)

        # During attack, amplitude should increase
        assert audio[0] == 0.0  # Starts at zero
        assert abs(audio[-1]) > abs(audio[0])  # Increases over time

    def test_oscillator_envelope_sustain(self):
        """Test ADSR sustain phase."""
        config.adsr['attack'] = 0.01  # Very short attack
        config.adsr['decay'] = 0.01   # Very short decay
        config.adsr['sustain'] = 0.5  # 50% sustain level
        config.adsr['release'] = 0.1

        osc = Oscillator(440.0, 'sine')

        # Fast-forward past attack and decay
        frames_past_ad = int((config.adsr['attack'] + config.adsr['decay'] + 0.1) * config.sample_rate)
        audio, _ = osc.generate(frames_past_ad)

        # Now we should be in sustain
        frames = 1024
        audio, _ = osc.generate(frames)

        # Check that amplitude is stable at sustain level
        peak = np.max(np.abs(audio))
        assert abs(peak - config.adsr['sustain']) < 0.1

    def test_oscillator_envelope_release(self):
        """Test ADSR release phase."""
        config.adsr['attack'] = 0.01
        config.adsr['decay'] = 0.01
        config.adsr['sustain'] = 0.5
        config.adsr['release'] = 0.1

        osc = Oscillator(440.0, 'sine')

        # Get to sustain phase
        frames_to_sustain = int((config.adsr['attack'] + config.adsr['decay'] + 0.1) * config.sample_rate)
        osc.generate(frames_to_sustain)

        # Trigger release
        osc.released = True
        osc.env_time = 0.0  # Reset time for release phase

        frames = int(0.05 * config.sample_rate)  # 50ms of release
        audio, _ = osc.generate(frames)

        # Amplitude should decrease during release
        assert abs(audio[0]) > abs(audio[-1])

    def test_oscillator_done_state(self):
        """Test that oscillator marks itself as done after release."""
        config.adsr['attack'] = 0.01
        config.adsr['decay'] = 0.01
        config.adsr['sustain'] = 0.5
        config.adsr['release'] = 0.1

        osc = Oscillator(440.0, 'sine')

        # Trigger release
        osc.released = True
        osc.env_time = 0.0

        # Generate enough samples to complete release
        frames = int(config.adsr['release'] * config.sample_rate * 1.5)
        audio, _ = osc.generate(frames)

        assert osc.done

    def test_oscillator_velocity(self, audio_helper):
        """Test velocity scaling."""
        frames = 1024

        # Test different velocities
        for velocity in [0.1, 0.5, 1.0]:
            osc = Oscillator(440.0, 'sine')
            osc.velocity = velocity

            # Fast-forward past attack and decay to reach sustain phase
            frames_past_ad = int((config.adsr['attack'] + config.adsr['decay'] + 0.1) * config.sample_rate)
            osc.generate(frames_past_ad)

            # Now generate in sustain phase
            audio, _ = osc.generate(frames)
            peak = audio_helper.peak(audio)

            # Peak should be proportional to velocity in sustain phase
            expected_peak = velocity * config.adsr['sustain']  # More accurate in sustain
            assert abs(peak - expected_peak) < 0.1  # Tighter tolerance in sustain

    def test_oscillator_glide(self):
        """Test frequency glide functionality."""
        config.glide_time = 0.1  # 100ms glide

        osc = Oscillator(440.0, 'sine')
        osc.target_freq = 880.0  # Glide to octave

        frames = int(0.05 * config.sample_rate)  # 50ms of samples
        audio, _ = osc.generate(frames)

        # Frequency should be between start and target
        assert osc.freq > 440.0
        assert osc.freq < 880.0

    def test_oscillator_filter_envelope(self):
        """Test filter envelope generation."""
        config.filter_adsr['attack'] = 0.1
        config.filter_adsr['decay'] = 0.1
        config.filter_adsr['sustain'] = 0.5
        config.filter_adsr['release'] = 0.1

        osc = Oscillator(440.0, 'sine')
        frames = int(0.05 * config.sample_rate)  # 50ms

        audio, filter_env = osc.generate(frames)

        assert len(filter_env) == frames
        # Filter envelope should start at 0 and increase
        assert filter_env[0] == 0.0
        assert np.max(filter_env) > 0.0

    def test_oscillator_zero_frames(self):
        """Test handling of zero frame requests."""
        osc = Oscillator(440.0, 'sine')
        audio, filter_env = osc.generate(0)

        assert len(audio) == 0
        assert len(filter_env) == 0

    def test_oscillator_large_frames(self):
        """Test handling of large frame counts."""
        osc = Oscillator(440.0, 'sine')
        frames = 100000  # Large number of frames

        audio, filter_env = osc.generate(frames)

        assert len(audio) == frames
        assert len(filter_env) == frames
        assert not np.any(np.isnan(audio))
        assert not np.any(np.isinf(audio))


class TestSynthFunctions:
    """Test cases for synthesizer utility functions."""

    def test_mark_chorus_params_changed(self):
        """Test chorus parameter change marking."""
        mark_chorus_params_changed()
        # We can't directly test the global variable, but we can test that
        # update_chorus_params doesn't raise an exception
        update_chorus_params()

    def test_mark_delay_params_changed(self):
        """Test delay parameter change marking."""
        mark_delay_params_changed()
        # Similar to chorus, test that update doesn't raise an exception
        update_delay_params()

    @patch('qwerty_synth.synth.chorus')
    def test_update_chorus_params(self, mock_chorus):
        """Test chorus parameter updates."""
        config.chorus_rate = 1.0
        config.chorus_depth = 0.01
        config.chorus_mix = 0.3
        config.chorus_voices = 2

        mark_chorus_params_changed()
        update_chorus_params()

        mock_chorus.set_rate.assert_called_with(1.0)
        mock_chorus.set_depth.assert_called_with(0.01)
        mock_chorus.set_mix.assert_called_with(0.3)
        mock_chorus.set_voices.assert_called_with(2)

    @patch('qwerty_synth.synth.delay')
    def test_update_delay_params(self, mock_delay):
        """Test delay parameter updates."""
        config.delay_time_ms = 500

        mark_delay_params_changed()
        update_delay_params()

        mock_delay.set_time.assert_called_with(500)


class TestAudioCallback:
    """Test cases for the audio callback function."""

    def setup_method(self):
        """Set up test environment for audio callback tests."""
        config.active_notes = {}
        config.max_active_notes = 8

    def test_audio_callback_silent(self):
        """Test audio callback with no active notes."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)

        audio_callback(outdata, frames, None, None)

        # Output should be silent
        assert np.allclose(outdata, 0.0)

    def test_audio_callback_with_oscillator(self, audio_helper):
        """Test audio callback with one active oscillator."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)

        # Add an oscillator to active notes
        config.active_notes['test_key'] = Oscillator(440.0, 'sine')

        audio_callback(outdata, frames, None, None)

        # Output should have signal
        assert audio_helper.has_signal(outdata[:, 0])  # Left channel
        assert audio_helper.has_signal(outdata[:, 1])  # Right channel
        assert not audio_helper.is_clipped(outdata[:, 0])
        assert not audio_helper.is_clipped(outdata[:, 1])

    def test_audio_callback_multiple_oscillators(self, audio_helper):
        """Test audio callback with multiple oscillators."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)

        # Add multiple oscillators
        config.active_notes['key1'] = Oscillator(440.0, 'sine')
        config.active_notes['key2'] = Oscillator(554.37, 'sine')  # C#5
        config.active_notes['key3'] = Oscillator(659.25, 'sine')  # E5

        audio_callback(outdata, frames, None, None)

        # Output should have signal but not be clipped
        assert audio_helper.has_signal(outdata[:, 0])
        assert audio_helper.has_signal(outdata[:, 1])
        assert not audio_helper.is_clipped(outdata[:, 0])
        assert not audio_helper.is_clipped(outdata[:, 1])

    def test_audio_callback_max_notes_limit(self):
        """Test that audio callback respects max active notes limit."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)
        config.max_active_notes = 3

        # Add more oscillators than the limit
        for i in range(10):
            osc = Oscillator(440.0 + i * 50, 'sine')
            osc.env_time = i * 0.1  # Make them different ages
            config.active_notes[f'key{i}'] = osc

        audio_callback(outdata, frames, None, None)

        # Should only process the newest notes (up to max_active_notes)
        active_count = sum(1 for osc in config.active_notes.values() if not osc.released)
        assert active_count <= config.max_active_notes

    def test_audio_callback_cleanup_done_oscillators(self):
        """Test that finished oscillators are removed from active notes."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)

        # Create a finished oscillator
        osc = Oscillator(440.0, 'sine')
        osc.done = True
        config.active_notes['test_key'] = osc

        audio_callback(outdata, frames, None, None)

        # Finished oscillator should be removed
        assert 'test_key' not in config.active_notes

    def test_audio_callback_volume_control(self, audio_helper):
        """Test volume control in audio callback."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)

        config.active_notes['test_key'] = Oscillator(440.0, 'sine')

        # Test with different volume levels
        for volume in [0.1, 0.5, 1.0]:
            config.volume = volume
            outdata.fill(0.0)  # Reset output buffer

            audio_callback(outdata, frames, None, None)

            peak = audio_helper.peak(outdata)
            # Peak should be proportional to volume
            assert peak > 0.0
            # Allow some tolerance for envelope and other effects
            assert peak <= volume * 1.2

    @patch('qwerty_synth.synth.record.is_recording')
    @patch('qwerty_synth.synth.record.add_audio_block')
    def test_audio_callback_recording(self, mock_add_audio, mock_is_recording):
        """Test audio callback recording functionality."""
        mock_is_recording.return_value = True

        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)
        config.active_notes['test_key'] = Oscillator(440.0, 'sine')

        audio_callback(outdata, frames, None, None)

        # Recording should be called
        mock_add_audio.assert_called_once()
        args = mock_add_audio.call_args[0]
        recorded_audio = args[0]

        assert recorded_audio.shape == (frames, 2)

    def test_audio_callback_status_handling(self, capsys):
        """Test audio callback status message handling."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)
        status = "Test status message"

        audio_callback(outdata, frames, None, status)

        captured = capsys.readouterr()
        assert "Test status message" in captured.out

    @patch('qwerty_synth.filter.apply_filter')
    def test_audio_callback_filter_application(self, mock_filter):
        """Test that filters are applied when enabled."""
        config.filter_enabled = True
        config.filter_cutoff = 1000  # Well below Nyquist

        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)
        config.active_notes['test_key'] = Oscillator(440.0, 'sine')

        # Mock filter to return the input unchanged
        mock_filter.side_effect = lambda x, *args: x

        audio_callback(outdata, frames, None, None)

        # Filter should have been called
        mock_filter.assert_called_once()

    @patch('qwerty_synth.synth.apply_drive')
    def test_audio_callback_drive_application(self, mock_drive):
        """Test that drive effect is applied."""
        frames = 1024
        outdata = np.zeros((frames, 2), dtype=np.float32)
        config.active_notes['test_key'] = Oscillator(440.0, 'sine')

        # Mock drive to return the input unchanged
        mock_drive.side_effect = lambda x: x

        audio_callback(outdata, frames, None, None)

        # Drive should have been called
        mock_drive.assert_called_once()


class TestCreateAudioStream:
    """Test cases for audio stream creation."""

    @patch('sounddevice.OutputStream')
    def test_create_audio_stream_high_latency(self, mock_output_stream):
        """Test creating audio stream with high latency."""
        mock_stream = Mock()
        mock_output_stream.return_value = mock_stream

        stream = create_audio_stream('high')

        mock_output_stream.assert_called_once_with(
            samplerate=config.sample_rate,
            channels=2,
            callback=audio_callback,
            blocksize=config.blocksize,
            latency='high'
        )
        assert stream == mock_stream

    @patch('sounddevice.OutputStream')
    def test_create_audio_stream_low_latency(self, mock_output_stream):
        """Test creating audio stream with low latency."""
        mock_stream = Mock()
        mock_output_stream.return_value = mock_stream

        stream = create_audio_stream('low')

        mock_output_stream.assert_called_once_with(
            samplerate=config.sample_rate,
            channels=2,
            callback=audio_callback,
            blocksize=config.blocksize,
            latency='low'
        )
        assert stream == mock_stream

    @patch('sounddevice.OutputStream')
    def test_create_audio_stream_default_latency(self, mock_output_stream):
        """Test creating audio stream with default latency."""
        mock_stream = Mock()
        mock_output_stream.return_value = mock_stream

        stream = create_audio_stream()

        mock_output_stream.assert_called_once_with(
            samplerate=config.sample_rate,
            channels=2,
            callback=audio_callback,
            blocksize=config.blocksize,
            latency='high'
        )
        assert stream == mock_stream


class TestThreadSafety:
    """Test cases for thread safety in synthesizer components."""

    def test_active_notes_thread_safety(self):
        """Test thread-safe access to active notes."""
        def add_notes():
            for i in range(100):
                with config.notes_lock:
                    config.active_notes[f'thread1_key{i}'] = Oscillator(440.0, 'sine')
                time.sleep(0.001)  # Small delay to increase contention

        def remove_notes():
            for i in range(100):
                with config.notes_lock:
                    key = f'thread1_key{i}'
                    if key in config.active_notes:
                        del config.active_notes[key]
                time.sleep(0.001)

        # Start two threads that modify active_notes
        thread1 = threading.Thread(target=add_notes)
        thread2 = threading.Thread(target=remove_notes)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should not crash or leave inconsistent state
        assert isinstance(config.active_notes, dict)

    def test_waveform_buffer_thread_safety(self):
        """Test thread-safe access to waveform buffer."""
        def modify_buffer():
            for i in range(100):
                with config.buffer_lock:
                    config.waveform_buffer = np.roll(config.waveform_buffer, -1)
                    config.waveform_buffer[-1] = i
                time.sleep(0.001)

        # Start multiple threads modifying the buffer
        threads = [threading.Thread(target=modify_buffer) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Buffer should still be valid
        assert isinstance(config.waveform_buffer, np.ndarray)
        assert len(config.waveform_buffer) > 0
