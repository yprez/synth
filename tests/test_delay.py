"""Comprehensive unit tests for the delay effect module."""

import numpy as np

from qwerty_synth import config
from qwerty_synth.delay import Delay, DIV2MULT


class TestDelayBasics:
    """Test basic delay functionality."""

    def test_delay_initialization(self):
        """Test delay initialization with default parameters."""
        delay = Delay()

        assert delay.sample_rate == config.sample_rate
        assert delay.delay_ms > 0
        assert delay.delay_samples >= 0
        assert delay._buf_len >= 0
        assert delay._mask >= 0

    def test_delay_initialization_custom_params(self):
        """Test delay initialization with custom parameters."""
        sample_rate = 48000
        delay_ms = 500

        delay = Delay(sample_rate=sample_rate, delay_ms=delay_ms)

        assert delay.sample_rate == sample_rate
        assert delay.delay_ms == delay_ms

    def test_delay_set_time(self):
        """Test setting delay time."""
        delay = Delay()

        new_time = 250  # ms
        delay.set_time(new_time)

        assert delay.delay_ms == new_time
        assert delay.delay_samples > 0

    def test_delay_set_time_with_interpolation(self):
        """Test setting delay time with interpolation enabled."""
        delay = Delay()

        new_time = 333.33  # Non-integer sample delay
        delay.set_time(new_time, use_interpolation=True)

        assert delay.delay_ms == new_time
        assert delay.delay_samples_f > delay.delay_samples

    def test_delay_set_time_without_interpolation(self):
        """Test setting delay time with interpolation disabled."""
        delay = Delay()

        new_time = 333.33
        delay.set_time(new_time, use_interpolation=False)

        assert delay.delay_ms == new_time
        assert delay.delay_samples_f == delay.delay_samples


class TestDelayProcessing:
    """Test delay audio processing."""

    def test_delay_process_block_basic(self, sample_audio):
        """Test basic delay processing."""
        delay = Delay()
        delay.set_time(100)  # 100ms delay

        feedback = 0.3
        mix = 0.5

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_delay_process_block_zero_feedback(self, sample_audio):
        """Test delay processing with zero feedback."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.0
        mix = 1.0  # Full wet

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        # With zero feedback, should just be delayed signal

    def test_delay_process_block_zero_mix(self, sample_audio):
        """Test delay processing with zero mix (dry signal only)."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.5
        mix = 0.0  # Completely dry

        output = delay.process_block(sample_audio, feedback, mix)

        # Should be approximately the same as input (dry signal)
        assert np.allclose(output, sample_audio, rtol=1e-5)

    def test_delay_process_block_full_mix(self, sample_audio):
        """Test delay processing with full mix (wet signal only)."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.0  # No feedback to avoid build-up
        mix = 1.0  # Completely wet

        # Process first block (will be mostly silent due to empty buffer)
        output1 = delay.process_block(sample_audio, feedback, mix)

        # Process second block (should have delayed signal from first block)
        output2 = delay.process_block(sample_audio, feedback, mix)

        assert len(output1) == len(sample_audio)
        assert len(output2) == len(sample_audio)

    def test_delay_process_block_with_feedback(self, sample_audio, audio_helper):
        """Test delay processing with feedback."""
        delay = Delay()
        delay.set_time(50)  # Short delay for testing

        feedback = 0.5
        mix = 0.8

        # Process multiple blocks to build up feedback
        output1 = delay.process_block(sample_audio, feedback, mix)
        output2 = delay.process_block(sample_audio, feedback, mix)
        output3 = delay.process_block(sample_audio, feedback, mix)

        for output in [output1, output2, output3]:
            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

        # Signal should build up with feedback
        assert audio_helper.rms(output3) >= audio_helper.rms(output1)

    def test_delay_process_block_high_feedback(self, sample_audio):
        """Test delay processing with high feedback (stability test)."""
        delay = Delay()
        delay.set_time(50)

        feedback = 0.9  # High feedback
        mix = 0.5

        # Process multiple blocks
        for i in range(10):
            output = delay.process_block(sample_audio, feedback, mix)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
            # Check that signal doesn't explode
            assert np.max(np.abs(output)) < 10.0

    def test_delay_process_block_interpolation(self, sample_audio):
        """Test delay processing with interpolation."""
        delay = Delay()

        # Fractional delay time requiring interpolation
        delay.set_time(123.45, use_interpolation=True)

        feedback = 0.3
        mix = 0.5

        output = delay.process_block(sample_audio, feedback, mix, use_interpolation=True)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_delay_process_block_no_interpolation(self, sample_audio):
        """Test delay processing without interpolation."""
        delay = Delay()
        delay.set_time(123.45, use_interpolation=False)

        feedback = 0.3
        mix = 0.5

        output = delay.process_block(sample_audio, feedback, mix, use_interpolation=False)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestDelayPingPong:
    """Test ping-pong stereo delay functionality."""

    def test_delay_pingpong_basic(self, stereo_audio):
        """Test basic ping-pong delay processing."""
        delay = Delay()
        delay.set_time(100)

        left, right = stereo_audio
        mix = 0.5
        feedback = 0.3

        output_L, output_R = delay.pingpong(left, right, mix, feedback)

        assert len(output_L) == len(left)
        assert len(output_R) == len(right)
        assert not np.any(np.isnan(output_L))
        assert not np.any(np.isnan(output_R))
        assert not np.any(np.isinf(output_L))
        assert not np.any(np.isinf(output_R))

    def test_delay_pingpong_zero_mix(self, stereo_audio):
        """Test ping-pong delay with zero mix (dry signal)."""
        delay = Delay()
        delay.set_time(100)

        left, right = stereo_audio
        mix = 0.0  # Completely dry
        feedback = 0.5

        output_L, output_R = delay.pingpong(left, right, mix, feedback)

        # Should be approximately the same as input
        assert np.allclose(output_L, left, rtol=1e-5)
        assert np.allclose(output_R, right, rtol=1e-5)

    def test_delay_pingpong_cross_feedback(self, stereo_audio):
        """Test ping-pong delay cross-feedback behavior."""
        delay = Delay()
        delay.set_time(50)  # Short delay

        left, right = stereo_audio
        mix = 0.8
        feedback = 0.6

        # Process multiple blocks to see cross-feedback effect
        for i in range(5):
            output_L, output_R = delay.pingpong(left, right, mix, feedback)

            assert len(output_L) == len(left)
            assert len(output_R) == len(right)
            assert not np.any(np.isnan(output_L))
            assert not np.any(np.isnan(output_R))

    def test_delay_pingpong_with_interpolation(self, stereo_audio):
        """Test ping-pong delay with interpolation."""
        delay = Delay()
        delay.set_time(87.5, use_interpolation=True)  # Fractional delay

        left, right = stereo_audio
        mix = 0.5
        feedback = 0.3

        output_L, output_R = delay.pingpong(left, right, mix, feedback, use_interpolation=True)

        assert len(output_L) == len(left)
        assert len(output_R) == len(right)
        assert not np.any(np.isnan(output_L))
        assert not np.any(np.isnan(output_R))


class TestDelayBPMSync:
    """Test BPM synchronization functionality."""

    def test_delay_update_from_bpm_basic(self):
        """Test basic BPM delay time calculation."""
        delay = Delay()

        bpm = 120
        division = '1/4'  # Quarter note

        delay.update_delay_from_bpm(bpm, division)

        # Quarter note at 120 BPM should be 500ms
        expected_ms = (60.0 / bpm) * 4 * DIV2MULT[division] * 1000
        assert abs(delay.delay_ms - expected_ms) < 1.0

    def test_delay_update_from_bpm_different_divisions(self):
        """Test BPM delay with different note divisions."""
        delay = Delay()
        bpm = 120

        for division, multiplier in DIV2MULT.items():
            delay.update_delay_from_bpm(bpm, division)

            expected_ms = (60.0 / bpm) * 4 * multiplier * 1000
            assert abs(delay.delay_ms - expected_ms) < 1.0

    def test_delay_update_from_bpm_different_tempos(self):
        """Test BPM delay with different tempos."""
        delay = Delay()
        division = '1/8'

        for bpm in [60, 120, 140, 180]:
            delay.update_delay_from_bpm(bpm, division)

            expected_ms = (60.0 / bpm) * 4 * DIV2MULT[division] * 1000
            assert abs(delay.delay_ms - expected_ms) < 1.0

    def test_delay_bpm_division_constants(self):
        """Test that division constants are correct."""
        assert DIV2MULT['1/1'] == 1.0
        assert DIV2MULT['1/2'] == 0.5
        assert DIV2MULT['1/4'] == 0.25
        assert DIV2MULT['1/8'] == 0.125
        assert DIV2MULT['1/16'] == 0.0625

        # Check dotted and triplet values
        assert abs(DIV2MULT['1/8d'] - 0.1875) < 1e-6  # 1/8 * 1.5
        assert abs(DIV2MULT['1/16t'] - 0.0416667) < 1e-6  # 1/16 * 2/3


class TestDelayEdgeCases:
    """Test delay edge cases and error conditions."""

    def test_delay_zero_time(self, sample_audio):
        """Test delay with zero delay time."""
        delay = Delay()
        delay.set_time(0)

        feedback = 0.3
        mix = 0.5

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_delay_very_short_time(self, sample_audio):
        """Test delay with very short delay time."""
        delay = Delay()
        delay.set_time(0.1)  # 0.1 ms

        feedback = 0.3
        mix = 0.5

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_delay_very_long_time(self, sample_audio):
        """Test delay with very long delay time."""
        delay = Delay()
        delay.set_time(5000)  # 5 seconds

        feedback = 0.2
        mix = 0.3

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_delay_negative_feedback(self, sample_audio):
        """Test delay with negative feedback."""
        delay = Delay()
        delay.set_time(100)

        feedback = -0.3
        mix = 0.5

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_delay_feedback_above_one(self, sample_audio):
        """Test delay with feedback above 1.0."""
        delay = Delay()
        delay.set_time(100)

        feedback = 1.5  # Above 1.0
        mix = 0.5

        # Should handle gracefully (may clamp or allow)
        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_delay_negative_mix(self, sample_audio):
        """Test delay with negative mix value."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.3
        mix = -0.2

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_delay_mix_above_one(self, sample_audio):
        """Test delay with mix above 1.0."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.3
        mix = 1.5

        output = delay.process_block(sample_audio, feedback, mix)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_delay_empty_input(self):
        """Test delay with empty input."""
        delay = Delay()
        delay.set_time(100)

        empty_audio = np.array([], dtype=np.float32)

        output = delay.process_block(empty_audio, 0.3, 0.5)

        assert len(output) == 0

    def test_delay_with_silence(self, silence):
        """Test delay with silent input."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.5
        mix = 0.8

        output = delay.process_block(silence, feedback, mix)

        assert len(output) == len(silence)
        # Silent input should remain silent (or nearly silent)
        assert np.max(np.abs(output)) < 1e-6


class TestDelayBufferManagement:
    """Test delay buffer management and memory handling."""

    def test_delay_buffer_resize(self):
        """Test delay buffer resizing when delay time changes."""
        delay = Delay()

        # Start with short delay
        delay.set_time(50)
        initial_buf_len = delay._buf_len

        # Increase delay time significantly
        delay.set_time(1000)
        new_buf_len = delay._buf_len

        # Buffer should have grown
        assert new_buf_len >= initial_buf_len

    def test_delay_buffer_power_of_two(self):
        """Test that delay buffer size is power of two."""
        delay = Delay()

        for delay_ms in [100, 250, 500, 1000]:
            delay.set_time(delay_ms)

            # Buffer length should be power of 2
            buf_len = delay._buf_len
            assert buf_len > 0
            assert (buf_len & (buf_len - 1)) == 0  # Power of 2 check

    def test_delay_clear_cache(self):
        """Test delay cache clearing."""
        delay = Delay()
        delay.set_time(100)

        # This should not raise an exception
        delay.clear_cache()

    def test_delay_multiple_time_changes(self, sample_audio):
        """Test delay with multiple time changes during processing."""
        delay = Delay()

        for delay_time in [50, 200, 75, 300, 25]:
            delay.set_time(delay_time)

            output = delay.process_block(sample_audio, 0.3, 0.5)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))


class TestDelayPerformance:
    """Test delay performance and optimization."""

    def test_delay_large_buffer_processing(self):
        """Test delay with large audio buffers."""
        delay = Delay()
        delay.set_time(200)

        # Large buffer
        frames = 100000
        large_signal = np.random.normal(0, 0.1, frames).astype(np.float32)

        output = delay.process_block(large_signal, 0.3, 0.5)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_delay_repeated_processing(self, sample_audio):
        """Test delay with repeated processing (stability test)."""
        delay = Delay()
        delay.set_time(150)

        feedback = 0.4
        mix = 0.6

        # Process many times
        for i in range(100):
            output = delay.process_block(sample_audio, feedback, mix)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
            # Check stability
            assert np.max(np.abs(output)) < 10.0

    def test_delay_different_buffer_sizes(self):
        """Test delay with different buffer sizes."""
        delay = Delay()
        delay.set_time(100)

        feedback = 0.3
        mix = 0.5

        for size in [64, 256, 512, 1024, 2048, 4096]:
            signal = np.random.normal(0, 0.1, size).astype(np.float32)

            output = delay.process_block(signal, feedback, mix)

            assert len(output) == size
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_delay_pingpong_performance(self, stereo_audio):
        """Test ping-pong delay performance."""
        delay = Delay()
        delay.set_time(100)

        left, right = stereo_audio

        # Process multiple times
        for i in range(50):
            output_L, output_R = delay.pingpong(left, right, 0.5, 0.3)

            assert len(output_L) == len(left)
            assert len(output_R) == len(right)
            assert not np.any(np.isnan(output_L))
            assert not np.any(np.isnan(output_R))


class TestDelayConfiguration:
    """Test delay configuration and parameter handling."""

    def test_delay_sample_rate_changes(self, sample_audio):
        """Test delay behavior when sample rate changes."""
        # Create delay with one sample rate
        delay = Delay(sample_rate=44100, delay_ms=100)

        output1 = delay.process_block(sample_audio, 0.3, 0.5)

        # Create new delay with different sample rate
        delay2 = Delay(sample_rate=48000, delay_ms=100)

        # Should have different delay_samples but same delay_ms
        assert delay.delay_ms == delay2.delay_ms
        assert delay.delay_samples != delay2.delay_samples

    def test_delay_parameter_validation(self):
        """Test delay parameter validation and bounds checking."""
        delay = Delay()

        # These should not crash
        delay.set_time(-10)  # Negative time
        delay.set_time(0)    # Zero time
        delay.set_time(10000)  # Very large time

        # Delay should handle these gracefully
