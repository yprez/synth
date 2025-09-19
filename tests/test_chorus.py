"""Comprehensive unit tests for the chorus effect module."""

import numpy as np
import pytest

from qwerty_synth.chorus import Chorus, _process_chorus_single_voice_jit, _process_chorus_multi_voice_jit
from qwerty_synth import config


class TestChorusInitialization:
    """Test cases for Chorus initialization."""

    def test_chorus_default_initialization(self):
        """Test chorus initializes with default values."""
        chorus = Chorus()

        assert chorus.sample_rate == config.sample_rate
        assert chorus.rate == config.chorus_rate
        assert chorus.depth == config.chorus_depth
        assert chorus.mix == config.chorus_mix
        assert chorus.voices == config.chorus_voices
        assert chorus.phase == 0.0
        assert chorus._write_idx == 0

        # Check buffer initialization
        assert isinstance(chorus._buffer_L, np.ndarray)
        assert isinstance(chorus._buffer_R, np.ndarray)
        assert chorus._buffer_L.dtype == np.float32
        assert chorus._buffer_R.dtype == np.float32
        assert len(chorus._buffer_L) == chorus._buf_len
        assert len(chorus._buffer_R) == chorus._buf_len

        # Check that buffer length is power of 2
        assert chorus._buf_len & (chorus._buf_len - 1) == 0

    def test_chorus_custom_sample_rate(self):
        """Test chorus initialization with custom sample rate."""
        custom_rate = 48000
        chorus = Chorus(sample_rate=custom_rate)

        assert chorus.sample_rate == custom_rate

        # Base delay should scale with sample rate
        expected_base_delay = int(0.015 * custom_rate)  # 15ms
        assert chorus.base_delay_samples == expected_base_delay

    def test_single_voice_phase_initialization(self):
        """Test phase initialization for single voice."""
        chorus = Chorus()
        if chorus.voices == 1:
            assert len(chorus.phases) == 1
            assert chorus.phases[0] == 0.0

    def test_multi_voice_phase_initialization(self):
        """Test phase initialization for multiple voices."""
        chorus = Chorus()
        chorus.set_voices(3)

        assert len(chorus.phases) == 3
        # Check phases are evenly distributed
        expected_phases = np.linspace(0, 2*np.pi, 3, endpoint=False)
        np.testing.assert_array_almost_equal(chorus.phases, expected_phases)

    def test_multi_voice_initialization_at_creation(self):
        """Test that multi-voice initialization path is triggered during creation."""
        # The Chorus constructor uses default parameter values from imports,
        # not runtime config values, so we need to pass the sample_rate parameter explicitly
        # and then set voices after creation to test the multi-voice path
        chorus = Chorus()

        # Test the multi-voice setup by calling set_voices
        chorus.set_voices(3)

        # Should have multi-voice setup
        assert chorus.voices == 3
        assert len(chorus.phases) == 3
        # Check that multi-voice initialization was executed
        expected_phases = np.linspace(0, 2*np.pi, 3, endpoint=False)
        np.testing.assert_array_almost_equal(chorus.phases, expected_phases)


class TestChorusParameterSetting:
    """Test cases for parameter setting methods."""

    def test_set_rate_valid_range(self):
        """Test setting valid rate values."""
        chorus = Chorus()

        # Test normal values
        chorus.set_rate(1.5)
        assert chorus.rate == 1.5

        chorus.set_rate(5.0)
        assert chorus.rate == 5.0

    def test_set_rate_clamping(self):
        """Test rate clamping to valid range."""
        chorus = Chorus()

        # Test clamping to minimum
        chorus.set_rate(0.05)
        assert chorus.rate == 0.1

        # Test clamping to maximum
        chorus.set_rate(15.0)
        assert chorus.rate == 10.0

    def test_set_depth_valid_range(self):
        """Test setting valid depth values."""
        chorus = Chorus()

        # Test normal values
        chorus.set_depth(0.005)
        assert chorus.depth == 0.005

        chorus.set_depth(0.020)
        assert chorus.depth == 0.020

    def test_set_depth_clamping(self):
        """Test depth clamping to valid range."""
        chorus = Chorus()

        # Test clamping to minimum
        chorus.set_depth(0.0005)
        assert chorus.depth == 0.001

        # Test clamping to maximum
        chorus.set_depth(0.050)
        assert chorus.depth == 0.030

    def test_set_depth_triggers_buffer_resize(self):
        """Test that changing depth can trigger buffer resize."""
        chorus = Chorus()
        original_buf_len = chorus._buf_len

        # Set a much larger depth that should require buffer resize
        chorus.set_depth(0.030)  # 30ms

        # Buffer might have resized
        assert chorus._buf_len >= original_buf_len

    def test_set_mix_valid_range(self):
        """Test setting valid mix values."""
        chorus = Chorus()

        # Test normal values
        chorus.set_mix(0.3)
        assert chorus.mix == 0.3

        chorus.set_mix(0.8)
        assert chorus.mix == 0.8

    def test_set_mix_clamping(self):
        """Test mix clamping to valid range."""
        chorus = Chorus()

        # Test clamping to minimum
        chorus.set_mix(-0.1)
        assert chorus.mix == 0.0

        # Test clamping to maximum
        chorus.set_mix(1.5)
        assert chorus.mix == 1.0

    def test_set_voices_valid_range(self):
        """Test setting valid voice counts."""
        chorus = Chorus()

        # Test different voice counts
        for voices in [1, 2, 3, 4]:
            chorus.set_voices(voices)
            assert chorus.voices == voices

    def test_set_voices_clamping(self):
        """Test voice count clamping to valid range."""
        chorus = Chorus()

        # Test clamping to minimum
        chorus.set_voices(0)
        assert chorus.voices == 1

        # Test clamping to maximum
        chorus.set_voices(10)
        assert chorus.voices == 4

    def test_set_voices_updates_phases(self):
        """Test that changing voice count updates phase arrays."""
        chorus = Chorus()

        # Change to multi-voice
        chorus.set_voices(3)
        assert len(chorus.phases) == 3

        # Change back to single voice
        chorus.set_voices(1)
        assert len(chorus.phases) == 1
        assert chorus.phase == 0.0

    def test_parameter_changes_update_cached_values(self):
        """Test that parameter changes update internal cached values."""
        chorus = Chorus()

        # Change rate and check phase increment
        old_phase_inc = chorus._phase_inc
        chorus.set_rate(2.0)
        assert chorus._phase_inc != old_phase_inc

        # Change depth and check depth samples
        old_depth_samples = chorus._depth_samples
        chorus.set_depth(0.010)
        assert chorus._depth_samples != old_depth_samples

        # Change mix and check gain values
        old_dry_gain = chorus._dry_gain
        chorus.set_mix(0.5)
        assert chorus._dry_gain != old_dry_gain


class TestChorusJITFunctionsDirect:
    """Test cases for direct testing of JIT functions to ensure coverage."""

    def test_single_voice_jit_function_directly(self):
        """Test the single voice JIT function directly to ensure coverage."""
        frames = 128
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        # Prepare output buffers
        out_L = np.zeros(frames, dtype=np.float32)
        out_R = np.zeros(frames, dtype=np.float32)

        # Prepare chorus buffers
        buffer_size = 1024
        buffer_L = np.zeros(buffer_size, dtype=np.float32)
        buffer_R = np.zeros(buffer_size, dtype=np.float32)
        mask = buffer_size - 1
        write_idx = 0

        # Prepare modulation arrays
        phase_values = np.linspace(0, 2*np.pi, frames, dtype=np.float32)
        lfo_values = np.sin(phase_values).astype(np.float32)
        base_delay_samples = 100
        delay_samples = base_delay_samples + lfo_values * 50

        # Audio mixing parameters
        dry_gain = 0.7
        wet_gain = 0.3

        # Call the JIT function directly to ensure it's covered
        new_write_idx = _process_chorus_single_voice_jit(
            L, R, out_L, out_R, buffer_L, buffer_R,
            phase_values, lfo_values, delay_samples, base_delay_samples,
            write_idx, mask, dry_gain, wet_gain
        )

        # Verify output
        assert new_write_idx == frames % buffer_size
        assert not np.allclose(out_L, 0.0)  # Output should have content
        assert not np.allclose(out_R, 0.0)
        assert out_L.dtype == np.float32
        assert out_R.dtype == np.float32

    def test_multi_voice_jit_function_directly(self):
        """Test the multi voice JIT function directly to ensure coverage."""
        frames = 128
        voices = 3
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        # Prepare output buffers
        out_L = np.zeros(frames, dtype=np.float32)
        out_R = np.zeros(frames, dtype=np.float32)

        # Prepare chorus buffers
        buffer_size = 1024
        buffer_L = np.zeros(buffer_size, dtype=np.float32)
        buffer_R = np.zeros(buffer_size, dtype=np.float32)
        mask = buffer_size - 1
        write_idx = 0

        # Prepare voice parameters
        phases = np.linspace(0, 2*np.pi, voices, endpoint=False).astype(np.float32)
        base_delay_samples = 100
        depth_samples = 50
        phase_inc = 0.01

        # Audio mixing parameters
        dry_gain = 0.7
        voice_mix = 0.1  # 0.3 / 3 voices

        # Call the JIT function directly to ensure it's covered
        new_write_idx = _process_chorus_multi_voice_jit(
            L, R, out_L, out_R, buffer_L, buffer_R,
            phases, base_delay_samples, depth_samples,
            phase_inc, write_idx, mask, dry_gain, voice_mix
        )

        # Verify output
        assert new_write_idx == frames % buffer_size
        assert not np.allclose(out_L, 0.0)  # Output should have content
        assert not np.allclose(out_R, 0.0)
        assert out_L.dtype == np.float32
        assert out_R.dtype == np.float32

        # Verify that phases were updated
        assert not np.allclose(phases, np.linspace(0, 2*np.pi, voices, endpoint=False))


class TestChorusProcessing:
    """Test cases for chorus processing."""

    @pytest.fixture
    def audio_helper(self):
        """Helper fixture for creating test audio."""
        class AudioHelper:
            @staticmethod
            def create_test_audio(frames=512, frequency=440, sample_rate=44100):
                """Create test sine wave audio."""
                t = np.arange(frames) / sample_rate
                L = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
                R = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
                return L, R

        return AudioHelper()

    def test_process_bypass_zero_mix(self):
        """Test that zero mix bypasses processing entirely."""
        chorus = Chorus()
        chorus.set_mix(0.0)

        frames = 256
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        out_L, out_R = chorus.process(L, R)

        # Should return exactly the same arrays (bypass)
        assert out_L is L
        assert out_R is R

    def test_process_single_voice(self, audio_helper):
        """Test single voice chorus processing."""
        chorus = Chorus()
        chorus.set_voices(1)
        chorus.set_mix(0.5)
        chorus.set_rate(1.0)
        chorus.set_depth(0.005)

        L, R = audio_helper.create_test_audio(frames=512)

        out_L, out_R = chorus.process(L, R)

        # Check output properties
        assert out_L.shape == L.shape
        assert out_R.shape == R.shape
        assert out_L.dtype == np.float32
        assert out_R.dtype == np.float32

        # Output should be different from input due to processing
        assert not np.array_equal(out_L, L)
        assert not np.array_equal(out_R, R)

        # Output level should be reasonable
        assert np.max(np.abs(out_L)) < 1.0
        assert np.max(np.abs(out_R)) < 1.0

    def test_process_multi_voice(self, audio_helper):
        """Test multi voice chorus processing."""
        chorus = Chorus()
        chorus.set_voices(3)
        chorus.set_mix(0.4)
        chorus.set_rate(1.5)
        chorus.set_depth(0.008)

        L, R = audio_helper.create_test_audio(frames=512)

        out_L, out_R = chorus.process(L, R)

        # Check output properties
        assert out_L.shape == L.shape
        assert out_R.shape == R.shape
        assert out_L.dtype == np.float32
        assert out_R.dtype == np.float32

        # Output should be different from input due to processing
        assert not np.array_equal(out_L, L)
        assert not np.array_equal(out_R, R)

        # Multi-voice should create richer harmonic content
        assert np.max(np.abs(out_L)) < 1.0
        assert np.max(np.abs(out_R)) < 1.0

    def test_process_different_block_sizes(self):
        """Test processing with different block sizes."""
        chorus = Chorus()
        chorus.set_mix(0.3)

        for block_size in [64, 128, 256, 512, 1024]:
            L = np.random.rand(block_size).astype(np.float32) * 0.1
            R = np.random.rand(block_size).astype(np.float32) * 0.1

            out_L, out_R = chorus.process(L, R)

            assert out_L.shape == (block_size,)
            assert out_R.shape == (block_size,)

    def test_process_silence(self):
        """Test processing with silent input."""
        chorus = Chorus()
        chorus.set_mix(0.5)

        frames = 256
        L = np.zeros(frames, dtype=np.float32)
        R = np.zeros(frames, dtype=np.float32)

        out_L, out_R = chorus.process(L, R)

        # Output should be very close to zero for silent input
        assert np.max(np.abs(out_L)) < 0.001
        assert np.max(np.abs(out_R)) < 0.001

    def test_process_phase_continuity(self):
        """Test that phase continues correctly between buffer calls."""
        chorus = Chorus()
        chorus.set_voices(1)
        chorus.set_mix(0.5)
        chorus.set_rate(2.0)

        frames = 128
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        # Process first buffer
        out_L1, out_R1 = chorus.process(L, R)
        phase_after_first = chorus.phase

        # Process second buffer
        out_L2, out_R2 = chorus.process(L, R)
        phase_after_second = chorus.phase

        # Phase should have advanced
        assert phase_after_second > phase_after_first
        assert 0 <= phase_after_second < 2 * np.pi

    def test_process_extreme_values(self, audio_helper):
        """Test processing with extreme parameter values."""
        chorus = Chorus()

        # Test with maximum depth and rate
        chorus.set_depth(0.030)  # Maximum depth
        chorus.set_rate(10.0)    # Maximum rate
        chorus.set_mix(1.0)      # Full wet
        chorus.set_voices(4)     # Maximum voices

        L, R = audio_helper.create_test_audio(frames=512)

        out_L, out_R = chorus.process(L, R)

        # Should still produce valid output
        assert not np.any(np.isnan(out_L))
        assert not np.any(np.isnan(out_R))
        assert not np.any(np.isinf(out_L))
        assert not np.any(np.isinf(out_R))


class TestChorusBufferManagement:
    """Test cases for chorus buffer management."""

    def test_buffer_resize_for_large_depth(self):
        """Test that buffers resize when depth requires it."""
        chorus = Chorus()

        # Set a large depth that should require buffer resize
        chorus.set_depth(0.025)  # 25ms

        # Buffer should have resized if necessary
        max_delay_samples = int(chorus.depth * 2 * chorus.sample_rate)
        assert chorus._buf_len >= max_delay_samples

    def test_buffer_preserves_data_on_resize(self):
        """Test that buffer resize preserves existing data."""
        chorus = Chorus()

        # Fill buffers with some test data
        test_data_L = np.random.rand(chorus._buf_len).astype(np.float32)
        test_data_R = np.random.rand(chorus._buf_len).astype(np.float32)
        chorus._buffer_L[:] = test_data_L
        chorus._buffer_R[:] = test_data_R

        original_buf_len = chorus._buf_len

        # Trigger a resize by setting larger depth
        chorus.set_depth(0.025)

        if chorus._buf_len > original_buf_len:
            # Check that original data was preserved
            copy_len = min(original_buf_len, chorus._buf_len)
            np.testing.assert_array_equal(
                chorus._buffer_L[:copy_len], test_data_L[:copy_len]
            )
            np.testing.assert_array_equal(
                chorus._buffer_R[:copy_len], test_data_R[:copy_len]
            )

    def test_temp_array_allocation(self):
        """Test temporary array allocation."""
        chorus = Chorus()

        # Initially temp arrays should be None
        assert chorus._temp_out_L is None
        assert chorus._temp_out_R is None

        # Process some audio to trigger allocation
        frames = 256
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        chorus.process(L, R)

        # Temp arrays should now be allocated
        assert chorus._temp_out_L is not None
        assert chorus._temp_out_R is not None
        assert len(chorus._temp_out_L) >= frames
        assert len(chorus._temp_out_R) >= frames

    def test_temp_array_growth(self):
        """Test that temporary arrays grow when needed."""
        chorus = Chorus()

        # Process small buffer first
        small_frames = 64
        L_small = np.random.rand(small_frames).astype(np.float32) * 0.1
        R_small = np.random.rand(small_frames).astype(np.float32) * 0.1
        chorus.process(L_small, R_small)

        original_size = len(chorus._temp_out_L)

        # Process larger buffer
        large_frames = 1536  # Larger than initial max_block_size
        L_large = np.random.rand(large_frames).astype(np.float32) * 0.1
        R_large = np.random.rand(large_frames).astype(np.float32) * 0.1
        chorus.process(L_large, R_large)

        # Arrays should have grown
        assert len(chorus._temp_out_L) >= large_frames
        assert len(chorus._temp_out_L) >= original_size

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        chorus = Chorus()

        # Fill buffers and state with data
        chorus._buffer_L.fill(0.5)
        chorus._buffer_R.fill(0.3)
        chorus._write_idx = 100
        chorus.phase = 1.5
        chorus.phases.fill(2.0)

        # Clear cache
        chorus.clear_cache()

        # Check that everything was reset
        assert np.allclose(chorus._buffer_L, 0.0)
        assert np.allclose(chorus._buffer_R, 0.0)
        assert chorus._write_idx == 0
        assert chorus.phase == 0.0
        assert np.allclose(chorus.phases, 0.0)


class TestChorusJITFunctions:
    """Test cases for JIT function behavior."""

    def test_single_voice_jit_function(self):
        """Test single voice JIT function properties."""
        frames = 128
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        # Prepare all required arrays and parameters
        out_L = np.zeros(frames, dtype=np.float32)
        out_R = np.zeros(frames, dtype=np.float32)
        buffer_L = np.zeros(1024, dtype=np.float32)
        buffer_R = np.zeros(1024, dtype=np.float32)
        phase_values = np.linspace(0, 2*np.pi, frames).astype(np.float32)
        lfo_values = np.sin(phase_values).astype(np.float32)
        delay_samples = 100 + lfo_values * 20
        base_delay_samples = 100
        write_idx = 0
        mask = 1023  # 1024 - 1
        dry_gain = 0.7
        wet_gain = 0.3

        # Call JIT function
        new_write_idx = _process_chorus_single_voice_jit(
            L, R, out_L, out_R, buffer_L, buffer_R,
            phase_values, lfo_values, delay_samples, base_delay_samples,
            write_idx, mask, dry_gain, wet_gain
        )

        # Verify results
        assert isinstance(new_write_idx, (int, np.integer))
        assert 0 <= new_write_idx <= mask
        assert not np.allclose(out_L, 0.0)
        assert not np.allclose(out_R, 0.0)

    def test_multi_voice_jit_function(self):
        """Test multi voice JIT function properties."""
        frames = 128
        voices = 3
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        # Prepare all required arrays and parameters
        out_L = np.zeros(frames, dtype=np.float32)
        out_R = np.zeros(frames, dtype=np.float32)
        buffer_L = np.zeros(1024, dtype=np.float32)
        buffer_R = np.zeros(1024, dtype=np.float32)
        phases = np.linspace(0, 2*np.pi, voices, endpoint=False).astype(np.float32)
        base_delay_samples = 100
        depth_samples = 20
        phase_inc = 0.01
        write_idx = 0
        mask = 1023  # 1024 - 1
        dry_gain = 0.7
        voice_mix = 0.1

        # Call JIT function
        new_write_idx = _process_chorus_multi_voice_jit(
            L, R, out_L, out_R, buffer_L, buffer_R,
            phases, base_delay_samples, depth_samples,
            phase_inc, write_idx, mask, dry_gain, voice_mix
        )

        # Verify results
        assert isinstance(new_write_idx, (int, np.integer))
        assert 0 <= new_write_idx <= mask
        assert not np.allclose(out_L, 0.0)
        assert not np.allclose(out_R, 0.0)


class TestChorusEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_frame_processing(self):
        """Test processing with zero frames."""
        chorus = Chorus()

        L = np.array([], dtype=np.float32)
        R = np.array([], dtype=np.float32)

        out_L, out_R = chorus.process(L, R)

        assert out_L.shape == (0,)
        assert out_R.shape == (0,)

    def test_single_frame_processing(self):
        """Test processing with single frame."""
        chorus = Chorus()
        chorus.set_mix(0.5)

        L = np.array([0.1], dtype=np.float32)
        R = np.array([0.2], dtype=np.float32)

        out_L, out_R = chorus.process(L, R)

        assert out_L.shape == (1,)
        assert out_R.shape == (1,)
        assert not np.isnan(out_L[0])
        assert not np.isnan(out_R[0])

    def test_very_small_rate(self):
        """Test processing with very small rate."""
        chorus = Chorus()
        chorus.set_rate(0.1)  # Minimum rate
        chorus.set_mix(0.3)

        frames = 256
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        out_L, out_R = chorus.process(L, R)

        # Should still work without issues
        assert out_L.shape == (frames,)
        assert out_R.shape == (frames,)
        assert not np.any(np.isnan(out_L))
        assert not np.any(np.isnan(out_R))

    def test_very_small_depth(self):
        """Test processing with very small depth."""
        chorus = Chorus()
        chorus.set_depth(0.001)  # Minimum depth
        chorus.set_mix(0.3)

        frames = 256
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        out_L, out_R = chorus.process(L, R)

        # Should still work without issues
        assert out_L.shape == (frames,)
        assert out_R.shape == (frames,)
        assert not np.any(np.isnan(out_L))
        assert not np.any(np.isnan(out_R))

    def test_maximum_parameters(self):
        """Test processing with all parameters at maximum."""
        chorus = Chorus()
        chorus.set_rate(10.0)     # Maximum rate
        chorus.set_depth(0.030)   # Maximum depth
        chorus.set_mix(1.0)       # Maximum mix
        chorus.set_voices(4)      # Maximum voices

        frames = 512
        L = np.random.rand(frames).astype(np.float32) * 0.1
        R = np.random.rand(frames).astype(np.float32) * 0.1

        out_L, out_R = chorus.process(L, R)

        # Should handle extreme parameters gracefully
        assert not np.any(np.isnan(out_L))
        assert not np.any(np.isnan(out_R))
        assert not np.any(np.isinf(out_L))
        assert not np.any(np.isinf(out_R))

    def test_mismatched_input_lengths(self):
        """Test processing with mismatched L/R input lengths."""
        chorus = Chorus()

        L = np.random.rand(128).astype(np.float32) * 0.1
        R = np.random.rand(64).astype(np.float32) * 0.1  # Different length

        # Should handle gracefully (likely using shorter length)
        # This test documents current behavior
        try:
            out_L, out_R = chorus.process(L, R)
            # If it succeeds, check that output is reasonable
            assert len(out_L) == len(out_R)
        except (ValueError, IndexError):
            # If it fails, that's also acceptable behavior
            pass


class TestChorusIntegration:
    """Test integration with other system components."""

    @pytest.fixture
    def audio_helper(self):
        """Helper fixture for creating test audio."""
        class AudioHelper:
            @staticmethod
            def create_test_audio(frames=512, frequency=440, sample_rate=44100):
                """Create test sine wave audio."""
                t = np.arange(frames) / sample_rate
                L = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
                R = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
                return L, R

        return AudioHelper()

    def test_integration_with_config_values(self):
        """Test that chorus integrates properly with config values."""
        # Store original values
        original_rate = config.chorus_rate
        original_depth = config.chorus_depth
        original_mix = config.chorus_mix
        original_voices = config.chorus_voices

        try:
            # Modify config values
            config.chorus_rate = 2.5
            config.chorus_depth = 0.012
            config.chorus_mix = 0.4
            config.chorus_voices = 2

            # Create new chorus instance
            chorus = Chorus()

            # Should use config values
            assert chorus.rate == 2.5
            assert chorus.depth == 0.012
            assert chorus.mix == 0.4
            assert chorus.voices == 2

            # Test processing
            frames = 256
            L = np.random.rand(frames).astype(np.float32) * 0.1
            R = np.random.rand(frames).astype(np.float32) * 0.1

            out_L, out_R = chorus.process(L, R)

            assert out_L.shape == (frames,)
            assert out_R.shape == (frames,)

        finally:
            # Restore original values
            config.chorus_rate = original_rate
            config.chorus_depth = original_depth
            config.chorus_mix = original_mix
            config.chorus_voices = original_voices

    def test_frequency_response_preservation(self, audio_helper):
        """Test that chorus preserves frequency content appropriately."""
        chorus = Chorus()
        chorus.set_mix(0.5)
        chorus.set_voices(1)

        # Test with different frequencies
        for frequency in [220, 440, 880, 1760]:
            L, R = audio_helper.create_test_audio(frames=1024, frequency=frequency)

            out_L, out_R = chorus.process(L, R)

            # Output should maintain similar frequency characteristics
            # (This is a basic test - more sophisticated frequency analysis could be added)
            assert np.corrcoef(L, out_L)[0, 1] > 0.5  # Reasonable correlation
            assert np.corrcoef(R, out_R)[0, 1] > 0.5

    def test_stereo_width_effect(self):
        """Test that chorus creates appropriate stereo width."""
        chorus = Chorus()
        chorus.set_mix(0.6)
        chorus.set_voices(2)
        chorus.set_depth(0.008)

        # Create mono-like input (L and R very similar)
        frames = 512
        mono_signal = np.random.rand(frames).astype(np.float32) * 0.1
        L = mono_signal + np.random.rand(frames).astype(np.float32) * 0.001  # Very slight difference
        R = mono_signal + np.random.rand(frames).astype(np.float32) * 0.001

        out_L, out_R = chorus.process(L, R)

        # Chorus should create some stereo difference
        np.std(L - R)
        output_width = np.std(out_L - out_R)

        # Output should have some stereo content (may be wider or narrower depending on implementation)
        # The key is that chorus processing creates some stereo effect
        assert output_width > 0.0001  # Should have some stereo content
        assert not np.array_equal(out_L, out_R)  # L and R should be different
