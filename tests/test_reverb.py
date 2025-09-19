"""Tests for the reverb effect module."""

import numpy as np
import pytest
from unittest.mock import patch
from qwerty_synth import config
from qwerty_synth.reverb import Reverb


class TestReverbBasics:
    """Test basic reverb functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset config to defaults
        config.reverb_enabled = False
        config.reverb_room_size = 0.5
        config.reverb_damping = 0.5
        config.reverb_mix = 0.25

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        return np.random.normal(0, 0.1, 512).astype(np.float32)

    def test_reverb_initialization(self):
        """Test reverb can be initialized."""
        reverb = Reverb()
        assert reverb is not None
        assert reverb.sample_rate == config.sample_rate

    def test_reverb_initialization_custom_rate(self):
        """Test reverb with custom sample rate."""
        reverb = Reverb(sample_rate=48000)
        assert reverb.sample_rate == 48000

    def test_reverb_disabled_passthrough(self, sample_audio):
        """Test that disabled reverb passes audio through unchanged."""
        reverb = Reverb()
        config.reverb_mix = 0.0

        L = sample_audio.copy()
        R = sample_audio.copy()

        out_L, out_R = reverb.process(L, R)

        # Should be identical when disabled
        np.testing.assert_array_equal(L, out_L)
        np.testing.assert_array_equal(R, out_R)

    def test_reverb_enabled_modifies_signal(self, sample_audio):
        """Test that enabled reverb modifies the signal."""
        reverb = Reverb()
        config.reverb_mix = 0.5

        L = sample_audio.copy()
        R = sample_audio.copy()

        out_L, out_R = reverb.process(L, R)

        # Should be different when enabled (may be subtle)
        # We check that it's not identical
        assert not np.array_equal(L, out_L) or not np.array_equal(R, out_R)

    def test_reverb_empty_input(self):
        """Test reverb with empty input."""
        reverb = Reverb()
        config.reverb_mix = 0.3

        L = np.array([], dtype=np.float32)
        R = np.array([], dtype=np.float32)

        out_L, out_R = reverb.process(L, R)

        assert len(out_L) == 0
        assert len(out_R) == 0

    def test_reverb_stereo_decorrelation(self, sample_audio):
        """Test that reverb creates stereo decorrelation."""
        reverb = Reverb()
        config.reverb_mix = 0.8

        # Use identical input for both channels
        L = sample_audio.copy()
        R = sample_audio.copy()

        out_L, out_R = reverb.process(L, R)

        # Outputs should be different due to decorrelation
        correlation = np.corrcoef(out_L, out_R)[0, 1]
        assert correlation < 0.99  # Should be decorrelated


class TestReverbParameters:
    """Test reverb parameter control."""

    def setup_method(self):
        """Set up test fixtures."""
        config.reverb_enabled = True
        config.reverb_room_size = 0.5
        config.reverb_damping = 0.5
        config.reverb_mix = 0.25

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        return np.random.normal(0, 0.1, 512).astype(np.float32)

    def test_room_size_parameter(self, sample_audio):
        """Test room size parameter affects output."""
        reverb = Reverb()
        config.reverb_mix = 0.5

        L = sample_audio.copy()
        R = sample_audio.copy()

        # Test small room
        reverb.set_room_size(0.1)
        out_L_small, out_R_small = reverb.process(L, R)

        # Clear state
        reverb.clear_cache()

        # Test large room
        reverb.set_room_size(0.9)
        out_L_large, out_R_large = reverb.process(L, R)

        # Different room sizes should produce different outputs
        assert not np.allclose(out_L_small, out_L_large, rtol=0.01)

    def test_damping_parameter(self, sample_audio):
        """Test damping parameter affects output."""
        reverb = Reverb()
        config.reverb_mix = 0.5

        L = sample_audio.copy()
        R = sample_audio.copy()

        # Test bright (low damping)
        reverb.set_damping(0.1)
        out_L_bright, out_R_bright = reverb.process(L, R)

        # Clear state
        reverb.clear_cache()

        # Test dark (high damping)
        reverb.set_damping(0.9)
        out_L_dark, out_R_dark = reverb.process(L, R)

        # Different damping should produce different outputs
        assert not np.allclose(out_L_bright, out_L_dark, rtol=0.01)

    def test_mix_parameter(self, sample_audio):
        """Test mix parameter controls wet/dry balance."""
        reverb = Reverb()

        L = sample_audio.copy()
        R = sample_audio.copy()

        # Test dry (no reverb)
        reverb.set_mix(0.0)
        out_L_dry, out_R_dry = reverb.process(L, R)

        # Test wet (full reverb)
        reverb.set_mix(1.0)
        out_L_wet, out_R_wet = reverb.process(L, R)

        # Dry should be closer to original
        dry_diff = np.mean(np.abs(L - out_L_dry))
        wet_diff = np.mean(np.abs(L - out_L_wet))
        assert dry_diff < wet_diff

    def test_parameter_clamping(self):
        """Test that parameters are clamped to valid ranges."""
        reverb = Reverb()

        # Test room size clamping
        reverb.set_room_size(-0.5)
        assert config.reverb_room_size >= 0.0

        reverb.set_room_size(1.5)
        assert config.reverb_room_size <= 1.0

        # Test damping clamping
        reverb.set_damping(-0.5)
        assert config.reverb_damping >= 0.0

        reverb.set_damping(1.5)
        assert config.reverb_damping <= 1.0

        # Test mix clamping
        reverb.set_mix(-0.5)
        assert config.reverb_mix >= 0.0

        reverb.set_mix(1.5)
        assert config.reverb_mix <= 1.0


class TestReverbStability:
    """Test reverb stability and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        config.reverb_enabled = True
        config.reverb_room_size = 0.5
        config.reverb_damping = 0.5
        config.reverb_mix = 0.25

    def test_reverb_stability_long_signal(self):
        """Test reverb with long audio signal."""
        reverb = Reverb()
        config.reverb_mix = 0.5

        # Long signal (10 seconds at 44.1kHz)
        signal_length = 441000
        L = np.random.normal(0, 0.1, signal_length).astype(np.float32)
        R = np.random.normal(0, 0.1, signal_length).astype(np.float32)

        # Process in chunks
        chunk_size = 1024
        for i in range(0, signal_length, chunk_size):
            end_idx = min(i + chunk_size, signal_length)
            chunk_L = L[i:end_idx]
            chunk_R = R[i:end_idx]

            out_L, out_R = reverb.process(chunk_L, chunk_R)

            # Check for stability (no infinite values)
            assert np.all(np.isfinite(out_L))
            assert np.all(np.isfinite(out_R))

            # Check for reasonable output levels
            assert np.max(np.abs(out_L)) < 10.0
            assert np.max(np.abs(out_R)) < 10.0

    def test_reverb_silence_input(self):
        """Test reverb with silent input."""
        reverb = Reverb()
        config.reverb_mix = 0.5

        L = np.zeros(1024, dtype=np.float32)
        R = np.zeros(1024, dtype=np.float32)

        out_L, out_R = reverb.process(L, R)

        # Output should be very small (reverb tail from previous processing)
        assert np.max(np.abs(out_L)) < 1e-3
        assert np.max(np.abs(out_R)) < 1e-3

    def test_reverb_extreme_parameters(self):
        """Test reverb with extreme parameter values."""
        reverb = Reverb()

        # Test extreme values
        test_signal = np.random.normal(0, 0.1, 512).astype(np.float32)

        # Maximum room size and damping
        reverb.set_room_size(1.0)
        reverb.set_damping(1.0)
        reverb.set_mix(1.0)

        out_L, out_R = reverb.process(test_signal, test_signal)

        # Should still be stable
        assert np.all(np.isfinite(out_L))
        assert np.all(np.isfinite(out_R))

        # Minimum values
        reverb.clear_cache()
        reverb.set_room_size(0.0)
        reverb.set_damping(0.0)
        reverb.set_mix(0.0)

        out_L, out_R = reverb.process(test_signal, test_signal)

        # Should still be stable
        assert np.all(np.isfinite(out_L))
        assert np.all(np.isfinite(out_R))

    def test_reverb_different_buffer_sizes(self):
        """Test reverb with various buffer sizes."""
        reverb = Reverb()
        config.reverb_mix = 0.3

        buffer_sizes = [1, 16, 64, 256, 1024, 2048]

        for size in buffer_sizes:
            L = np.random.normal(0, 0.1, size).astype(np.float32)
            R = np.random.normal(0, 0.1, size).astype(np.float32)

            out_L, out_R = reverb.process(L, R)

            assert len(out_L) == size
            assert len(out_R) == size
            assert np.all(np.isfinite(out_L))
            assert np.all(np.isfinite(out_R))


class TestReverbBufferManagement:
    """Test reverb buffer management and memory usage."""

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        reverb = Reverb()
        config.reverb_mix = 0.5

        # Process some audio to fill buffers
        test_signal = np.random.normal(0, 0.1, 1024).astype(np.float32)
        reverb.process(test_signal, test_signal)

        # Clear cache
        reverb.clear_cache()

        # Process silence - should be truly silent after clearing
        silence = np.zeros(512, dtype=np.float32)
        out_L, out_R = reverb.process(silence, silence)

        # Output should be very close to zero after cache clear
        assert np.max(np.abs(out_L)) < 1e-6
        assert np.max(np.abs(out_R)) < 1e-6

    def test_buffer_initialization(self):
        """Test that buffers are properly initialized."""
        reverb = Reverb()

        # Check that buffers exist and are zeroed
        for buffer in reverb.comb_buffers_L + reverb.comb_buffers_R:
            assert buffer is not None
            assert len(buffer) > 0
            assert np.all(buffer == 0.0)

        for buffer in reverb.allpass_buffers_L + reverb.allpass_buffers_R:
            assert buffer is not None
            assert len(buffer) > 0
            assert np.all(buffer == 0.0)

    def test_sample_rate_scaling(self):
        """Test that delay lengths scale with sample rate."""
        reverb_44k = Reverb(sample_rate=44100)
        reverb_48k = Reverb(sample_rate=48000)

        # Delays should be scaled proportionally
        ratio = 48000 / 44100

        for i in range(len(reverb_44k.comb_delays_L)):
            expected_delay = int(reverb_44k.comb_delays_L[i] * ratio)
            actual_delay = reverb_48k.comb_delays_L[i]
            # Allow for small rounding differences
            assert abs(expected_delay - actual_delay) <= 1


class TestReverbIntegration:
    """Test reverb integration with config system."""

    def setup_method(self):
        """Set up test fixtures."""
        config.reverb_enabled = True
        config.reverb_room_size = 0.5
        config.reverb_damping = 0.5
        config.reverb_mix = 0.25

    def test_config_integration(self):
        """Test that reverb reads from config correctly."""
        reverb = Reverb()

        # Set config values
        config.reverb_room_size = 0.7
        config.reverb_damping = 0.3
        config.reverb_mix = 0.4

        # Create test signal
        test_signal = np.random.normal(0, 0.1, 512).astype(np.float32)

        # Process - should pick up config values
        out_L, out_R = reverb.process(test_signal, test_signal)

        # Verify the reverb used the config values
        assert reverb.mix == config.reverb_mix

    def test_parameter_updates(self):
        """Test real-time parameter updates."""
        reverb = Reverb()
        test_signal = np.random.normal(0, 0.1, 512).astype(np.float32)

        # Initial processing
        reverb.set_mix(0.2)
        out1_L, out1_R = reverb.process(test_signal, test_signal)

        # Update parameters
        reverb.set_mix(0.8)
        reverb.set_room_size(0.9)
        reverb.set_damping(0.1)

        # Process again
        out2_L, out2_R = reverb.process(test_signal, test_signal)

        # Outputs should be different
        assert not np.allclose(out1_L, out2_L, rtol=0.05)


class TestReverbAlgorithm:
    """Test reverb algorithm characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        config.reverb_enabled = True
        config.reverb_room_size = 0.5
        config.reverb_damping = 0.5
        config.reverb_mix = 0.5

    def test_comb_filter_delays(self):
        """Test that comb filter delays are reasonable."""
        reverb = Reverb()

        # Check that delays are in reasonable range (5-50ms at 44.1kHz)
        min_delay = int(0.005 * 44100)  # 5ms
        max_delay = int(0.050 * 44100)  # 50ms

        for delay in reverb.comb_delays_L:
            assert min_delay <= delay <= max_delay

        for delay in reverb.comb_delays_R:
            assert min_delay <= delay <= max_delay

    def test_allpass_filter_delays(self):
        """Test that allpass filter delays are reasonable."""
        reverb = Reverb()

        # Allpass delays should be shorter (1-15ms at 44.1kHz)
        min_delay = int(0.001 * 44100)  # 1ms
        max_delay = int(0.015 * 44100)  # 15ms

        for delay in reverb.allpass_delays_L:
            assert min_delay <= delay <= max_delay

        for delay in reverb.allpass_delays_R:
            assert min_delay <= delay <= max_delay

    def test_stereo_decorrelation_delays(self):
        """Test that L/R channels have different delays for decorrelation."""
        reverb = Reverb()

        # L/R delays should be different for stereo effect
        for i in range(len(reverb.comb_delays_L)):
            assert reverb.comb_delays_L[i] != reverb.comb_delays_R[i]

        for i in range(len(reverb.allpass_delays_L)):
            assert reverb.allpass_delays_L[i] != reverb.allpass_delays_R[i]

    def test_impulse_response(self):
        """Test reverb impulse response characteristics."""
        reverb = Reverb()
        config.reverb_mix = 1.0  # Full wet for testing

        # Create impulse signal
        impulse = np.zeros(8192, dtype=np.float32)
        impulse[0] = 1.0

        # Process impulse
        out_L, out_R = reverb.process(impulse, impulse)

        # Check that we get a decaying response
        early_energy = np.sum(out_L[:1000] ** 2)
        late_energy = np.sum(out_L[4000:8000] ** 2)

        # Early reflections should have more energy than late tail
        assert early_energy > late_energy

        # Response should be non-zero (reverb is working)
        assert np.sum(out_L ** 2) > 0.01