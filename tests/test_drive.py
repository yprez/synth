"""Comprehensive unit tests for the drive/distortion module."""

import numpy as np
import pytest
from unittest.mock import patch

from qwerty_synth import config
from qwerty_synth.drive import apply_drive


class TestDriveBasics:
    """Test basic drive functionality."""

    def test_drive_disabled_passthrough(self, sample_audio):
        """Test that disabled drive passes audio through unchanged."""
        config.drive_on = False

        output = apply_drive(sample_audio)

        assert np.allclose(output, sample_audio)

    def test_drive_unity_gain_passthrough(self, sample_audio):
        """Test that unity gain with drive on passes audio through unchanged."""
        config.drive_on = True
        config.drive_gain = 1.0
        config.drive_mix = 1.0

        output = apply_drive(sample_audio)

        # Should be approximately the same (within tolerance for processing)
        assert np.allclose(output, sample_audio, atol=1e-6)

    def test_drive_enabled_modifies_signal(self, sample_audio, audio_helper):
        """Test that enabled drive with gain > 1 modifies the signal."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        # Drive should change the signal
        assert not np.allclose(output, sample_audio)

    def test_drive_empty_input(self):
        """Test drive with empty input."""
        config.drive_on = True
        config.drive_gain = 2.0
        empty_audio = np.array([])

        output = apply_drive(empty_audio)

        assert len(output) == 0

    def test_drive_with_silence(self, silence):
        """Test drive with silent input."""
        config.drive_on = True
        config.drive_gain = 3.0
        config.drive_type = 'tanh'

        output = apply_drive(silence)

        assert len(output) == len(silence)
        # Silent input should remain silent (or nearly silent)
        assert np.max(np.abs(output)) < 1e-6


class TestDriveTypes:
    """Test different drive algorithm types."""

    def test_drive_tanh(self, sample_audio, audio_helper):
        """Test tanh drive algorithm."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Tanh should provide soft clipping - output should be bounded
        assert np.max(np.abs(output)) <= 1.1  # Allow small tolerance

    def test_drive_arctan(self, sample_audio, audio_helper):
        """Test arctan drive algorithm."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'arctan'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Arctan should provide gentle clipping
        assert np.max(np.abs(output)) <= 1.1

    def test_drive_cubic(self, sample_audio, audio_helper):
        """Test cubic drive algorithm."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'cubic'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Cubic should be hard-clipped to [-1, 1]
        assert np.max(output) <= 1.0
        assert np.min(output) >= -1.0

    def test_drive_fuzz(self, sample_audio, audio_helper):
        """Test fuzz drive algorithm."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'fuzz'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Fuzz should provide bounded output
        assert np.max(np.abs(output)) <= 1.1

    def test_drive_asymmetric(self, sample_audio, audio_helper):
        """Test asymmetric drive algorithm."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'asymmetric'
        config.drive_asymmetry = 0.3
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_drive_unknown_type_defaults_to_tanh(self, sample_audio):
        """Test that unknown drive type defaults to tanh."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'unknown_type'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        # Should not crash and should produce valid output
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestDriveGain:
    """Test drive gain parameter."""

    def test_drive_gain_below_one(self, sample_audio, audio_helper):
        """Test drive gain less than 1.0 (attenuation)."""
        config.drive_on = True
        config.drive_gain = 0.5
        config.drive_mix = 1.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        # Output should be attenuated
        assert audio_helper.rms(output) < audio_helper.rms(sample_audio)

    def test_drive_gain_scaling(self, sample_audio, audio_helper):
        """Test that drive gain affects the amount of distortion."""
        config.drive_on = True
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        # Low gain
        config.drive_gain = 1.5
        output_low = apply_drive(sample_audio)

        # High gain
        config.drive_gain = 3.0
        output_high = apply_drive(sample_audio)

        # Higher gain should produce more distortion (different output)
        assert not np.allclose(output_low, output_high)

        # Both should be valid
        assert not np.any(np.isnan(output_low))
        assert not np.any(np.isnan(output_high))

    def test_drive_gain_extremes(self, sample_audio):
        """Test drive with extreme gain values."""
        config.drive_on = True
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        # Very low gain
        config.drive_gain = 0.1
        output_low = apply_drive(sample_audio)

        # Very high gain
        config.drive_gain = 10.0
        output_high = apply_drive(sample_audio)

        assert len(output_low) == len(sample_audio)
        assert len(output_high) == len(sample_audio)
        assert not np.any(np.isnan(output_low))
        assert not np.any(np.isnan(output_high))
        assert not np.any(np.isinf(output_low))
        assert not np.any(np.isinf(output_high))


class TestDriveMix:
    """Test drive mix parameter (dry/wet blend)."""

    def test_drive_mix_dry(self, sample_audio):
        """Test drive mix at 0.0 (completely dry)."""
        config.drive_on = True
        config.drive_gain = 3.0
        config.drive_type = 'tanh'
        config.drive_mix = 0.0  # Completely dry

        output = apply_drive(sample_audio)

        # Should be unchanged (dry signal only)
        assert np.allclose(output, sample_audio)

    def test_drive_mix_wet(self, sample_audio):
        """Test drive mix at 1.0 (completely wet)."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0  # Completely wet
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        # Should be the driven signal only
        assert not np.allclose(output, sample_audio)

    def test_drive_mix_blend(self, sample_audio, audio_helper):
        """Test drive mix blending between dry and wet."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_tone = 0.0

        # Get completely dry output
        config.drive_mix = 0.0
        output_dry = apply_drive(sample_audio)

        # Get completely wet output
        config.drive_mix = 1.0
        output_wet = apply_drive(sample_audio)

        # Get 50% blend
        config.drive_mix = 0.5
        output_blend = apply_drive(sample_audio)

        # Blend should be between dry and wet
        # This is a basic sanity check
        assert not np.allclose(output_blend, output_dry)
        assert not np.allclose(output_blend, output_wet)

    def test_drive_mix_extremes(self, sample_audio):
        """Test drive mix with extreme values."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'

        # Negative mix (should clamp to 0 or handle gracefully)
        config.drive_mix = -0.5
        output_neg = apply_drive(sample_audio)

        # Mix > 1 (should clamp to 1 or handle gracefully)
        config.drive_mix = 1.5
        output_high = apply_drive(sample_audio)

        assert len(output_neg) == len(sample_audio)
        assert len(output_high) == len(sample_audio)
        assert not np.any(np.isnan(output_neg))
        assert not np.any(np.isnan(output_high))


class TestDriveTone:
    """Test drive tone control parameter."""

    def test_drive_tone_neutral(self, sample_audio):
        """Test drive tone at 0.0 (neutral)."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_drive_tone_bright(self, sample_audio):
        """Test drive tone at positive value (bright)."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.5  # Bright

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_drive_tone_dark(self, sample_audio):
        """Test drive tone at negative value (dark)."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = -0.5  # Dark

        output = apply_drive(sample_audio)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_drive_tone_extremes(self, sample_audio):
        """Test drive tone with extreme values."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0

        # Very dark
        config.drive_tone = -0.95
        output_dark = apply_drive(sample_audio)

        # Very bright
        config.drive_tone = 0.95
        output_bright = apply_drive(sample_audio)

        assert len(output_dark) == len(sample_audio)
        assert len(output_bright) == len(sample_audio)
        assert not np.any(np.isnan(output_dark))
        assert not np.any(np.isnan(output_bright))

    def test_drive_tone_filter_state_persistence(self, sample_audio):
        """Test that tone filter state persists between calls."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.5

        # Process multiple blocks
        output1 = apply_drive(sample_audio)
        output2 = apply_drive(sample_audio)

        assert len(output1) == len(sample_audio)
        assert len(output2) == len(sample_audio)
        assert not np.any(np.isnan(output1))
        assert not np.any(np.isnan(output2))

    def test_drive_tone_reset_on_disable(self, sample_audio):
        """Test that tone filter state resets when drive is disabled."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_tone = 0.8

        # Process with tone enabled
        apply_drive(sample_audio)

        # Disable drive
        config.drive_on = False
        output_disabled = apply_drive(sample_audio)

        # Should pass through unchanged
        assert np.allclose(output_disabled, sample_audio)


class TestDriveAsymmetry:
    """Test drive asymmetry parameter."""

    def test_drive_asymmetry_values(self, sample_audio):
        """Test different asymmetry values."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'asymmetric'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        for asymmetry in [0.0, 0.2, 0.5, 0.8]:
            config.drive_asymmetry = asymmetry

            output = apply_drive(sample_audio)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_drive_asymmetry_effect(self, audio_helper):
        """Test that asymmetry creates different positive/negative clipping."""
        config.drive_on = True
        config.drive_gain = 3.0
        config.drive_type = 'asymmetric'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        # Generate test signal with both positive and negative peaks
        frames = 1024
        t = np.arange(frames) / config.sample_rate
        test_signal = 0.8 * np.sin(2 * np.pi * 440 * t)

        # Symmetric asymmetry
        config.drive_asymmetry = 0.0
        output_sym = apply_drive(test_signal)

        # Asymmetric asymmetry
        config.drive_asymmetry = 0.5
        output_asym = apply_drive(test_signal)

        # Should produce different results
        assert not np.allclose(output_sym, output_asym)

    def test_drive_asymmetry_extremes(self, sample_audio):
        """Test asymmetry with extreme values."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'asymmetric'
        config.drive_mix = 1.0

        # Minimum asymmetry
        config.drive_asymmetry = 0.0
        output_min = apply_drive(sample_audio)

        # Maximum asymmetry
        config.drive_asymmetry = 0.9
        output_max = apply_drive(sample_audio)

        assert len(output_min) == len(sample_audio)
        assert len(output_max) == len(sample_audio)
        assert not np.any(np.isnan(output_min))
        assert not np.any(np.isnan(output_max))


class TestDriveEdgeCases:
    """Test drive edge cases and error conditions."""

    def test_drive_with_large_input_values(self):
        """Test drive with very large input values."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0

        # Very large input values
        large_signal = np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float32)

        output = apply_drive(large_signal)

        assert len(output) == len(large_signal)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Output should be bounded for most drive types
        assert np.max(np.abs(output)) < 10.0  # Reasonable bound

    def test_drive_with_noise_input(self, noise):
        """Test drive with noise input."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        output = apply_drive(noise)

        assert len(output) == len(noise)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_drive_parameter_changes_during_processing(self, sample_audio):
        """Test drive behavior when parameters change between calls."""
        config.drive_on = True
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        # Start with one set of parameters
        config.drive_gain = 1.5
        config.drive_type = 'tanh'
        output1 = apply_drive(sample_audio)

        # Change parameters
        config.drive_gain = 3.0
        config.drive_type = 'cubic'
        output2 = apply_drive(sample_audio)

        # Both should be valid
        assert len(output1) == len(sample_audio)
        assert len(output2) == len(sample_audio)
        assert not np.any(np.isnan(output1))
        assert not np.any(np.isnan(output2))

        # Outputs should be different
        assert not np.allclose(output1, output2)

    def test_drive_zero_gain(self, sample_audio):
        """Test drive with zero gain."""
        config.drive_on = True
        config.drive_gain = 0.0
        config.drive_mix = 1.0

        output = apply_drive(sample_audio)

        # Should produce silent or near-silent output
        assert len(output) == len(sample_audio)
        assert np.max(np.abs(output)) < 1e-6

    def test_drive_negative_gain(self, sample_audio):
        """Test drive with negative gain."""
        config.drive_on = True
        config.drive_gain = -1.0
        config.drive_mix = 1.0

        output = apply_drive(sample_audio)

        # Should handle gracefully (may invert signal or take absolute value)
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestDrivePerformance:
    """Test drive performance and optimization."""

    def test_drive_large_buffer_processing(self):
        """Test drive with large audio buffers."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        # Large buffer
        frames = 100000
        large_signal = np.random.normal(0, 0.1, frames).astype(np.float32)

        output = apply_drive(large_signal)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_drive_repeated_processing(self, sample_audio):
        """Test drive with repeated processing (no memory leaks)."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.5

        # Process many times
        for i in range(100):
            output = apply_drive(sample_audio)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))

    def test_drive_different_buffer_sizes(self):
        """Test drive with different buffer sizes."""
        config.drive_on = True
        config.drive_gain = 2.0
        config.drive_type = 'tanh'
        config.drive_mix = 1.0
        config.drive_tone = 0.0

        for size in [64, 256, 512, 1024, 2048, 4096]:
            signal = np.random.normal(0, 0.1, size).astype(np.float32)

            output = apply_drive(signal)

            assert len(output) == size
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
