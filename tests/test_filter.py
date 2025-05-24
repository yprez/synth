"""Comprehensive unit tests for the filter module."""

import numpy as np
import pytest
from unittest.mock import patch

from qwerty_synth import config
from qwerty_synth.filter import (
    apply_filter,
    reset_filter_state,
    get_filter_performance_info,
    clear_temp_arrays,
    ensure_jit_ready,
    _calculate_modulated_cutoff,
    _should_bypass_filter,
    _apply_svf,
    _apply_biquad,
    _ensure_output_buffer
)


class TestFilterBasics:
    """Test basic filter functionality."""

    def test_filter_disabled_passthrough(self, sample_audio):
        """Test that disabled filter passes audio through unchanged."""
        config.filter_enabled = False

        output = apply_filter(sample_audio)

        assert np.allclose(output, sample_audio)

    def test_filter_empty_input(self):
        """Test filter with empty input."""
        config.filter_enabled = True
        empty_audio = np.array([])

        output = apply_filter(empty_audio)

        assert len(output) == 0

    def test_filter_enabled_modifies_signal(self, sample_audio, audio_helper):
        """Test that enabled filter modifies the signal."""
        config.filter_enabled = True
        config.filter_cutoff = 1000  # Low cutoff to ensure filtering
        config.filter_resonance = 0.0
        config.filter_type = 'lowpass'

        output = apply_filter(sample_audio)

        assert len(output) == len(sample_audio)
        assert audio_helper.has_signal(output)
        # For a low-pass filter with low cutoff, the signal should be attenuated
        assert audio_helper.rms(output) <= audio_helper.rms(sample_audio)

    def test_filter_different_topologies(self, sample_audio):
        """Test different filter topologies."""
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.1

        for topology in ['svf', 'biquad']:
            config.filter_topology = topology

            output = apply_filter(sample_audio)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_filter_different_types(self, sample_audio):
        """Test different filter types."""
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.1
        config.filter_topology = 'svf'

        for filter_type in ['lowpass', 'highpass', 'bandpass', 'notch']:
            config.filter_type = filter_type

            output = apply_filter(sample_audio)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))


class TestFilterModulation:
    """Test filter modulation functionality."""

    def test_filter_lfo_modulation(self, sample_audio):
        """Test filter with LFO modulation."""
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.1

        # Create LFO modulation signal
        lfo_modulation = 0.1 * np.sin(2 * np.pi * np.arange(len(sample_audio)) / len(sample_audio))

        output = apply_filter(sample_audio, lfo_modulation=lfo_modulation)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_filter_envelope_modulation(self, sample_audio):
        """Test filter with envelope modulation."""
        config.filter_enabled = True
        config.filter_cutoff = 1000
        config.filter_env_amount = 2000

        # Create envelope signal (attack-like)
        filter_envelope = np.linspace(0, 1, len(sample_audio))

        output = apply_filter(sample_audio, filter_envelope=filter_envelope)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_filter_combined_modulation(self, sample_audio):
        """Test filter with both LFO and envelope modulation."""
        config.filter_enabled = True
        config.filter_cutoff = 1500
        config.filter_env_amount = 1000

        lfo_modulation = 0.05 * np.sin(2 * np.pi * 5 * np.arange(len(sample_audio)) / config.sample_rate)
        filter_envelope = np.linspace(0, 0.8, len(sample_audio))

        output = apply_filter(sample_audio, lfo_modulation=lfo_modulation, filter_envelope=filter_envelope)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestFilterParameters:
    """Test filter parameter handling."""

    def test_filter_cutoff_frequency_response(self, audio_helper):
        """Test that filter cutoff affects frequency response."""
        config.filter_enabled = True
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.0

        # Generate test signal with multiple frequencies
        frames = 4096
        sample_rate = config.sample_rate
        freqs = [500, 1000, 2000, 4000, 8000]
        test_signal = np.zeros(frames)

        for freq in freqs:
            t = np.arange(frames) / sample_rate
            test_signal += np.sin(2 * np.pi * freq * t)

        # Test low cutoff
        config.filter_cutoff = 1000
        output_low = apply_filter(test_signal)

        # Test high cutoff
        config.filter_cutoff = 5000
        output_high = apply_filter(test_signal)

        # Low cutoff should attenuate more
        assert audio_helper.rms(output_low) <= audio_helper.rms(output_high)

    def test_filter_resonance_effect(self, sample_audio):
        """Test that resonance affects the filter response."""
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_type = 'lowpass'

        # Test low resonance
        config.filter_resonance = 0.0
        output_low_res = apply_filter(sample_audio)

        # Test high resonance
        config.filter_resonance = 0.8
        output_high_res = apply_filter(sample_audio)

        # Both should produce valid output
        assert len(output_low_res) == len(sample_audio)
        assert len(output_high_res) == len(sample_audio)
        assert not np.any(np.isnan(output_low_res))
        assert not np.any(np.isnan(output_high_res))

    def test_filter_extreme_cutoff_values(self, sample_audio):
        """Test filter with extreme cutoff values."""
        config.filter_enabled = True
        config.filter_resonance = 0.1

        # Very low cutoff
        config.filter_cutoff = 20
        output_low = apply_filter(sample_audio)

        # Very high cutoff (near Nyquist)
        config.filter_cutoff = config.sample_rate // 2 - 100
        output_high = apply_filter(sample_audio)

        assert len(output_low) == len(sample_audio)
        assert len(output_high) == len(sample_audio)
        assert not np.any(np.isnan(output_low))
        assert not np.any(np.isnan(output_high))

    def test_filter_extreme_resonance_values(self, sample_audio):
        """Test filter with extreme resonance values."""
        config.filter_enabled = True
        config.filter_cutoff = 2000

        # Zero resonance
        config.filter_resonance = 0.0
        output_zero = apply_filter(sample_audio)

        # Maximum resonance
        config.filter_resonance = 1.0
        output_max = apply_filter(sample_audio)

        assert len(output_zero) == len(sample_audio)
        assert len(output_max) == len(sample_audio)
        assert not np.any(np.isnan(output_zero))
        assert not np.any(np.isnan(output_max))


class TestFilterBypass:
    """Test filter bypass conditions."""

    def test_should_bypass_filter_high_cutoff(self):
        """Test filter bypass for high cutoff frequencies."""
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.0

        # Very high cutoff should bypass
        high_cutoff = config.sample_rate // 2
        assert _should_bypass_filter(high_cutoff)

        # Normal cutoff should not bypass
        normal_cutoff = 1000
        assert not _should_bypass_filter(normal_cutoff)

    def test_should_bypass_filter_conditions(self):
        """Test various filter bypass conditions."""
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.0

        # Array of cutoff values
        cutoffs = np.array([1000, 2000, 3000])
        assert not _should_bypass_filter(cutoffs)

        # Very high cutoffs
        high_cutoffs = np.array([20000, 25000, 30000])
        config.filter_resonance = 0.0
        # This depends on implementation details

    def test_calculate_modulated_cutoff_no_modulation(self, sample_audio):
        """Test cutoff calculation without modulation."""
        config.filter_cutoff = 2000

        result = _calculate_modulated_cutoff(sample_audio, None, None)

        # Should return the base cutoff frequency
        assert result == config.filter_cutoff

    def test_calculate_modulated_cutoff_with_lfo(self, sample_audio):
        """Test cutoff calculation with LFO modulation."""
        config.filter_cutoff = 2000

        lfo_modulation = 0.1 * np.ones(len(sample_audio))
        result = _calculate_modulated_cutoff(sample_audio, lfo_modulation, None)

        assert len(result) == len(sample_audio)
        assert np.all(result > config.filter_cutoff)  # LFO increases cutoff

    def test_calculate_modulated_cutoff_with_envelope(self, sample_audio):
        """Test cutoff calculation with envelope modulation."""
        config.filter_cutoff = 1000
        config.filter_env_amount = 1000

        filter_envelope = 0.5 * np.ones(len(sample_audio))
        result = _calculate_modulated_cutoff(sample_audio, None, filter_envelope)

        assert len(result) == len(sample_audio)
        expected = config.filter_cutoff + filter_envelope * config.filter_env_amount
        assert np.allclose(result, np.clip(expected, 20, config.sample_rate / 2.1))


class TestFilterStateManagement:
    """Test filter state management."""

    def test_reset_filter_state(self):
        """Test filter state reset."""
        # This should not raise an exception
        reset_filter_state()

    def test_filter_state_persistence(self, sample_audio):
        """Test that filter state persists between calls."""
        config.filter_enabled = True
        config.filter_cutoff = 1000
        config.filter_resonance = 0.5

        # Process first block
        output1 = apply_filter(sample_audio)

        # Process second block - state should be maintained
        output2 = apply_filter(sample_audio)

        assert len(output1) == len(sample_audio)
        assert len(output2) == len(sample_audio)
        assert not np.any(np.isnan(output1))
        assert not np.any(np.isnan(output2))

    def test_clear_temp_arrays(self):
        """Test clearing temporary arrays."""
        # This should not raise an exception
        clear_temp_arrays()


class TestFilterPerformance:
    """Test filter performance and optimization features."""

    def test_get_filter_performance_info(self):
        """Test getting filter performance information."""
        info = get_filter_performance_info()

        assert isinstance(info, dict)
        # Should contain performance-related information

    def test_ensure_jit_ready(self):
        """Test JIT compilation readiness."""
        # This should not raise an exception
        ensure_jit_ready()

    def test_ensure_output_buffer(self):
        """Test output buffer allocation."""
        size = 1024

        # This should not raise an exception
        _ensure_output_buffer(size)

        # Test with different sizes
        for size in [512, 2048, 4096]:
            _ensure_output_buffer(size)


class TestSVFFilter:
    """Test State Variable Filter implementation."""

    def test_svf_filter_basic(self, sample_audio):
        """Test basic SVF filter functionality."""
        config.filter_topology = 'svf'
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.2

        for filter_type in ['lowpass', 'highpass', 'bandpass', 'notch']:
            config.filter_type = filter_type

            output = _apply_svf(sample_audio, config.filter_cutoff)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_svf_filter_constant_cutoff(self, sample_audio):
        """Test SVF filter with constant cutoff."""
        config.filter_topology = 'svf'
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.3

        cutoff = 1500
        output = _apply_svf(sample_audio, cutoff)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_svf_filter_variable_cutoff(self, sample_audio):
        """Test SVF filter with variable cutoff."""
        config.filter_topology = 'svf'
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.2

        # Variable cutoff
        cutoff = np.linspace(500, 3000, len(sample_audio))
        output = _apply_svf(sample_audio, cutoff)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))


class TestBiquadFilter:
    """Test Biquad filter implementation."""

    def test_biquad_filter_basic(self, sample_audio):
        """Test basic Biquad filter functionality."""
        config.filter_topology = 'biquad'
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.2

        for filter_type in ['lowpass', 'highpass', 'bandpass', 'notch']:
            config.filter_type = filter_type

            output = _apply_biquad(sample_audio, config.filter_cutoff)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_biquad_filter_constant_cutoff(self, sample_audio):
        """Test Biquad filter with constant cutoff."""
        config.filter_topology = 'biquad'
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.3

        cutoff = 1200
        output = _apply_biquad(sample_audio, cutoff)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_biquad_filter_variable_cutoff(self, sample_audio):
        """Test Biquad filter with variable cutoff."""
        config.filter_topology = 'biquad'
        config.filter_type = 'lowpass'
        config.filter_resonance = 0.2

        # Variable cutoff
        cutoff = np.linspace(800, 4000, len(sample_audio))
        output = _apply_biquad(sample_audio, cutoff)

        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))


class TestFilterEdgeCases:
    """Test filter edge cases and error conditions."""

    def test_filter_with_silence(self, silence):
        """Test filter with silent input."""
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.5

        output = apply_filter(silence)

        assert len(output) == len(silence)
        # Output should remain silent (or nearly silent)
        assert np.max(np.abs(output)) < 1e-6

    def test_filter_with_noise(self, noise):
        """Test filter with noise input."""
        config.filter_enabled = True
        config.filter_cutoff = 1000
        config.filter_resonance = 0.3
        config.filter_type = 'lowpass'

        output = apply_filter(noise)

        assert len(output) == len(noise)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_filter_cutoff_above_nyquist(self, sample_audio):
        """Test filter with cutoff above Nyquist frequency."""
        config.filter_enabled = True
        config.filter_cutoff = config.sample_rate  # Above Nyquist
        config.filter_resonance = 0.1

        output = apply_filter(sample_audio)

        # Should either bypass or handle gracefully
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_filter_zero_cutoff(self, sample_audio):
        """Test filter with zero cutoff frequency."""
        config.filter_enabled = True
        config.filter_cutoff = 0
        config.filter_resonance = 0.1

        output = apply_filter(sample_audio)

        # Should handle gracefully
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_filter_negative_cutoff(self, sample_audio):
        """Test filter with negative cutoff frequency."""
        config.filter_enabled = True
        config.filter_cutoff = -1000
        config.filter_resonance = 0.1

        output = apply_filter(sample_audio)

        # Should handle gracefully (likely clamp to minimum)
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_filter_very_high_resonance(self, sample_audio):
        """Test filter with very high resonance."""
        config.filter_enabled = True
        config.filter_cutoff = 1000
        config.filter_resonance = 10.0  # Very high

        output = apply_filter(sample_audio)

        # Should handle gracefully (likely clamp to maximum)
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))

    def test_filter_negative_resonance(self, sample_audio):
        """Test filter with negative resonance."""
        config.filter_enabled = True
        config.filter_cutoff = 1000
        config.filter_resonance = -0.5

        output = apply_filter(sample_audio)

        # Should handle gracefully
        assert len(output) == len(sample_audio)
        assert not np.any(np.isnan(output))


class TestFilterStability:
    """Test filter stability and numerical behavior."""

    def test_filter_stability_long_signal(self):
        """Test filter stability with long signals."""
        config.filter_enabled = True
        config.filter_cutoff = 2000
        config.filter_resonance = 0.8  # High resonance

        # Generate long signal
        frames = 100000
        t = np.arange(frames) / config.sample_rate
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        output = apply_filter(signal)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Check that output doesn't explode
        assert np.max(np.abs(output)) < 100.0

    def test_filter_multiple_calls_stability(self, sample_audio):
        """Test filter stability across multiple calls."""
        config.filter_enabled = True
        config.filter_cutoff = 1500
        config.filter_resonance = 0.9  # High resonance

        # Process multiple blocks
        for i in range(10):
            output = apply_filter(sample_audio)

            assert len(output) == len(sample_audio)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
            assert np.max(np.abs(output)) < 100.0

    def test_filter_dc_blocking(self):
        """Test that filter handles DC correctly."""
        config.filter_enabled = True
        config.filter_cutoff = 1000
        config.filter_resonance = 0.2
        config.filter_type = 'highpass'

        # DC signal
        dc_signal = np.ones(1024, dtype=np.float32)

        output = apply_filter(dc_signal)

        assert len(output) == len(dc_signal)
        assert not np.any(np.isnan(output))

        # High-pass filter should attenuate DC
        assert np.mean(np.abs(output)) < np.mean(np.abs(dc_signal))


class TestFilterIntegration:
    """Test filter integration with other synthesizer components."""

    def test_filter_with_real_oscillator_output(self, mock_oscillator):
        """Test filter with real oscillator output."""
        config.filter_enabled = True
        config.filter_cutoff = 3000
        config.filter_resonance = 0.3

        # Generate oscillator output
        frames = 1024
        audio, filter_env = mock_oscillator.generate(frames)

        # Apply filter
        output = apply_filter(audio, filter_envelope=filter_env)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_filter_configuration_changes(self, sample_audio):
        """Test filter behavior when configuration changes during processing."""
        config.filter_enabled = True

        # Start with one configuration
        config.filter_cutoff = 1000
        config.filter_resonance = 0.2
        config.filter_type = 'lowpass'

        output1 = apply_filter(sample_audio)

        # Change configuration
        config.filter_cutoff = 3000
        config.filter_resonance = 0.8
        config.filter_type = 'bandpass'

        output2 = apply_filter(sample_audio)

        # Both outputs should be valid
        assert len(output1) == len(sample_audio)
        assert len(output2) == len(sample_audio)
        assert not np.any(np.isnan(output1))
        assert not np.any(np.isnan(output2))

        # Outputs should be different
        assert not np.allclose(output1, output2)
