"""Comprehensive unit tests for the LFO (Low Frequency Oscillator) module."""

import numpy as np

from qwerty_synth import config
from qwerty_synth.lfo import LFO


class TestLFO:
    """Test cases for the LFO class."""

    def test_lfo_initialization(self):
        """Test LFO initialization."""
        lfo = LFO()

        assert lfo.phase == 0.0
        assert lfo.env_time == 0.0

    def test_lfo_reset(self):
        """Test LFO reset functionality."""
        lfo = LFO()
        lfo.env_time = 1.0
        lfo.phase = 3.14

        lfo.reset()

        assert lfo.env_time == 0.0
        # Phase should not be reset by reset() method
        assert lfo.phase == 3.14

    def test_lfo_disabled_generates_zeros(self):
        """Test that disabled LFO generates zero output."""
        config.lfo_enabled = False
        config.lfo_depth = 0.5

        lfo = LFO()
        frames = 1024
        output = lfo.generate(frames)

        assert len(output) == frames
        assert np.allclose(output, 0.0)

    def test_lfo_zero_depth_generates_zeros(self):
        """Test that LFO with zero depth generates zero output."""
        config.lfo_enabled = True
        config.lfo_depth = 0.0

        lfo = LFO()
        frames = 1024
        output = lfo.generate(frames)

        assert len(output) == frames
        assert np.allclose(output, 0.0)

    def test_lfo_during_delay_generates_zeros(self):
        """Test that LFO generates zeros during delay period."""
        config.lfo_enabled = True
        config.lfo_depth = 0.5
        config.lfo_delay_time = 0.1  # 100ms delay
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = int(0.05 * config.sample_rate)  # 50ms worth of samples (less than delay)
        output = lfo.generate(frames)

        assert len(output) == frames
        assert np.allclose(output, 0.0)

    def test_lfo_after_delay_generates_signal(self, audio_helper):
        """Test that LFO generates signal after delay period."""
        config.lfo_enabled = True
        config.lfo_depth = 0.5
        config.lfo_delay_time = 0.05  # 50ms delay
        config.lfo_attack_time = 0.01  # Very short attack
        config.lfo_rate = 5.0

        lfo = LFO()

        # Fast-forward past delay and attack
        delay_frames = int((config.lfo_delay_time + config.lfo_attack_time + 0.01) * config.sample_rate)
        lfo.generate(delay_frames)

        # Now generate signal
        frames = 1024
        output = lfo.generate(frames)

        assert len(output) == frames
        assert audio_helper.has_signal(output)

    def test_lfo_attack_envelope(self):
        """Test LFO attack envelope functionality."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0  # No delay
        config.lfo_attack_time = 0.1  # 100ms attack
        config.lfo_rate = 5.0

        lfo = LFO()

        # Generate samples during attack phase
        frames = int(0.05 * config.sample_rate)  # 50ms (half of attack time)
        output = lfo.generate(frames)

        # Signal should be present but attenuated
        max_output = np.max(np.abs(output))
        assert max_output > 0.0
        assert max_output < config.lfo_depth  # Should be less than full depth

    def test_lfo_full_envelope(self):
        """Test LFO full envelope after attack."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.05  # 50ms attack
        config.lfo_rate = 5.0

        lfo = LFO()

        # Fast-forward past attack
        attack_frames = int((config.lfo_attack_time + 0.01) * config.sample_rate)
        lfo.generate(attack_frames)

        # Generate signal after attack
        frames = 1024
        output = lfo.generate(frames)

        # Should reach full depth
        max_output = np.max(np.abs(output))
        assert max_output >= config.lfo_depth * 0.9  # Allow small tolerance

    def test_lfo_frequency_accuracy(self, audio_helper):
        """Test that LFO generates expected frequency."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0  # No attack for cleaner frequency analysis
        config.lfo_rate = 2.0  # 2 Hz

        lfo = LFO()

        # Generate a longer signal for better frequency analysis
        # Use at least 4 seconds to capture multiple cycles
        frames = int(4.0 * config.sample_rate)  # 4 seconds
        output = lfo.generate(frames)

        dominant_freq = audio_helper.dominant_frequency(output, config.sample_rate)
        # Allow some tolerance for windowing effects
        assert abs(dominant_freq - config.lfo_rate) < 0.5

    def test_lfo_sine_waveform(self):
        """Test that LFO generates sine wave."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0  # No attack
        config.lfo_rate = 1.0  # 1 Hz for easy verification

        lfo = LFO()

        # Generate exactly one period at 1 Hz
        frames = config.sample_rate  # 1 second = 1 period
        output = lfo.generate(frames)

        # Check that it starts near zero
        assert abs(output[0]) < 0.1

        # Check that it reaches approximately +/- depth
        assert np.max(output) >= config.lfo_depth * 0.9
        assert np.min(output) <= -config.lfo_depth * 0.9

    def test_lfo_depth_scaling(self):
        """Test LFO depth scaling."""
        config.lfo_enabled = True
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 5.0

        test_depths = [0.1, 0.5, 1.0]

        for depth in test_depths:
            config.lfo_depth = depth
            lfo = LFO()

            # Use enough frames to capture at least one full cycle
            frames = int(config.sample_rate / config.lfo_rate * 1.5)  # 1.5 cycles
            output = lfo.generate(frames)

            max_output = np.max(np.abs(output))
            # Allow some tolerance
            assert max_output >= depth * 0.8
            assert max_output <= depth * 1.2

    def test_lfo_phase_continuity(self):
        """Test that LFO phase is continuous across generate calls."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = 100

        # Generate first block
        output1 = lfo.generate(frames)
        last_sample1 = output1[-1]

        # Generate second block
        output2 = lfo.generate(frames)
        first_sample2 = output2[0]

        # Check phase continuity - adjacent samples should be similar
        # Calculate expected next sample
        phase_increment = 2 * np.pi * config.lfo_rate / config.sample_rate
        expected_next = config.lfo_depth * np.sin(lfo.phase - phase_increment)

        # Allow reasonable tolerance
        assert abs(first_sample2 - expected_next) < 0.1

    def test_lfo_zero_frames(self):
        """Test LFO with zero frames."""
        lfo = LFO()
        output = lfo.generate(0)

        assert len(output) == 0

    def test_lfo_large_frames(self):
        """Test LFO with large frame count."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = 100000

        output = lfo.generate(frames)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestLFOModulation:
    """Test cases for LFO modulation methods."""

    def test_pitch_modulation_disabled_target(self):
        """Test pitch modulation when LFO target is not pitch."""
        config.lfo_target = 'volume'  # Not pitch

        lfo = LFO()
        freq_array = np.array([440.0, 440.0, 440.0, 440.0])
        lfo_values = np.array([0.1, -0.1, 0.2, -0.2])

        result = lfo.apply_pitch_modulation(freq_array, lfo_values)

        # Should return unchanged frequency array
        assert np.allclose(result, freq_array)

    def test_pitch_modulation_zero_lfo(self):
        """Test pitch modulation with zero LFO signal."""
        config.lfo_target = 'pitch'

        lfo = LFO()
        freq_array = np.array([440.0, 440.0, 440.0, 440.0])
        lfo_values = np.array([0.0, 0.0, 0.0, 0.0])

        result = lfo.apply_pitch_modulation(freq_array, lfo_values)

        # Should return unchanged frequency array
        assert np.allclose(result, freq_array)

    def test_pitch_modulation_vibrato(self):
        """Test pitch modulation creates vibrato effect."""
        config.lfo_target = 'pitch'

        lfo = LFO()
        base_freq = 440.0
        freq_array = np.array([base_freq, base_freq, base_freq, base_freq])

        # Positive LFO should increase frequency
        lfo_values = np.array([1.0, 1.0, 1.0, 1.0])  # 1 semitone up
        result_up = lfo.apply_pitch_modulation(freq_array, lfo_values)

        assert np.all(result_up > base_freq)

        # Negative LFO should decrease frequency
        lfo_values = np.array([-1.0, -1.0, -1.0, -1.0])  # 1 semitone down
        result_down = lfo.apply_pitch_modulation(freq_array, lfo_values)

        assert np.all(result_down < base_freq)

    def test_pitch_modulation_semitone_accuracy(self):
        """Test pitch modulation semitone accuracy."""
        config.lfo_target = 'pitch'

        lfo = LFO()
        base_freq = 440.0  # A4
        freq_array = np.array([base_freq])

        # 12 semitones should double frequency (octave)
        lfo_values = np.array([12.0])
        result = lfo.apply_pitch_modulation(freq_array, lfo_values)

        expected_freq = base_freq * 2.0  # Octave up
        assert abs(result[0] - expected_freq) < 1.0

    def test_amplitude_modulation_disabled_target(self):
        """Test amplitude modulation when LFO target is not volume."""
        config.lfo_target = 'pitch'  # Not volume

        lfo = LFO()
        env = np.array([0.5, 0.5, 0.5, 0.5])
        lfo_values = np.array([0.1, -0.1, 0.2, -0.2])

        result = lfo.apply_amplitude_modulation(env, lfo_values)

        # Should return unchanged envelope
        assert np.allclose(result, env)

    def test_amplitude_modulation_zero_lfo(self):
        """Test amplitude modulation with zero LFO signal."""
        config.lfo_target = 'volume'

        lfo = LFO()
        env = np.array([0.5, 0.5, 0.5, 0.5])
        lfo_values = np.array([0.0, 0.0, 0.0, 0.0])

        result = lfo.apply_amplitude_modulation(env, lfo_values)

        # Should return unchanged envelope
        assert np.allclose(result, env)

    def test_amplitude_modulation_tremolo(self):
        """Test amplitude modulation creates tremolo effect."""
        config.lfo_target = 'volume'

        lfo = LFO()
        base_env = 0.5
        env = np.array([base_env, base_env, base_env, base_env])

        # Positive LFO should increase amplitude
        lfo_values = np.array([0.5, 0.5, 0.5, 0.5])
        result_up = lfo.apply_amplitude_modulation(env, lfo_values)

        assert np.all(result_up > base_env)

        # Negative LFO should decrease amplitude
        lfo_values = np.array([-0.5, -0.5, -0.5, -0.5])
        result_down = lfo.apply_amplitude_modulation(env, lfo_values)

        assert np.all(result_down < base_env)

    def test_amplitude_modulation_clipping(self):
        """Test amplitude modulation clipping to valid range."""
        config.lfo_target = 'volume'

        lfo = LFO()
        env = np.array([0.8, 0.8, 0.8, 0.8])

        # Large positive LFO that would exceed 1.0
        lfo_values = np.array([1.0, 1.0, 1.0, 1.0])
        result = lfo.apply_amplitude_modulation(env, lfo_values)

        # Should be clipped to valid range [0.0, 1.0]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Large negative LFO that would go below 0.0
        lfo_values = np.array([-2.0, -2.0, -2.0, -2.0])
        result = lfo.apply_amplitude_modulation(env, lfo_values)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_cutoff_modulation_disabled_target(self):
        """Test cutoff modulation when LFO target is not cutoff."""
        config.lfo_target = 'pitch'  # Not cutoff
        config.lfo_enabled = True
        config.lfo_depth = 0.5

        lfo = LFO()
        frames = 1024

        result = lfo.get_cutoff_modulation(frames)

        # Should return None when not targeting cutoff
        assert result is None

    def test_cutoff_modulation_disabled_lfo(self):
        """Test cutoff modulation when LFO is disabled."""
        config.lfo_target = 'cutoff'
        config.lfo_enabled = False
        config.lfo_depth = 0.5

        lfo = LFO()
        frames = 1024

        result = lfo.get_cutoff_modulation(frames)

        # Should return None when LFO is disabled
        assert result is None

    def test_cutoff_modulation_zero_depth(self):
        """Test cutoff modulation with zero depth."""
        config.lfo_target = 'cutoff'
        config.lfo_enabled = True
        config.lfo_depth = 0.0

        lfo = LFO()
        frames = 1024

        result = lfo.get_cutoff_modulation(frames)

        # Should return None with zero depth
        assert result is None

    def test_cutoff_modulation_valid(self, audio_helper):
        """Test valid cutoff modulation."""
        config.lfo_target = 'cutoff'
        config.lfo_enabled = True
        config.lfo_depth = 0.5
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = 1024

        result = lfo.get_cutoff_modulation(frames)

        assert result is not None
        assert len(result) == frames
        assert audio_helper.has_signal(result)

        # Should be bounded by depth
        assert np.max(np.abs(result)) <= config.lfo_depth * 1.1  # Small tolerance


class TestLFOEdgeCases:
    """Test edge cases and error conditions for LFO."""

    def test_lfo_very_high_rate(self):
        """Test LFO with very high rate."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 100.0  # Very high rate

        lfo = LFO()
        frames = 1024

        output = lfo.generate(frames)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_lfo_very_low_rate(self):
        """Test LFO with very low rate."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 0.1  # Very low rate

        lfo = LFO()
        frames = 1024

        output = lfo.generate(frames)

        assert len(output) == frames
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_lfo_zero_rate(self):
        """Test LFO with zero rate."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 0.0

        lfo = LFO()
        frames = 1024

        output = lfo.generate(frames)

        assert len(output) == frames
        # With zero rate, output should be constant
        assert np.allclose(output, output[0])

    def test_lfo_very_long_attack(self):
        """Test LFO with very long attack time."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 10.0  # Very long attack
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = 1024

        # Should still be in attack phase
        output = lfo.generate(frames)

        assert len(output) == frames
        # Output should be attenuated during attack
        max_output = np.max(np.abs(output))
        assert max_output < config.lfo_depth

    def test_lfo_very_long_delay(self):
        """Test LFO with very long delay time."""
        config.lfo_enabled = True
        config.lfo_depth = 1.0
        config.lfo_delay_time = 10.0  # Very long delay
        config.lfo_attack_time = 0.1
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = 1024

        # Should still be in delay phase
        output = lfo.generate(frames)

        assert len(output) == frames
        assert np.allclose(output, 0.0)

    def test_lfo_negative_depth(self):
        """Test LFO with negative depth."""
        config.lfo_enabled = True
        config.lfo_depth = -0.5  # Negative depth
        config.lfo_delay_time = 0.0
        config.lfo_attack_time = 0.0
        config.lfo_rate = 5.0

        lfo = LFO()
        frames = 1024

        output = lfo.generate(frames)

        assert len(output) == frames
        # Should still work, just inverted
        assert not np.allclose(output, 0.0)

    def test_lfo_time_accumulation(self):
        """Test that LFO time accumulates correctly."""
        lfo = LFO()

        initial_time = lfo.env_time
        frames = 1024

        lfo.generate(frames)

        expected_time = initial_time + frames / config.sample_rate
        assert abs(lfo.env_time - expected_time) < 1e-6

    def test_lfo_phase_accumulation(self):
        """Test that LFO phase accumulates correctly."""
        # Enable LFO so phase gets updated
        config.lfo_enabled = True
        config.lfo_depth = 0.5
        config.lfo_rate = 5.0

        lfo = LFO()

        initial_phase = lfo.phase
        frames = 1024

        lfo.generate(frames)

        expected_phase = initial_phase + frames / config.sample_rate
        assert abs(lfo.phase - expected_phase) < 1e-6
