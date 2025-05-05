"""Low Frequency Oscillator (LFO) functionality for QWERTY Synth."""

import numpy as np
from qwerty_synth import config


class LFO:
    """Low Frequency Oscillator for modulation effects."""

    def __init__(self):
        """Initialize the LFO with default settings."""
        self.phase = 0.0
        self.env_time = 0.0  # Time tracker for LFO envelope

    def reset(self):
        """Reset the LFO envelope time for a new note."""
        self.env_time = 0.0

    def generate(self, frames):
        """Generate LFO signal with envelope.

        Args:
            frames: Number of audio frames to generate

        Returns:
            numpy array of LFO values
        """
        # Always update timing even when disabled
        self.env_time += frames / config.sample_rate
        self.phase += frames / config.sample_rate

        # Skip generation entirely if disabled or depth is zero
        if not config.lfo_enabled or config.lfo_depth <= 0.001:
            return np.zeros(frames)

        # Generate time array for LFO
        t = np.arange(frames) / config.sample_rate + self.phase

        # Quick envelope processing based on current state
        if self.env_time < config.lfo_delay_time:
            # Still in delay phase - return zeros quickly
            return np.zeros(frames)

        # Calculate envelope level
        if self.env_time < config.lfo_delay_time + config.lfo_attack_time:
            # In attack phase - apply attack envelope after delay
            attack_time = self.env_time - config.lfo_delay_time
            lfo_env = attack_time / config.lfo_attack_time
        else:
            # After attack - full envelope
            lfo_env = 1.0

        # Generate LFO signal with envelope - vectorized
        lfo = lfo_env * config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)

        return lfo

    def apply_pitch_modulation(self, freq_array, lfo_values):
        """Apply LFO modulation to pitch (vibrato effect).

        Args:
            freq_array: Array of frequency values
            lfo_values: LFO signal values

        Returns:
            Modulated frequency array
        """
        # Quick return if not targeting pitch or if LFO is all zeros
        if config.lfo_target != 'pitch' or np.all(lfo_values == 0):
            return freq_array

        # The exponential formula converts semitones to frequency ratio
        # We divide by 12 to convert LFO range to semitones
        return freq_array * (2 ** (lfo_values / 12))

    def apply_amplitude_modulation(self, env, lfo_values):
        """Apply LFO modulation to amplitude (tremolo effect).

        Args:
            env: Amplitude envelope values
            lfo_values: LFO signal values

        Returns:
            Modulated amplitude envelope
        """
        # Quick return if not targeting volume or if LFO is all zeros
        if config.lfo_target != 'volume' or np.all(lfo_values == 0):
            return env

        # Modulate amplitude using LFO
        modulated_env = env * (1.0 + lfo_values)
        # Clip to avoid extreme values
        return np.clip(modulated_env, 0.0, 1.0)

    def get_cutoff_modulation(self, frames):
        """Generate LFO modulation for filter cutoff.

        Args:
            frames: Number of frames to generate

        Returns:
            LFO values for cutoff modulation or None if not targeting cutoff
        """
        # Skip processing completely when not needed
        if config.lfo_target != 'cutoff' or not config.lfo_enabled or config.lfo_depth <= 0.001:
            return None

        t = np.arange(frames) / config.sample_rate + self.phase
        return config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)
