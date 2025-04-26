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
        if not config.lfo_enabled:
            # Still increment time and phase even when disabled
            self.env_time += frames / config.sample_rate
            self.phase += frames / config.sample_rate
            return np.zeros(frames)

        # Generate time array for LFO
        t = np.arange(frames) / config.sample_rate + self.phase

        # Compute LFO envelope with delay and attack
        if self.env_time < config.lfo_delay_time:
            # In delay phase - no LFO
            lfo_env = 0.0
        elif config.lfo_attack_time > 0:
            # In attack phase - apply attack envelope after delay
            attack_time = self.env_time - config.lfo_delay_time
            lfo_env = np.clip(attack_time / config.lfo_attack_time, 0, 1.0)
        else:
            # No attack time - full envelope after delay
            lfo_env = 1.0

        lfo_env_array = np.full(frames, lfo_env)

        # Generate LFO signal with envelope
        lfo = lfo_env_array * config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)

        # Increment LFO envelope time
        self.env_time += frames / config.sample_rate

        # Update phase for next buffer
        self.phase += frames / config.sample_rate

        return lfo

    def apply_pitch_modulation(self, freq_array, lfo_values):
        """Apply LFO modulation to pitch (vibrato effect).

        Args:
            freq_array: Array of frequency values
            lfo_values: LFO signal values

        Returns:
            Modulated frequency array
        """
        if config.lfo_target != 'pitch':
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
        if config.lfo_target != 'volume':
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
        if config.lfo_target != 'cutoff' or not config.lfo_enabled:
            return None

        t = np.arange(frames) / config.sample_rate + self.phase
        return config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)
