import numpy as np
from qwerty_synth.config import sample_rate
from qwerty_synth import adsr


cutoff = 10000  # Default cutoff frequency in Hz
_last_output = 0.0  # Internal state for continuity


def apply_filter(samples, lfo_modulation=None, filter_envelope=None):
    """
    Apply a simple one-pole low-pass filter to the input signal.

    Parameters:
        samples (np.ndarray): Input audio signal array (1D).
        lfo_modulation (np.ndarray, optional): LFO modulation signal for cutoff.
        filter_envelope (np.ndarray, optional): Filter envelope values to modulate cutoff.

    Returns:
        np.ndarray: Filtered output signal.
    """
    global _last_output

    # Create array of cutoff values (base + modulations)
    if len(samples) == 0:
        return samples

    # Start with base cutoff value
    modulated_cutoff = np.full(len(samples), cutoff)

    # Apply LFO modulation if provided
    if lfo_modulation is not None:
        # Scale LFO to modulate between 50% and 150% of the base cutoff
        modulated_cutoff = modulated_cutoff * (1.0 + lfo_modulation)

    # Apply filter envelope modulation if provided
    if filter_envelope is not None:
        # Add the envelope contribution (envelope * amount)
        modulated_cutoff = modulated_cutoff + (filter_envelope * adsr.filter_env_amount)

    # Ensure the cutoff stays within reasonable bounds (20Hz to just below Nyquist)
    modulated_cutoff = np.clip(modulated_cutoff, 20, sample_rate / 2.1)

    # Skip filtering if all cutoff values are too high
    if np.min(modulated_cutoff) >= sample_rate / 2:
        return samples

    # Apply filter with the modulated cutoff
    filtered = np.zeros_like(samples)

    # Apply filter sample by sample with potentially varying cutoff
    for i, x in enumerate(samples):
        rc = 1.0 / (2 * np.pi * modulated_cutoff[i])
        dt = 1.0 / sample_rate
        alpha = dt / (rc + dt)

        _last_output = alpha * x + (1 - alpha) * _last_output
        filtered[i] = _last_output

    return filtered
