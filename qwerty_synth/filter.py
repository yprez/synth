import numpy as np
from qwerty_synth.config import sample_rate


cutoff = 10000  # Default cutoff frequency in Hz
_last_output = 0.0  # Internal state for continuity


def apply_filter(samples, lfo_modulation=None):
    """
    Apply a simple one-pole low-pass filter to the input signal.

    Parameters:
        samples (np.ndarray): Input audio signal array (1D).
        lfo_modulation (np.ndarray, optional): LFO modulation signal for cutoff.

    Returns:
        np.ndarray: Filtered output signal.
    """
    global _last_output

    # Create array of cutoff values (either constant or modulated)
    if lfo_modulation is not None and len(samples) > 0:
        # Scale LFO to modulate between 50% and 150% of the base cutoff
        # This ensures cutoff never goes to zero which would cause artifacts
        modulated_cutoff = cutoff * (1.0 + lfo_modulation)
        # Ensure the cutoff stays within reasonable bounds
        modulated_cutoff = np.clip(modulated_cutoff, 20, sample_rate / 2.1)
    else:
        # Use constant cutoff if no modulation
        if cutoff >= sample_rate / 2:
            return samples  # No filtering if cutoff is too high
        modulated_cutoff = np.full(len(samples), cutoff)

    filtered = np.zeros_like(samples)

    # Apply filter sample by sample with potentially varying cutoff
    for i, x in enumerate(samples):
        rc = 1.0 / (2 * np.pi * modulated_cutoff[i])
        dt = 1.0 / sample_rate
        alpha = dt / (rc + dt)

        _last_output = alpha * x + (1 - alpha) * _last_output
        filtered[i] = _last_output

    return filtered
