import numpy as np
from qwerty_synth.config import sample_rate


cutoff = 10000  # Default cutoff frequency in Hz
_last_output = 0.0  # Internal state for continuity


def apply_filter(samples):
    """
    Apply a simple one-pole low-pass filter to the input signal.

    Parameters:
        samples (np.ndarray): Input audio signal array (1D).

    Returns:
        np.ndarray: Filtered output signal.
    """
    global _last_output
    if cutoff >= sample_rate / 2:
        return samples  # No filtering if cutoff is too high

    alpha = 2 * np.pi * cutoff / sample_rate
    alpha = min(alpha, 1.0)

    filtered = np.zeros_like(samples)
    for i, x in enumerate(samples):
        _last_output = alpha * x + (1 - alpha) * _last_output
        filtered[i] = _last_output

    return filtered
