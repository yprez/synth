import numpy as np
from qwerty_synth.config import sample_rate
from qwerty_synth import adsr


cutoff = 10000  # Default cutoff frequency in Hz
resonance = 0.0  # Default resonance (0.0-1.0), higher values create more pronounced peaks
filter_enabled = True  # Flag to enable/disable the filter
_last_output_1 = 0.0  # Internal state for continuity (first stage)
_last_output_2 = 0.0  # Internal state for continuity (second stage)
_last_input = 0.0  # Previous input sample


def apply_filter(samples, lfo_modulation=None, filter_envelope=None):
    """
    Apply a two-pole low-pass filter with resonance to the input signal.

    Parameters:
        samples (np.ndarray): Input audio signal array (1D).
        lfo_modulation (np.ndarray, optional): LFO modulation signal for cutoff.
        filter_envelope (np.ndarray, optional): Filter envelope values to modulate cutoff.

    Returns:
        np.ndarray: Filtered output signal.
    """
    global _last_output_1, _last_output_2, _last_input

    # If filter is disabled, return the original samples
    if not filter_enabled:
        return samples

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

    # Limit resonance to safe values to prevent instability
    # Higher resonance values create more pronounced peaks at the cutoff frequency
    safe_resonance = min(resonance, 0.99)

    # Calculate feedback factor based on resonance
    # As resonance approaches 1.0, feedback increases dramatically
    feedback = safe_resonance * 0.98

    # Apply two-pole filter with resonance sample by sample
    for i, x in enumerate(samples):
        # Calculate filter coefficient for current cutoff
        rc = 1.0 / (2 * np.pi * modulated_cutoff[i])
        dt = 1.0 / sample_rate
        alpha = dt / (rc + dt)

        # Apply first filter stage with feedback for resonance
        input_with_feedback = x - (_last_output_2 * feedback)
        output_1 = alpha * input_with_feedback + (1 - alpha) * _last_output_1
        _last_output_1 = output_1

        # Apply second filter stage
        output_2 = alpha * output_1 + (1 - alpha) * _last_output_2
        _last_output_2 = output_2

        filtered[i] = output_2
        _last_input = x

    return filtered


def reset_filter_state():
    """Reset the filter state variables."""
    global _last_output_1, _last_output_2, _last_input
    _last_output_1 = 0.0
    _last_output_2 = 0.0
    _last_input = 0.0
