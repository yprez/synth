import numpy as np
from qwerty_synth import config


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
    if not config.filter_enabled:
        return samples

    # Skip processing empty arrays
    if len(samples) == 0:
        return samples

    # Start with base cutoff value
    modulated_cutoff = np.full(len(samples), config.filter_cutoff)

    # Apply LFO modulation if provided
    if lfo_modulation is not None:
        # Scale LFO to modulate between 50% and 150% of the base cutoff
        modulated_cutoff = modulated_cutoff * (1.0 + lfo_modulation)

    # Apply filter envelope modulation if provided
    if filter_envelope is not None:
        # Add the envelope contribution (envelope * amount)
        modulated_cutoff = modulated_cutoff + (filter_envelope * config.filter_env_amount)

    # Ensure the cutoff stays within reasonable bounds (20Hz to just below Nyquist)
    modulated_cutoff = np.clip(modulated_cutoff, 20, config.sample_rate / 2.1)

    # Skip filtering if all cutoff values are too high
    if np.min(modulated_cutoff) >= config.sample_rate / 2:
        return samples

    # Skip filtering if resonance is zero and cutoff is very high (optimization)
    if config.filter_resonance < 0.01 and np.min(modulated_cutoff) > config.sample_rate / 3:
        return samples

    # Use a faster filter implementation when cutoff is constant across all samples
    if np.allclose(modulated_cutoff, modulated_cutoff[0], rtol=0.01):
        return apply_filter_constant_cutoff(samples, modulated_cutoff[0])

    # Apply the standard filter with varying cutoff
    return apply_filter_variable_cutoff(samples, modulated_cutoff)


def apply_filter_constant_cutoff(samples, cutoff_freq):
    """Optimized filter implementation for constant cutoff frequency."""
    global _last_output_1, _last_output_2, _last_input

    # Calculate filter coefficient for constant cutoff
    rc = 1.0 / (2 * np.pi * cutoff_freq)
    dt = 1.0 / config.sample_rate
    alpha = dt / (rc + dt)
    alpha = max(0.001, min(alpha, 0.999))  # Clamp alpha for stability

    # Limit resonance to safe values to prevent instability
    safe_resonance = min(config.filter_resonance, 0.99)
    feedback = safe_resonance**2 * 0.98

    # Apply filter in one pass
    filtered = np.zeros_like(samples)
    for i, x in enumerate(samples):
        fb = _last_output_2 * feedback
        input_with_feedback = x - fb
        output_1 = alpha * input_with_feedback + (1 - alpha) * _last_output_1
        _last_output_1 = output_1

        output_2 = alpha * output_1 + (1 - alpha) * _last_output_2
        _last_output_2 = output_2

        filtered[i] = output_2

    # Prevent DC drift
    _last_output_1 *= 0.999
    _last_output_2 *= 0.999

    return filtered


def apply_filter_variable_cutoff(samples, modulated_cutoff):
    """Standard filter implementation for variable cutoff frequency."""
    global _last_output_1, _last_output_2, _last_input

    # Calculate all filter coefficients at once
    rc = 1.0 / (2 * np.pi * modulated_cutoff)
    dt = 1.0 / config.sample_rate
    alpha = dt / (rc + dt)
    alpha = np.clip(alpha, 0.001, 0.999)  # Clamp alpha for stability

    # Limit resonance to safe values to prevent instability
    safe_resonance = min(config.filter_resonance, 0.99)
    feedback = safe_resonance**2 * 0.98

    filtered = np.zeros_like(samples)

    # Apply filter sample by sample, but with pre-calculated coefficients
    for i, x in enumerate(samples):
        fb = _last_output_2 * feedback
        input_with_feedback = x - fb
        output_1 = alpha[i] * input_with_feedback + (1 - alpha[i]) * _last_output_1
        _last_output_1 = output_1

        output_2 = alpha[i] * output_1 + (1 - alpha[i]) * _last_output_2
        _last_output_2 = output_2

        filtered[i] = output_2

    # Prevent DC drift
    _last_output_1 *= 0.999
    _last_output_2 *= 0.999

    return filtered


def reset_filter_state():
    """Reset the filter state variables."""
    global _last_output_1, _last_output_2, _last_input
    _last_output_1 = 0.0
    _last_output_2 = 0.0
    _last_input = 0.0
