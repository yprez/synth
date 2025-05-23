import numpy as np
from qwerty_synth import config


# State Variable Filter state variables
_v0z = 0.0  # First integrator state
_v1z = 0.0  # Second integrator state


def apply_filter(samples, lfo_modulation=None, filter_envelope=None):
    """
    Apply a State Variable Filter with support for multiple filter types.

    Parameters:
        samples (np.ndarray): Input audio signal array (1D).
        lfo_modulation (np.ndarray, optional): LFO modulation signal for cutoff.
        filter_envelope (np.ndarray, optional): Filter envelope values to modulate cutoff.

    Returns:
        np.ndarray: Filtered output signal.
    """
    global _v0z, _v1z

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

    # Skip filtering if all cutoff values are too high for low-pass
    if config.filter_type == 'lowpass' and np.min(modulated_cutoff) >= config.sample_rate / 2:
        return samples

    # Skip filtering if resonance is zero and cutoff is very high (optimization for low-pass)
    if (config.filter_type == 'lowpass' and config.filter_resonance < 0.01 and
        np.min(modulated_cutoff) > config.sample_rate / 3):
        return samples

    # Use a faster filter implementation when cutoff is constant across all samples
    if np.allclose(modulated_cutoff, modulated_cutoff[0], rtol=0.01):
        return apply_svf_constant_cutoff(samples, modulated_cutoff[0])

    # Apply the standard filter with varying cutoff
    return apply_svf_variable_cutoff(samples, modulated_cutoff)


def apply_svf_constant_cutoff(samples, cutoff_freq):
    """State Variable Filter implementation for constant cutoff frequency."""
    global _v0z, _v1z

    # Calculate filter coefficients
    fs = config.sample_rate
    f = cutoff_freq

    # Frequency warping
    w = 2 * np.pi * f / fs
    g = np.tan(w / 2)

    # Resonance control (Q factor)
    safe_resonance = min(config.filter_resonance, 0.99)
    q = 1.0 / (2.0 * safe_resonance + 0.001)  # Convert resonance to Q factor
    r = 1.0 / (2.0 * q)  # Damping coefficient

    # State-space coefficients
    g1 = 1.0 / (1.0 + g * (g + r))
    g2 = g * g1
    g3 = g * g2

    # Apply filter
    filtered = np.zeros_like(samples)

    for i, x in enumerate(samples):
        # State Variable Filter equations
        hp = g1 * (x - r * _v1z - _v0z)
        bp = g2 * hp + _v1z
        lp = g3 * hp + _v0z

        # Update state variables
        _v0z = lp + g2 * hp
        _v1z = bp + g3 * hp

        # Select output based on filter type
        if config.filter_type == 'lowpass':
            filtered[i] = lp
        elif config.filter_type == 'highpass':
            filtered[i] = hp
        elif config.filter_type == 'bandpass':
            filtered[i] = bp
        elif config.filter_type == 'notch':
            filtered[i] = hp + lp  # Notch = HP + LP
        else:
            filtered[i] = lp  # Default to low-pass

    # Prevent DC drift and denormal numbers
    _v0z *= 0.995
    _v1z *= 0.995

    # Denormal protection
    if abs(_v0z) < 1e-20:
        _v0z = 0.0
    if abs(_v1z) < 1e-20:
        _v1z = 0.0

    return filtered


def apply_svf_variable_cutoff(samples, modulated_cutoff):
    """State Variable Filter implementation for variable cutoff frequency."""
    global _v0z, _v1z

    fs = config.sample_rate

    # Resonance control (Q factor)
    safe_resonance = min(config.filter_resonance, 0.99)
    q = 1.0 / (2.0 * safe_resonance + 0.001)  # Convert resonance to Q factor
    r = 1.0 / (2.0 * q)  # Damping coefficient

    filtered = np.zeros_like(samples)

    # Apply filter sample by sample with variable coefficients
    for i, x in enumerate(samples):
        # Calculate coefficients for this sample
        f = modulated_cutoff[i]
        w = 2 * np.pi * f / fs
        g = np.tan(w / 2)

        # State-space coefficients
        g1 = 1.0 / (1.0 + g * (g + r))
        g2 = g * g1
        g3 = g * g2

        # State Variable Filter equations
        hp = g1 * (x - r * _v1z - _v0z)
        bp = g2 * hp + _v1z
        lp = g3 * hp + _v0z

        # Update state variables
        _v0z = lp + g2 * hp
        _v1z = bp + g3 * hp

        # Select output based on filter type
        if config.filter_type == 'lowpass':
            filtered[i] = lp
        elif config.filter_type == 'highpass':
            filtered[i] = hp
        elif config.filter_type == 'bandpass':
            filtered[i] = bp
        elif config.filter_type == 'notch':
            filtered[i] = hp + lp  # Notch = HP + LP
        else:
            filtered[i] = lp  # Default to low-pass

    # Prevent DC drift and denormal numbers
    _v0z *= 0.995
    _v1z *= 0.995

    # Denormal protection
    if abs(_v0z) < 1e-20:
        _v0z = 0.0
    if abs(_v1z) < 1e-20:
        _v1z = 0.0

    return filtered


# Keep the old functions for backwards compatibility during transition
def apply_filter_constant_cutoff(samples, cutoff_freq):
    """Legacy function - redirects to new SVF implementation."""
    return apply_svf_constant_cutoff(samples, cutoff_freq)


def apply_filter_variable_cutoff(samples, modulated_cutoff):
    """Legacy function - redirects to new SVF implementation."""
    return apply_svf_variable_cutoff(samples, modulated_cutoff)


def reset_filter_state():
    """Reset the filter state variables."""
    global _v0z, _v1z
    _v0z = 0.0
    _v1z = 0.0
