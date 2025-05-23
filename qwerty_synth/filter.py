import numpy as np
from qwerty_synth import config


# State Variable Filter state variables
_v0z = 0.0  # First integrator state
_v1z = 0.0  # Second integrator state

# Biquad filter state variables
_bq_x1 = 0.0  # Input delay 1
_bq_x2 = 0.0  # Input delay 2
_bq_y1 = 0.0  # Output delay 1
_bq_y2 = 0.0  # Output delay 2


def apply_filter(samples, lfo_modulation=None, filter_envelope=None):
    """
    Apply a filter with support for multiple filter types and topologies.

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

    # Route to appropriate filter implementation based on topology
    if config.filter_topology == 'biquad':
        # Use biquad filter implementation
        if np.allclose(modulated_cutoff, modulated_cutoff[0], rtol=0.01):
            return apply_biquad_constant_cutoff(samples, modulated_cutoff[0])
        else:
            return apply_biquad_variable_cutoff(samples, modulated_cutoff)
    else:
        # Use State Variable Filter implementation (default)
        if np.allclose(modulated_cutoff, modulated_cutoff[0], rtol=0.01):
            return apply_svf_constant_cutoff(samples, modulated_cutoff[0])
        else:
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
    """Reset the filter state variables for both SVF and biquad."""
    global _v0z, _v1z, _bq_x1, _bq_x2, _bq_y1, _bq_y2
    # Reset SVF state
    _v0z = 0.0
    _v1z = 0.0
    # Reset biquad state
    _bq_x1 = 0.0
    _bq_x2 = 0.0
    _bq_y1 = 0.0
    _bq_y2 = 0.0


def calculate_biquad_coeffs(cutoff_freq, q_factor, filter_type, slope):
    """Calculate biquad filter coefficients.

    Args:
        cutoff_freq: Cutoff frequency in Hz
        q_factor: Q factor (resonance)
        filter_type: 'lowpass', 'highpass', 'bandpass', 'notch'
        slope: Filter slope (12 or 24 dB/octave)

    Returns:
        Tuple of (b0, b1, b2, a1, a2) coefficients
    """
    fs = config.sample_rate
    w = 2.0 * np.pi * cutoff_freq / fs
    cos_w = np.cos(w)
    sin_w = np.sin(w)

    # Calculate alpha for Q factor
    alpha = sin_w / (2.0 * q_factor)

    if filter_type == 'lowpass':
        if slope == 12:
            # Single-pole approximation using biquad
            b0 = (1.0 - cos_w) / 2.0
            b1 = 1.0 - cos_w
            b2 = (1.0 - cos_w) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w
            a2 = 1.0 - alpha
        else:  # 24 dB/octave
            b0 = (1.0 - cos_w) / 2.0
            b1 = 1.0 - cos_w
            b2 = (1.0 - cos_w) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w
            a2 = 1.0 - alpha

    elif filter_type == 'highpass':
        if slope == 12:
            # Single-pole approximation
            b0 = (1.0 + cos_w) / 2.0
            b1 = -(1.0 + cos_w)
            b2 = (1.0 + cos_w) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w
            a2 = 1.0 - alpha
        else:  # 24 dB/octave
            b0 = (1.0 + cos_w) / 2.0
            b1 = -(1.0 + cos_w)
            b2 = (1.0 + cos_w) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w
            a2 = 1.0 - alpha

    elif filter_type == 'bandpass':
        # Bandpass is inherently 12dB/octave per side (24dB/octave total)
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    elif filter_type == 'notch':
        # Notch filter
        b0 = 1.0
        b1 = -2.0 * cos_w
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    else:
        # Default to lowpass
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = (1.0 - cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    # Normalize by a0
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return b0, b1, b2, a1, a2


def apply_biquad_constant_cutoff(samples, cutoff_freq):
    """Biquad filter implementation for constant cutoff frequency."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    # Convert resonance to Q factor
    safe_resonance = min(config.filter_resonance, 0.99)
    q = 1.0 / (2.0 * safe_resonance + 0.001)

    # Calculate filter coefficients
    b0, b1, b2, a1, a2 = calculate_biquad_coeffs(
        cutoff_freq, q, config.filter_type, config.filter_slope
    )

    # Apply filter
    filtered = np.zeros_like(samples)

    for i, x in enumerate(samples):
        # Biquad difference equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        y = b0 * x + b1 * _bq_x1 + b2 * _bq_x2 - a1 * _bq_y1 - a2 * _bq_y2

        # Update delay line
        _bq_x2 = _bq_x1
        _bq_x1 = x
        _bq_y2 = _bq_y1
        _bq_y1 = y

        filtered[i] = y

    # Prevent denormal numbers
    if abs(_bq_x1) < 1e-20:
        _bq_x1 = 0.0
    if abs(_bq_x2) < 1e-20:
        _bq_x2 = 0.0
    if abs(_bq_y1) < 1e-20:
        _bq_y1 = 0.0
    if abs(_bq_y2) < 1e-20:
        _bq_y2 = 0.0

    return filtered


def apply_biquad_variable_cutoff(samples, modulated_cutoff):
    """Biquad filter implementation for variable cutoff frequency."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    # Convert resonance to Q factor
    safe_resonance = min(config.filter_resonance, 0.99)
    q = 1.0 / (2.0 * safe_resonance + 0.001)

    filtered = np.zeros_like(samples)

    # Apply filter sample by sample with variable coefficients
    for i, x in enumerate(samples):
        # Calculate coefficients for this sample
        b0, b1, b2, a1, a2 = calculate_biquad_coeffs(
            modulated_cutoff[i], q, config.filter_type, config.filter_slope
        )

        # Biquad difference equation
        y = b0 * x + b1 * _bq_x1 + b2 * _bq_x2 - a1 * _bq_y1 - a2 * _bq_y2

        # Update delay line
        _bq_x2 = _bq_x1
        _bq_x1 = x
        _bq_y2 = _bq_y1
        _bq_y1 = y

        filtered[i] = y

    # Prevent denormal numbers
    if abs(_bq_x1) < 1e-20:
        _bq_x1 = 0.0
    if abs(_bq_x2) < 1e-20:
        _bq_x2 = 0.0
    if abs(_bq_y1) < 1e-20:
        _bq_y1 = 0.0
    if abs(_bq_y2) < 1e-20:
        _bq_y2 = 0.0

    return filtered
