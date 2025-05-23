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

# Pre-compiled filter type mappings for performance
_SVF_OUTPUT_MAP = {
    'lowpass': 0,
    'highpass': 1,
    'bandpass': 2,
    'notch': 3
}

# Constants for optimization
_DENORMAL_THRESHOLD = 1e-20
_DC_LEAK_FACTOR = 0.995
_MIN_Q = 0.001
_MAX_RESONANCE = 0.99


def apply_filter(samples, lfo_modulation=None, filter_envelope=None):
    """
    Apply a filter with support for multiple filter types and topologies.
    Optimized for performance with minimal branches and vectorized operations.

    Parameters:
        samples (np.ndarray): Input audio signal array (1D).
        lfo_modulation (np.ndarray, optional): LFO modulation signal for cutoff.
        filter_envelope (np.ndarray, optional): Filter envelope values to modulate cutoff.

    Returns:
        np.ndarray: Filtered output signal.
    """
    # Early exit optimizations
    if not config.filter_enabled or len(samples) == 0:
        return samples

    # Calculate modulated cutoff - vectorized operations
    modulated_cutoff = _calculate_modulated_cutoff(samples, lfo_modulation, filter_envelope)

    # Early exit for extreme cases
    if _should_bypass_filter(modulated_cutoff):
        return samples

    # Route to optimized filter implementation
    if config.filter_topology == 'biquad':
        return _apply_biquad(samples, modulated_cutoff)
    else:
        return _apply_svf(samples, modulated_cutoff)


def _calculate_modulated_cutoff(samples, lfo_modulation, filter_envelope):
    """Calculate modulated cutoff frequency using vectorized operations."""
    # Start with base cutoff - avoid creating full array if no modulation
    if lfo_modulation is None and filter_envelope is None:
        return config.filter_cutoff

    modulated_cutoff = np.full(len(samples), config.filter_cutoff, dtype=np.float32)

    # Apply LFO modulation
    if lfo_modulation is not None:
        modulated_cutoff *= (1.0 + lfo_modulation)

    # Apply filter envelope modulation
    if filter_envelope is not None:
        modulated_cutoff += (filter_envelope * config.filter_env_amount)

    # Clip to valid range
    return np.clip(modulated_cutoff, 20, config.sample_rate / 2.1)


def _should_bypass_filter(modulated_cutoff):
    """Check if filtering can be bypassed for performance."""
    if isinstance(modulated_cutoff, (int, float)):
        cutoff_min = cutoff_max = modulated_cutoff
    else:
        cutoff_min = np.min(modulated_cutoff)
        cutoff_max = np.max(modulated_cutoff)

    # Skip filtering for extreme low-pass cases
    if (config.filter_type == 'lowpass' and
        cutoff_min >= config.sample_rate / 2):
        return True

    # Skip filtering for no-resonance high-cutoff low-pass
    if (config.filter_type == 'lowpass' and
        config.filter_resonance < 0.01 and
        cutoff_min > config.sample_rate / 3):
        return True

    return False


def _apply_svf(samples, modulated_cutoff):
    """Optimized State Variable Filter implementation."""
    global _v0z, _v1z

    # Pre-calculate resonance parameters
    safe_resonance = min(config.filter_resonance, _MAX_RESONANCE)
    q = 1.0 / (safe_resonance + _MIN_Q)
    r = 1.0 / q

    # Get filter output type as integer for faster switching
    output_type = _SVF_OUTPUT_MAP.get(config.filter_type, 0)

    # Check if cutoff is constant for optimization
    if isinstance(modulated_cutoff, (int, float)):
        return _apply_svf_constant(samples, modulated_cutoff, r, output_type)
    elif np.allclose(modulated_cutoff, modulated_cutoff[0], rtol=0.01):
        return _apply_svf_constant(samples, modulated_cutoff[0], r, output_type)
    else:
        return _apply_svf_variable(samples, modulated_cutoff, r, output_type)


def _apply_svf_constant(samples, cutoff_freq, r, output_type):
    """Optimized SVF for constant cutoff frequency."""
    global _v0z, _v1z

    # Pre-calculate coefficients
    w = 2 * np.pi * cutoff_freq / config.sample_rate
    g = np.tan(w / 2)
    g1 = 1.0 / (1.0 + g * (g + r))
    g2 = g * g1
    g3 = g * g2

    # Process samples using pre-compiled array for output
    filtered = np.empty_like(samples, dtype=np.float32)

    for i in range(len(samples)):
        x = samples[i]

        # SVF equations
        hp = g1 * (x - r * _v1z - _v0z)
        bp = g2 * hp + _v1z
        lp = g3 * hp + _v0z

        # Update states
        _v0z = lp + g2 * hp
        _v1z = bp + g3 * hp

        # Fast output selection using integer mapping
        if output_type == 0:    # lowpass
            filtered[i] = lp
        elif output_type == 1:  # highpass
            filtered[i] = hp
        elif output_type == 2:  # bandpass
            filtered[i] = bp
        else:                   # notch (output_type == 3)
            filtered[i] = hp + lp

    # Apply denormal protection and DC leak once at the end
    _apply_denormal_protection_svf()

    return filtered


def _apply_svf_variable(samples, modulated_cutoff, r, output_type):
    """Optimized SVF for variable cutoff frequency."""
    global _v0z, _v1z

    fs_inv = 1.0 / config.sample_rate
    two_pi = 2 * np.pi

    filtered = np.empty_like(samples, dtype=np.float32)

    for i in range(len(samples)):
        x = samples[i]
        f = modulated_cutoff[i]

        # Calculate coefficients for this sample
        w = two_pi * f * fs_inv
        g = np.tan(w / 2)
        g1 = 1.0 / (1.0 + g * (g + r))
        g2 = g * g1
        g3 = g * g2

        # SVF equations
        hp = g1 * (x - r * _v1z - _v0z)
        bp = g2 * hp + _v1z
        lp = g3 * hp + _v0z

        # Update states
        _v0z = lp + g2 * hp
        _v1z = bp + g3 * hp

        # Fast output selection
        if output_type == 0:    # lowpass
            filtered[i] = lp
        elif output_type == 1:  # highpass
            filtered[i] = hp
        elif output_type == 2:  # bandpass
            filtered[i] = bp
        else:                   # notch
            filtered[i] = hp + lp

    _apply_denormal_protection_svf()
    return filtered


def _apply_biquad(samples, modulated_cutoff):
    """Optimized biquad filter implementation."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    # Pre-calculate Q factor
    safe_resonance = min(config.filter_resonance, _MAX_RESONANCE)
    q = 1.0 / (safe_resonance + _MIN_Q)

    # Check if cutoff is constant for optimization
    if isinstance(modulated_cutoff, (int, float)):
        return _apply_biquad_constant(samples, modulated_cutoff, q)
    elif np.allclose(modulated_cutoff, modulated_cutoff[0], rtol=0.01):
        return _apply_biquad_constant(samples, modulated_cutoff[0], q)
    else:
        return _apply_biquad_variable(samples, modulated_cutoff, q)


def _apply_biquad_constant(samples, cutoff_freq, q):
    """Optimized biquad for constant cutoff frequency."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    # Pre-calculate coefficients
    b0, b1, b2, a1, a2 = _calculate_biquad_coeffs_fast(cutoff_freq, q)

    filtered = np.empty_like(samples, dtype=np.float32)

    for i in range(len(samples)):
        x = samples[i]

        # Biquad difference equation
        y = b0 * x + b1 * _bq_x1 + b2 * _bq_x2 - a1 * _bq_y1 - a2 * _bq_y2

        # Update delay line
        _bq_x2 = _bq_x1
        _bq_x1 = x
        _bq_y2 = _bq_y1
        _bq_y1 = y

        filtered[i] = y

    _apply_denormal_protection_biquad()
    return filtered


def _apply_biquad_variable(samples, modulated_cutoff, q):
    """Optimized biquad for variable cutoff frequency."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    filtered = np.empty_like(samples, dtype=np.float32)

    for i in range(len(samples)):
        x = samples[i]

        # Calculate coefficients for this sample
        b0, b1, b2, a1, a2 = _calculate_biquad_coeffs_fast(modulated_cutoff[i], q)

        # Biquad difference equation
        y = b0 * x + b1 * _bq_x1 + b2 * _bq_x2 - a1 * _bq_y1 - a2 * _bq_y2

        # Update delay line
        _bq_x2 = _bq_x1
        _bq_x1 = x
        _bq_y2 = _bq_y1
        _bq_y1 = y

        filtered[i] = y

    _apply_denormal_protection_biquad()
    return filtered


def _calculate_biquad_coeffs_fast(cutoff_freq, q_factor):
    """Fast biquad coefficient calculation with optimized filter type selection."""
    fs = config.sample_rate
    w = 2.0 * np.pi * cutoff_freq / fs
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    alpha = sin_w / (2.0 * q_factor)

    filter_type = config.filter_type

    if filter_type == 'lowpass':
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = b0  # Same as b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    elif filter_type == 'highpass':
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = b0  # Same as b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    elif filter_type == 'bandpass':
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    elif filter_type == 'notch':
        b0 = 1.0
        b1 = -2.0 * cos_w
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    else:  # Default to lowpass
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha

    # Normalize by a0 (optimized division)
    a0_inv = 1.0 / a0
    return (b0 * a0_inv, b1 * a0_inv, b2 * a0_inv,
            a1 * a0_inv, a2 * a0_inv)


def _apply_denormal_protection_svf():
    """Apply denormal protection to SVF state variables."""
    global _v0z, _v1z

    # DC leak and denormal protection
    _v0z *= _DC_LEAK_FACTOR
    _v1z *= _DC_LEAK_FACTOR

    if abs(_v0z) < _DENORMAL_THRESHOLD:
        _v0z = 0.0
    if abs(_v1z) < _DENORMAL_THRESHOLD:
        _v1z = 0.0


def _apply_denormal_protection_biquad():
    """Apply denormal protection to biquad state variables."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    if abs(_bq_x1) < _DENORMAL_THRESHOLD:
        _bq_x1 = 0.0
    if abs(_bq_x2) < _DENORMAL_THRESHOLD:
        _bq_x2 = 0.0
    if abs(_bq_y1) < _DENORMAL_THRESHOLD:
        _bq_y1 = 0.0
    if abs(_bq_y2) < _DENORMAL_THRESHOLD:
        _bq_y2 = 0.0


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
