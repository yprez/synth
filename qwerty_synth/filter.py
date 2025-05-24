import numpy as np
from numba import jit
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

# Trigonometric lookup tables for fast approximation
_LUT_SIZE = 8192
_LUT_SCALE = _LUT_SIZE / (2.0 * np.pi)
_sin_lut = np.sin(np.linspace(0, 2*np.pi, _LUT_SIZE, endpoint=False)).astype(np.float32)
_cos_lut = np.cos(np.linspace(0, 2*np.pi, _LUT_SIZE, endpoint=False)).astype(np.float32)
_tan_lut = np.tan(np.linspace(0, np.pi/2 * 0.99, _LUT_SIZE)).astype(np.float32)  # 0 to almost π/2
_tan_scale = _LUT_SIZE / (np.pi/2 * 0.99)

# Temporary arrays for performance optimization (track for cleanup)
_temp_arrays = []

# Pre-allocated output buffer for filter processing (avoids repeated allocation)
_output_buffer = np.zeros(0, dtype=np.float32)

# JIT compilation status
_jit_warmed_up = False


@jit(nopython=True, fastmath=True, cache=True)
def _fast_sin_lut(x):
    """Fast sine approximation using lookup table with linear interpolation."""
    # Normalize to [0, 2π) range
    x_norm = (x % (2.0 * np.pi)) * _LUT_SCALE
    idx = int(x_norm)
    frac = x_norm - idx

    # Linear interpolation
    return _sin_lut[idx] * (1.0 - frac) + _sin_lut[(idx + 1) % _LUT_SIZE] * frac


@jit(nopython=True, fastmath=True, cache=True)
def _fast_cos_lut(x):
    """Fast cosine approximation using lookup table with linear interpolation."""
    # Normalize to [0, 2π) range
    x_norm = (x % (2.0 * np.pi)) * _LUT_SCALE
    idx = int(x_norm)
    frac = x_norm - idx

    # Linear interpolation
    return _cos_lut[idx] * (1.0 - frac) + _cos_lut[(idx + 1) % _LUT_SIZE] * frac


@jit(nopython=True, fastmath=True, cache=True)
def _fast_tan_lut(x):
    """Fast tangent approximation using lookup table for small angles."""
    # Clamp to valid range [0, π/2 * 0.99)
    x_clamped = min(abs(x), np.pi/2 * 0.99)
    x_norm = x_clamped * _tan_scale
    idx = int(x_norm)
    frac = x_norm - idx

    # Linear interpolation
    if idx >= _LUT_SIZE - 1:
        return _tan_lut[_LUT_SIZE - 1]
    return _tan_lut[idx] * (1.0 - frac) + _tan_lut[idx + 1] * frac


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
    # Ensure JIT functions are compiled and ready
    ensure_jit_ready()

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


@jit(nopython=True, fastmath=True, cache=True)
def _apply_svf_constant_jit(samples, output_buffer, cutoff_freq, r, output_type, sample_rate, v0z, v1z):
    """JIT-compiled SVF for constant cutoff frequency with pre-allocated output."""
    # Pre-calculate coefficients
    w_half = np.pi * cutoff_freq / sample_rate
    g = _fast_tan_lut(w_half)
    g1 = 1.0 / (1.0 + g * (g + r))
    g2 = g * g1
    g3 = g * g2

    # Process samples directly into the output buffer
    for i in range(len(samples)):
        x = samples[i]

        # SVF equations
        hp = g1 * (x - r * v1z - v0z)
        bp = g2 * hp + v1z
        lp = g3 * hp + v0z

        # Update states
        v0z = lp + g2 * hp
        v1z = bp + g3 * hp

        # Fast output selection using integer mapping
        if output_type == 0:    # lowpass
            output_buffer[i] = lp
        elif output_type == 1:  # highpass
            output_buffer[i] = hp
        elif output_type == 2:  # bandpass
            output_buffer[i] = bp
        else:                   # notch (output_type == 3)
            output_buffer[i] = hp + lp

    return v0z, v1z


def _apply_svf_constant(samples, cutoff_freq, r, output_type):
    """Optimized SVF for constant cutoff frequency."""
    global _v0z, _v1z

    # Get pre-allocated output buffer
    output_buffer = _ensure_output_buffer(len(samples))

    # Call JIT-compiled function
    _v0z, _v1z = _apply_svf_constant_jit(
        samples, output_buffer, cutoff_freq, r, output_type, config.sample_rate, _v0z, _v1z
    )

    # Apply denormal protection and DC leak once at the end
    _apply_denormal_protection_svf()

    return output_buffer.copy()  # Return a copy to avoid buffer reuse issues


@jit(nopython=True, fastmath=True, cache=True)
def _apply_svf_variable_jit(samples, output_buffer, modulated_cutoff, r, output_type, sample_rate, v0z, v1z):
    """JIT-compiled SVF for variable cutoff frequency with pre-allocated output."""
    fs_inv = 1.0 / sample_rate
    pi = np.pi

    for i in range(len(samples)):
        x = samples[i]
        f = modulated_cutoff[i]

        # Calculate coefficients for this sample
        w_half = pi * f * fs_inv
        g = _fast_tan_lut(w_half)
        g1 = 1.0 / (1.0 + g * (g + r))
        g2 = g * g1
        g3 = g * g2

        # SVF equations
        hp = g1 * (x - r * v1z - v0z)
        bp = g2 * hp + v1z
        lp = g3 * hp + v0z

        # Update states
        v0z = lp + g2 * hp
        v1z = bp + g3 * hp

        # Fast output selection using integer mapping
        if output_type == 0:    # lowpass
            output_buffer[i] = lp
        elif output_type == 1:  # highpass
            output_buffer[i] = hp
        elif output_type == 2:  # bandpass
            output_buffer[i] = bp
        else:                   # notch (output_type == 3)
            output_buffer[i] = hp + lp

    return v0z, v1z


def _apply_svf_variable(samples, modulated_cutoff, r, output_type):
    """Optimized SVF for variable cutoff frequency."""
    global _v0z, _v1z

    # Get pre-allocated output buffer
    output_buffer = _ensure_output_buffer(len(samples))

    # Call JIT-compiled function
    _v0z, _v1z = _apply_svf_variable_jit(
        samples, output_buffer, modulated_cutoff, r, output_type, config.sample_rate, _v0z, _v1z
    )

    _apply_denormal_protection_svf()
    return output_buffer.copy()  # Return a copy to avoid buffer reuse issues


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


@jit(nopython=True, fastmath=True, cache=True)
def _apply_biquad_constant_jit(samples, output_buffer, cutoff_freq, q, sample_rate, filter_type_int, bq_x1, bq_x2, bq_y1, bq_y2):
    """JIT-compiled biquad for constant cutoff frequency with pre-allocated output."""
    # Pre-calculate coefficients
    b0, b1, b2, a1, a2 = _calculate_biquad_coeffs_jit(cutoff_freq, q, sample_rate, filter_type_int)

    for i in range(len(samples)):
        x = samples[i]

        # Biquad difference equation
        y = b0 * x + b1 * bq_x1 + b2 * bq_x2 - a1 * bq_y1 - a2 * bq_y2

        # Update delay line
        bq_x2 = bq_x1
        bq_x1 = x
        bq_y2 = bq_y1
        bq_y1 = y

        output_buffer[i] = y

    return bq_x1, bq_x2, bq_y1, bq_y2


def _apply_biquad_constant(samples, cutoff_freq, q):
    """Optimized biquad for constant cutoff frequency."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    # Convert filter type to integer for JIT function
    filter_type_map = {
        'lowpass': 0,
        'highpass': 1,
        'bandpass': 2,
        'notch': 3
    }
    filter_type_int = filter_type_map.get(config.filter_type, 0)

    # Get pre-allocated output buffer
    output_buffer = _ensure_output_buffer(len(samples))

    # Call JIT-compiled function
    _bq_x1, _bq_x2, _bq_y1, _bq_y2 = _apply_biquad_constant_jit(
        samples, output_buffer, cutoff_freq, q, config.sample_rate, filter_type_int,
        _bq_x1, _bq_x2, _bq_y1, _bq_y2
    )

    _apply_denormal_protection_biquad()
    return output_buffer.copy()  # Return a copy to avoid buffer reuse issues


@jit(nopython=True, fastmath=True, cache=True)
def _apply_biquad_variable_jit(samples, output_buffer, modulated_cutoff, q, sample_rate, filter_type_int, bq_x1, bq_x2, bq_y1, bq_y2):
    """JIT-compiled biquad for variable cutoff frequency with pre-allocated output."""

    for i in range(len(samples)):
        x = samples[i]

        # Calculate coefficients for this sample
        b0, b1, b2, a1, a2 = _calculate_biquad_coeffs_jit(modulated_cutoff[i], q, sample_rate, filter_type_int)

        # Biquad difference equation
        y = b0 * x + b1 * bq_x1 + b2 * bq_x2 - a1 * bq_y1 - a2 * bq_y2

        # Update delay line
        bq_x2 = bq_x1
        bq_x1 = x
        bq_y2 = bq_y1
        bq_y1 = y

        output_buffer[i] = y

    return bq_x1, bq_x2, bq_y1, bq_y2


def _apply_biquad_variable(samples, modulated_cutoff, q):
    """Optimized biquad for variable cutoff frequency."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    # Convert filter type to integer for JIT function
    filter_type_map = {
        'lowpass': 0,
        'highpass': 1,
        'bandpass': 2,
        'notch': 3
    }
    filter_type_int = filter_type_map.get(config.filter_type, 0)

    # Get pre-allocated output buffer
    output_buffer = _ensure_output_buffer(len(samples))

    # Call JIT-compiled function
    _bq_x1, _bq_x2, _bq_y1, _bq_y2 = _apply_biquad_variable_jit(
        samples, output_buffer, modulated_cutoff, q, config.sample_rate, filter_type_int,
        _bq_x1, _bq_x2, _bq_y1, _bq_y2
    )

    _apply_denormal_protection_biquad()
    return output_buffer.copy()  # Return a copy to avoid buffer reuse issues


@jit(nopython=True, fastmath=True, cache=True)
def _calculate_biquad_coeffs_jit(cutoff_freq, q_factor, sample_rate, filter_type_int):
    """JIT-compiled fast biquad coefficient calculation."""
    fs = sample_rate
    w = 2.0 * np.pi * cutoff_freq / fs
    cos_w = _fast_cos_lut(w)
    sin_w = _fast_sin_lut(w)
    alpha = sin_w / (2.0 * q_factor)

    # Pre-compute common values
    one_plus_alpha = 1.0 + alpha
    one_minus_alpha = 1.0 - alpha
    neg_two_cos_w = -2.0 * cos_w

    # Use integer filter type: 0=lowpass, 1=highpass, 2=bandpass, 3=notch
    if filter_type_int == 0:  # lowpass
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = b0  # Same as b0
        a0 = one_plus_alpha
        a1 = neg_two_cos_w
        a2 = one_minus_alpha

    elif filter_type_int == 1:  # highpass
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = b0  # Same as b0
        a0 = one_plus_alpha
        a1 = neg_two_cos_w
        a2 = one_minus_alpha

    elif filter_type_int == 2:  # bandpass
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = one_plus_alpha
        a1 = neg_two_cos_w
        a2 = one_minus_alpha

    elif filter_type_int == 3:  # notch
        b0 = 1.0
        b1 = neg_two_cos_w
        b2 = 1.0
        a0 = one_plus_alpha
        a1 = neg_two_cos_w
        a2 = one_minus_alpha

    else:  # Default to lowpass
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = b0
        a0 = one_plus_alpha
        a1 = neg_two_cos_w
        a2 = one_minus_alpha

    # Normalize by a0 (optimized division)
    a0_inv = 1.0 / a0
    return (b0 * a0_inv, b1 * a0_inv, b2 * a0_inv,
            a1 * a0_inv, a2 * a0_inv)


def _calculate_biquad_coeffs_fast(cutoff_freq, q_factor):
    """Fast biquad coefficient calculation with optimized filter type selection."""
    # Convert filter type to integer for JIT function
    filter_type_map = {
        'lowpass': 0,
        'highpass': 1,
        'bandpass': 2,
        'notch': 3
    }
    filter_type_int = filter_type_map.get(config.filter_type, 0)  # Default to lowpass

    return _calculate_biquad_coeffs_jit(cutoff_freq, q_factor, config.sample_rate, filter_type_int)


def _apply_denormal_protection_svf():
    """Apply denormal protection to SVF state variables."""
    global _v0z, _v1z

    # DC leak and denormal protection
    _v0z *= _DC_LEAK_FACTOR
    _v1z *= _DC_LEAK_FACTOR

    if _v0z * _v0z < _DENORMAL_THRESHOLD:
        _v0z = 0.0
    if _v1z * _v1z < _DENORMAL_THRESHOLD:
        _v1z = 0.0


def _apply_denormal_protection_biquad():
    """Apply denormal protection to biquad state variables."""
    global _bq_x1, _bq_x2, _bq_y1, _bq_y2

    if _bq_x1 * _bq_x1 < _DENORMAL_THRESHOLD:
        _bq_x1 = 0.0
    if _bq_x2 * _bq_x2 < _DENORMAL_THRESHOLD:
        _bq_x2 = 0.0
    if _bq_y1 * _bq_y1 < _DENORMAL_THRESHOLD:
        _bq_y1 = 0.0
    if _bq_y2 * _bq_y2 < _DENORMAL_THRESHOLD:
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


def get_filter_performance_info():
    """Get information about filter performance optimizations and current state."""
    return {
        'vectorized_operations': True,
        'denormal_protection': True,
        'dc_leak_prevention': True,
        'constant_cutoff_optimization': True,
        'chunked_processing': True,
        'output_type_mapping': True,
        'coefficient_caching': True,
        'bypass_optimization': True,
        'temp_arrays_count': len(_temp_arrays),
        'svf_state_v0z': _v0z,
        'svf_state_v1z': _v1z,
        'biquad_state_x1': _bq_x1,
        'biquad_state_x2': _bq_x2,
        'biquad_state_y1': _bq_y1,
        'biquad_state_y2': _bq_y2
    }


def clear_temp_arrays():
    """Clear any temporary arrays used for optimization to free memory."""
    global _temp_arrays
    _temp_arrays.clear()


def _warmup_jit_functions():
    """Warm up JIT compilation by calling all JIT functions with dummy data."""
    global _jit_warmed_up

    if _jit_warmed_up:
        return

    try:
        # Create small dummy arrays for warm-up
        dummy_samples = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        dummy_modulated = np.array([1000.0, 1100.0, 1200.0, 1300.0], dtype=np.float32)
        dummy_output = np.zeros(4, dtype=np.float32)

        # Warm up trigonometric lookup functions
        _fast_sin_lut(1.0)
        _fast_cos_lut(1.0)
        _fast_tan_lut(0.5)

        # Warm up SVF functions
        _apply_svf_constant_jit(dummy_samples, dummy_output, 1000.0, 0.5, 0, 44100.0, 0.0, 0.0)
        _apply_svf_variable_jit(dummy_samples, dummy_output, dummy_modulated, 0.5, 0, 44100.0, 0.0, 0.0)

        # Warm up biquad functions for all filter types
        for filter_type in range(4):  # 0=lowpass, 1=highpass, 2=bandpass, 3=notch
            _apply_biquad_constant_jit(dummy_samples, dummy_output, 1000.0, 2.0, 44100.0, filter_type, 0.0, 0.0, 0.0, 0.0)
            _apply_biquad_variable_jit(dummy_samples, dummy_output, dummy_modulated, 2.0, 44100.0, filter_type, 0.0, 0.0, 0.0, 0.0)
            _calculate_biquad_coeffs_jit(1000.0, 2.0, 44100.0, filter_type)

        _jit_warmed_up = True

    except Exception as e:
        # If warm-up fails, continue without it - JIT will compile on first use
        print(f"JIT warm-up failed (this is usually fine): {e}")


def ensure_jit_ready():
    """Ensure JIT functions are compiled and ready for use."""
    if not _jit_warmed_up:
        _warmup_jit_functions()


# Initialize JIT compilation when module is imported
# This runs in a separate thread to avoid blocking import
def _async_warmup():
    """Perform JIT warm-up in background to avoid blocking module import."""
    import threading
    import time

    def warmup_thread():
        # Small delay to let the main application start
        time.sleep(0.1)
        _warmup_jit_functions()

    if not _jit_warmed_up:
        thread = threading.Thread(target=warmup_thread, daemon=True)
        thread.start()


# Start background warm-up when module loads
_async_warmup()


def _ensure_output_buffer(size):
    """Ensure the output buffer is large enough for the given size."""
    global _output_buffer
    if len(_output_buffer) < size:
        _output_buffer = np.zeros(size, dtype=np.float32)
    return _output_buffer[:size]
