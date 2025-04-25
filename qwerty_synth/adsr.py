"""ADSR (Attack, Decay, Sustain, Release) envelope functionality."""

import numpy as np

# ADSR envelope parameters for amplitude
adsr = {
    'attack': 0.02,
    'decay': 0.03,
    'sustain': 0.2,
    'release': 0.2,
}

# ADSR envelope parameters for filter cutoff
filter_adsr = {
    'attack': 0.05,
    'decay': 0.1,
    'sustain': 0.4,
    'release': 0.3,
}

# Filter envelope amount (how much the envelope affects the cutoff)
filter_env_amount = 5000  # Hz

# Pre-calculated ADSR curves for visualization
adsr_curve = np.zeros(512)
filter_adsr_curve = np.zeros(512)


def update_adsr_curve():
    """Update the amplitude ADSR curve based on current ADSR settings."""
    global adsr_curve
    total_time = adsr['attack'] + adsr['decay'] + adsr['release']
    t = np.linspace(0, total_time, len(adsr_curve))
    curve = np.zeros_like(t)

    for i, time in enumerate(t):
        if time < adsr['attack']:
            curve[i] = time / adsr['attack']
        elif time < adsr['attack'] + adsr['decay']:
            dt = time - adsr['attack']
            curve[i] = 1 - (1 - adsr['sustain']) * (dt / adsr['decay'])
        elif time < total_time:
            rt = time - (adsr['attack'] + adsr['decay'])
            curve[i] = adsr['sustain'] * (1 - rt / adsr['release'])
        else:
            curve[i] = 0

    adsr_curve = curve


def update_filter_adsr_curve():
    """Update the filter ADSR curve based on current filter ADSR settings."""
    global filter_adsr_curve
    total_time = filter_adsr['attack'] + filter_adsr['decay'] + filter_adsr['release']
    t = np.linspace(0, total_time, len(filter_adsr_curve))
    curve = np.zeros_like(t)

    for i, time in enumerate(t):
        if time < filter_adsr['attack']:
            curve[i] = time / filter_adsr['attack']
        elif time < filter_adsr['attack'] + filter_adsr['decay']:
            dt = time - filter_adsr['attack']
            curve[i] = 1 - (1 - filter_adsr['sustain']) * (dt / filter_adsr['decay'])
        elif time < total_time:
            rt = time - (filter_adsr['attack'] + filter_adsr['decay'])
            curve[i] = filter_adsr['sustain'] * (1 - rt / filter_adsr['release'])
        else:
            curve[i] = 0

    filter_adsr_curve = curve


def get_adsr_parameter_steps():
    """Return the step size for each ADSR parameter."""
    return {
        'attack': 0.005,
        'decay': 0.05,
        'sustain': 0.05,
        'release': 0.05,
    }


# Initialize the ADSR curves
update_adsr_curve()
update_filter_adsr_curve()
