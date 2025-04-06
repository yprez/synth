"""ADSR (Attack, Decay, Sustain, Release) envelope functionality."""

import numpy as np

# ADSR envelope parameters
adsr = {
    'attack': 0.1,
    'decay': 0.05,
    'sustain': 0.8,
    'release': 0.5,
}

# Pre-calculated ADSR curve for visualization
adsr_curve = np.zeros(512)


def update_adsr_curve():
    """Update the ADSR curve based on current ADSR settings."""
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


def get_adsr_parameter_steps():
    """Return the step size for each ADSR parameter."""
    return {
        'attack': 0.005,
        'decay': 0.05,
        'sustain': 0.05,
        'release': 0.05,
    }


# Initialize the ADSR curve
update_adsr_curve()
