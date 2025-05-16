"""ADSR (Attack, Decay, Sustain, Release) envelope functionality."""

import numpy as np
from qwerty_synth import config

# Pre-calculated ADSR curves for visualization
adsr_curve = np.zeros(512)
filter_adsr_curve = np.zeros(512)


def update_adsr_curve():
    """Update the amplitude ADSR curve based on current ADSR settings."""
    global adsr_curve
    total_time = config.adsr['attack'] + config.adsr['decay'] + config.adsr['release']
    t = np.linspace(0, total_time, len(adsr_curve))
    curve = np.zeros_like(t)

    for i, time in enumerate(t):
        if time < config.adsr['attack']:
            curve[i] = time / config.adsr['attack']
        elif time < config.adsr['attack'] + config.adsr['decay']:
            dt = time - config.adsr['attack']
            curve[i] = 1 - (1 - config.adsr['sustain']) * (dt / config.adsr['decay'])
        elif time < total_time:
            rt = time - (config.adsr['attack'] + config.adsr['decay'])
            curve[i] = config.adsr['sustain'] * (1 - rt / config.adsr['release'])
        else:
            curve[i] = 0

    adsr_curve = curve


def update_filter_adsr_curve():
    """Update the filter ADSR curve based on current filter ADSR settings."""
    global filter_adsr_curve
    total_time = config.filter_adsr['attack'] + config.filter_adsr['decay'] + config.filter_adsr['release']
    t = np.linspace(0, total_time, len(filter_adsr_curve))
    curve = np.zeros_like(t)

    for i, time in enumerate(t):
        if time < config.filter_adsr['attack']:
            curve[i] = time / config.filter_adsr['attack']
        elif time < config.filter_adsr['attack'] + config.filter_adsr['decay']:
            dt = time - config.filter_adsr['attack']
            curve[i] = 1 - (1 - config.filter_adsr['sustain']) * (dt / config.filter_adsr['decay'])
        elif time < total_time:
            rt = time - (config.filter_adsr['attack'] + config.filter_adsr['decay'])
            curve[i] = config.filter_adsr['sustain'] * (1 - rt / config.filter_adsr['release'])
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
