"""Wave folder / soft-clip drive implementation using tanh."""

import numpy as np
from qwerty_synth import config


def apply_drive(x):
    """Soft-clip via tanh; x is 1-D numpy array."""
    if not config.drive_on or config.drive_gain <= 0.0:
        return x
    return np.tanh(x * config.drive_gain)
