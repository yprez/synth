"""Wave folder / soft-clip drive implementation using tanh."""

import numpy as np
from qwerty_synth import config


def apply_drive(samples):
    """
    Apply drive (soft clipping/distortion) effect to audio samples.

    Args:
        samples: Input audio samples

    Returns:
        Processed audio samples with drive effect
    """
    # Skip processing entirely if drive is off or at default value
    if not config.drive_on or abs(config.drive_gain - 1.0) < 0.01:
        return samples

    # Process with drive effect
    if config.drive_gain <= 1.0:
        # Just use drive_gain as a volume multiplier when <= 1.0
        return samples * config.drive_gain

    # Apply soft clipping with gain
    # Pre-gain stage (boost signal)
    boosted = samples * config.drive_gain

    # Soft clipping stage
    # Using a smooth tanh function to create soft clipping
    # as signal exceeds -1 to 1 range
    return np.tanh(boosted)
