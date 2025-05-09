"""Wave folder / soft-clip drive implementation with multiple algorithms and tone control."""

import numpy as np
from qwerty_synth import config

# Filter state for tone control (persists between buffer calls)
_prev_filtered_sample = 0.0
_prev_tone = 0.0
_prev_alpha = 0.5

def apply_drive(samples):
    """
    Apply drive (soft clipping/distortion) effect to audio samples.

    Args:
        samples: Input audio samples

    Returns:
        Processed audio samples with drive effect
    """
    global _prev_filtered_sample, _prev_tone, _prev_alpha

    # Skip processing entirely if drive is off or at default value
    if not config.drive_on or abs(config.drive_gain - 1.0) < 0.01:
        # Reset filter state
        _prev_filtered_sample = 0.0
        return samples

    # Store original samples for mix control
    dry_samples = samples.copy()

    # Process with drive effect
    if config.drive_gain <= 1.0:
        # Just use drive_gain as a volume multiplier when <= 1.0
        wet_samples = samples * config.drive_gain
    else:
        # Pre-gain stage (boost signal)
        boosted = samples * config.drive_gain

        # Apply selected drive algorithm
        if config.drive_type == 'tanh':
            # Classic smooth tanh soft clipping
            wet_samples = np.tanh(boosted)
        elif config.drive_type == 'arctan':
            # Arctan clipping (gentler than tanh)
            wet_samples = (2/np.pi) * np.arctan(np.pi * boosted/2)
        elif config.drive_type == 'cubic':
            # Cubic soft clipping
            wet_samples = np.clip(boosted - (boosted**3)/3, -1.0, 1.0)
        elif config.drive_type == 'fuzz':
            # Hard clipping with a bit of smoothing
            wet_samples = np.sign(boosted) * (1 - np.exp(-np.abs(boosted)))
        elif config.drive_type == 'asymmetric':
            # Asymmetric clipping for tube-like distortion
            positive = np.where(boosted >= 0, boosted, 0)
            negative = np.where(boosted < 0, boosted, 0)
            wet_samples = np.tanh(positive * (1 + config.drive_asymmetry)) + np.tanh(negative * (1 - config.drive_asymmetry))
        else:
            # Default to tanh if unknown type
            wet_samples = np.tanh(boosted)

    # Apply tone control (simple high/low frequency balance)
    if abs(config.drive_tone) > 0.01:
        # Simple 1-pole filter as a tone control
        tone = np.clip(config.drive_tone, -0.95, 0.95)

        # If tone has changed significantly, smooth the transition of alpha
        if abs(tone - _prev_tone) > 0.01:
            # Gradually update alpha to avoid clicks
            alpha_target = (tone + 1) / 2  # Map from -0.95..0.95 to 0.025..0.975
            _prev_alpha = alpha_target
            _prev_tone = tone

        alpha = _prev_alpha

        # Initialize filtered output array
        filtered = np.zeros_like(wet_samples)

        # Apply filter with state maintained between calls
        filtered[0] = alpha * wet_samples[0] + (1 - alpha) * _prev_filtered_sample

        for i in range(1, len(wet_samples)):
            filtered[i] = alpha * wet_samples[i] + (1 - alpha) * filtered[i-1]

        # Store last sample for next buffer
        _prev_filtered_sample = filtered[-1]

        wet_samples = filtered
    else:
        # Reset filter state when not using tone
        _prev_filtered_sample = 0.0

    # Apply mix control (blend between dry and wet signals)
    return dry_samples * (1 - config.drive_mix) + wet_samples * config.drive_mix
