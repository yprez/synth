"""Recording functionality for QWERTY Synth."""

import numpy as np
import soundfile as sf
from datetime import datetime
from pathlib import Path

# Recording state variables
recording_enabled = False
recorded_audio = []
current_recording_path = None


def start_recording(output_path=None):
    """Start recording audio to buffer.

    Args:
        output_path: Optional path for the recording. If None, one will be generated.

    Returns:
        The path where the recording will be saved.
    """
    global recording_enabled, recorded_audio, current_recording_path

    # Clear any previous recorded audio
    recorded_audio = []

    # Generate a filename if none provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = Path.home() / "qwerty_synth_recordings"

        # Create directory if it doesn't exist
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

        output_path = str(directory / f"qwerty_synth_{timestamp}.wav")

    current_recording_path = output_path
    recording_enabled = True

    return current_recording_path


def stop_recording(sample_rate=44100, bit_depth=24):
    """Stop recording and save to WAV file.

    Args:
        sample_rate: The sample rate of the recording
        bit_depth: The bit depth for the saved file (16 or 24)

    Returns:
        Path to the saved file or None if no recording was done
    """
    global recording_enabled, recorded_audio, current_recording_path

    if not recording_enabled or not recorded_audio:
        return None

    recording_enabled = False

    # Convert list of buffers to a single numpy array
    audio_data = np.vstack(recorded_audio)

    # Determine subtype based on bit depth
    subtype = 'PCM_24' if bit_depth == 24 else 'PCM_16'

    # Save the file
    sf.write(current_recording_path, audio_data, sample_rate, subtype=subtype)

    # Clear buffer to free memory
    recorded_audio = []

    saved_path = current_recording_path
    current_recording_path = None

    return saved_path


def add_audio_block(block):
    """Add an audio block to the recording buffer.

    Args:
        block: A numpy array containing stereo audio data
    """
    global recording_enabled, recorded_audio

    if recording_enabled:
        recorded_audio.append(block.copy())


def is_recording():
    """Return True if recording is in progress."""
    return recording_enabled


def get_recording_time():
    """Get the current recording time in seconds.

    Returns:
        The current recording duration in seconds
    """
    global recorded_audio

    if not recorded_audio:
        return 0

    # Calculate total frames and convert to time
    total_frames = sum(block.shape[0] for block in recorded_audio)
    return total_frames / 44100  # Assuming 44.1kHz sample rate
