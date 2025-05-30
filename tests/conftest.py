"""Pytest configuration file with shared fixtures and test setup."""

import numpy as np
import pytest
import sys
import os

# Add the parent directory to the Python path so we can import qwerty_synth
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qwerty_synth import config


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration to default values before each test."""
    # Reset filter state to ensure clean start for all tests
    try:
        from qwerty_synth.filter import reset_filter_state
        reset_filter_state()
    except ImportError:
        # Filter module may not be available in all tests
        pass

    # Store original values
    original_values = {}
    for attr in dir(config):
        if not attr.startswith('_'):
            original_values[attr] = getattr(config, attr)

    # Reset to defaults
    config.sample_rate = 44100
    config.volume = 0.5
    config.fade_duration = 0.01
    config.max_active_notes = 8
    config.blocksize = 2048
    config.octave_offset = 0
    config.waveform_type = 'sine'

    # ADSR defaults
    config.adsr = {
        'attack': 0.01,
        'decay': 0.02,
        'sustain': 0.1,
        'release': 0.2,
    }

    # Filter ADSR defaults
    config.filter_adsr = {
        'attack': 0.05,
        'decay': 0.1,
        'sustain': 0.4,
        'release': 0.3,
    }

    # Filter defaults
    config.filter_cutoff = 10000
    config.filter_resonance = 0.0
    config.filter_enabled = False
    config.filter_type = 'lowpass'
    config.filter_topology = 'svf'
    config.filter_slope = 24
    config.filter_env_amount = 5000

    # Drive defaults
    config.drive_gain = 1.2
    config.drive_on = False
    config.drive_type = 'tanh'
    config.drive_tone = 0.0
    config.drive_mix = 1.0
    config.drive_asymmetry = 0.2

    # LFO defaults
    config.lfo_rate = 5.0
    config.lfo_depth = 0.1
    config.lfo_target = 'pitch'
    config.lfo_attack_time = 0.2
    config.lfo_delay_time = 0.1
    config.lfo_enabled = False

    # Delay defaults
    config.delay_time_ms = 350
    config.delay_feedback = 0.1
    config.delay_mix = 0.1
    config.delay_enabled = False
    config.delay_sync_enabled = True
    config.delay_pingpong = False

    # Chorus defaults
    config.chorus_rate = 0.5
    config.chorus_depth = 0.005
    config.chorus_mix = 0.2
    config.chorus_voices = 1
    config.chorus_enabled = False

    # Mono mode defaults
    config.mono_mode = False
    config.glide_time = 0.05
    config.mono_pressed_keys = []

    # Clear active notes
    config.active_notes = {}

    # Arpeggiator defaults
    config.arpeggiator_enabled = False
    config.arpeggiator_pattern = 'up'
    config.arpeggiator_rate = 120
    config.arpeggiator_gate = 0.8
    config.arpeggiator_octave_range = 1
    config.arpeggiator_sync_to_bpm = True
    config.arpeggiator_sustain_base = True
    config.arpeggiator_held_notes = set()

    yield  # Run the test

    # Restore original values after test
    for attr, value in original_values.items():
        if hasattr(config, attr):
            setattr(config, attr, value)


@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing."""
    frames = 1024
    sample_rate = 44100
    frequency = 440.0  # A4
    t = np.arange(frames) / sample_rate
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


@pytest.fixture
def stereo_audio():
    """Generate stereo audio data for testing."""
    frames = 1024
    sample_rate = 44100
    frequency_L = 440.0  # A4
    frequency_R = 554.37  # C#5
    t = np.arange(frames) / sample_rate
    left = np.sin(2 * np.pi * frequency_L * t).astype(np.float32)
    right = np.sin(2 * np.pi * frequency_R * t).astype(np.float32)
    return left, right


@pytest.fixture
def silence():
    """Generate silent audio for testing."""
    return np.zeros(1024, dtype=np.float32)


@pytest.fixture
def noise():
    """Generate white noise for testing."""
    return np.random.normal(0, 0.1, 1024).astype(np.float32)


@pytest.fixture
def mock_oscillator():
    """Create a mock oscillator for testing."""
    from qwerty_synth.synth import Oscillator
    return Oscillator(440.0, 'sine')


class AudioTestHelper:
    """Helper class for audio testing utilities."""

    @staticmethod
    def is_silent(audio, threshold=1e-6):
        """Check if audio signal is effectively silent."""
        return np.max(np.abs(audio)) < threshold

    @staticmethod
    def has_signal(audio, threshold=1e-6):
        """Check if audio signal has meaningful content."""
        return np.max(np.abs(audio)) > threshold

    @staticmethod
    def rms(audio):
        """Calculate RMS value of audio signal."""
        return np.sqrt(np.mean(audio ** 2))

    @staticmethod
    def peak(audio):
        """Get peak amplitude of audio signal."""
        return np.max(np.abs(audio))

    @staticmethod
    def is_clipped(audio, threshold=0.99):
        """Check if audio signal is clipped."""
        return np.any(np.abs(audio) > threshold)

    @staticmethod
    def frequency_content(audio, sample_rate=44100):
        """Get frequency content of audio signal using FFT."""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        return freqs, np.abs(fft)

    @staticmethod
    def dominant_frequency(audio, sample_rate=44100):
        """Find the dominant frequency in an audio signal."""
        freqs, magnitude = AudioTestHelper.frequency_content(audio, sample_rate)
        # Ignore DC component
        if len(magnitude) > 1:
            return freqs[np.argmax(magnitude[1:]) + 1]
        return 0.0


@pytest.fixture
def audio_helper():
    """Provide audio testing helper utilities."""
    return AudioTestHelper
