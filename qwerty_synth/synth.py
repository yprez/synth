"""Synthesizer core functionality including Oscillator and audio processing."""

import numpy as np
import sounddevice as sd

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import filter


class Oscillator:
    """Oscillator that generates waveforms with ADSR envelope."""

    def __init__(self, freq, waveform):
        """Initialize oscillator with frequency and waveform type."""
        self.freq = freq
        self.waveform = waveform
        self.phase = 0.0
        self.done = False
        self.env_time = 0.0
        self.released = False
        self.last_env_level = 0.0

    def generate(self, frames):
        """Generate audio samples with the current oscillator settings."""
        if self.done:
            return np.zeros(frames), np.zeros(frames)

        np.arange(frames) / config.sample_rate
        phase_increment = 2 * np.pi * self.freq / config.sample_rate
        phase_array = self.phase + phase_increment * np.arange(frames)
        self.phase = (phase_array[-1] + phase_increment) % (2 * np.pi)

        if self.waveform == 'sine':
            wave = np.sin(phase_array)
        elif self.waveform == 'square':
            wave = np.sign(np.sin(phase_array))
        elif self.waveform == 'triangle':
            wave = 2 * np.abs(2 * ((phase_array / (2 * np.pi)) % 1) - 1) - 1
        elif self.waveform == 'sawtooth':
            wave = 2 * ((phase_array / (2 * np.pi)) % 1) - 1
        else:
            wave = np.sin(phase_array)

        env = np.zeros(frames)
        for i in range(frames):
            time = self.env_time + i / config.sample_rate
            if not self.released:
                if time < adsr.adsr['attack']:
                    env[i] = (time / adsr.adsr['attack'])
                elif time < adsr.adsr['attack'] + adsr.adsr['decay']:
                    dt = time - adsr.adsr['attack']
                    env[i] = 1 - (1 - adsr.adsr['sustain']) * (dt / adsr.adsr['decay'])
                else:
                    env[i] = adsr.adsr['sustain']
            else:
                if time < adsr.adsr['release']:
                    env[i] = self.last_env_level * (1 - time / adsr.adsr['release'])
                else:
                    env[i] = 0.0
                    self.done = True

        self.env_time += frames / config.sample_rate
        self.last_env_level = env[-1]

        # Apply envelope to the wave (but don't filter individual oscillators)
        output = wave * env

        return output


def audio_callback(outdata, frames, time_info, status):
    """Audio callback function for the sounddevice output stream."""
    buffer = np.zeros(frames)
    unfiltered_buffer = np.zeros(frames)

    with config.notes_lock:
        finished_keys = []

        for key, osc in config.active_notes.items():
            wave = osc.generate(frames)
            buffer += wave
            unfiltered_buffer += wave
            if osc.done:
                finished_keys.append(key)

        for key in finished_keys:
            del config.active_notes[key]

    # Store a copy of the unfiltered buffer before filtering
    unfiltered_buffer_copy = buffer.copy()

    # Apply filter to the mixed signal (only if there are active notes)
    if len(config.active_notes) > 0:
        buffer = filter.apply_filter(buffer)

        # Soft limiting/compression to prevent clipping while maintaining volume
        max_amplitude = np.max(np.abs(buffer))
        if max_amplitude > 0.95:  # Only compress if we're close to clipping
            compression_factor = 0.95 / max_amplitude
            buffer *= compression_factor
            unfiltered_buffer_copy *= compression_factor

    outdata[:] = (config.volume * buffer).reshape(-1, 1)

    with config.buffer_lock:
        config.waveform_buffer = np.roll(config.waveform_buffer, -frames)
        config.waveform_buffer[-frames:] = buffer
        config.unfiltered_buffer = np.roll(config.unfiltered_buffer, -frames)
        config.unfiltered_buffer[-frames:] = unfiltered_buffer_copy


def create_audio_stream():
    """Create and return a configured audio output stream."""
    stream = sd.OutputStream(
        samplerate=config.sample_rate,
        channels=1,
        callback=audio_callback,
        blocksize=1024
    )
    return stream
