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
        self.target_freq = freq  # Target frequency for glide
        self.waveform = waveform
        self.phase = 0.0
        self.lfo_phase = 0.0  # Phase for the LFO
        self.done = False
        self.env_time = 0.0
        self.released = False
        self.last_env_level = 0.0
        self.key = None  # Store the key used to activate this oscillator

    def generate(self, frames):
        """Generate audio samples with the current oscillator settings."""
        if self.done:
            return np.zeros(frames)

        # Generate time array for LFO
        t = np.arange(frames) / config.sample_rate + self.lfo_phase

        # Generate LFO signal
        lfo = config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)

        # Implement glide effect if target frequency differs from current frequency
        if self.freq != self.target_freq:
            # Ensure glide_time is at least 0.001 seconds (1ms) to avoid division by zero
            safe_glide_time = max(0.001, config.glide_time)

            # Calculate step size based on glide time
            freq_step = (self.target_freq - self.freq) / (safe_glide_time * config.sample_rate)
            # Calculate how many frames we need for the glide
            glide_frames = min(frames, int(safe_glide_time * config.sample_rate))

            if glide_frames > 0:
                # Create a frequency array that smoothly transitions from current to target
                freq_array = np.zeros(frames)

                # Linear interpolation for the glide portion
                for i in range(glide_frames):
                    freq_array[i] = self.freq + freq_step * i

                # Fill the rest with the target frequency
                if glide_frames < frames:
                    freq_array[glide_frames:] = self.target_freq

                # Update current frequency to where we ended
                self.freq = freq_array[-1]
            else:
                # If glide_time is zero, jump immediately to target frequency
                self.freq = self.target_freq
                freq_array = np.full(frames, self.target_freq)
        else:
            # No glide needed, use constant frequency
            freq_array = np.full(frames, self.freq)

        # Apply LFO modulation to pitch if target is 'pitch'
        if config.lfo_target == 'pitch':
            # Modulate frequency using LFO (vibrato effect)
            # The exponential formula converts semitones to frequency ratio
            # We divide by 12 to convert LFO range to semitones
            mod_freq_array = freq_array * (2 ** (lfo / 12))
        else:
            mod_freq_array = freq_array

        # Calculate phase increments based on modulated frequency
        phase_increments = 2 * np.pi * mod_freq_array / config.sample_rate

        # Accumulate phase
        phase_array = np.zeros(frames)
        current_phase = self.phase

        for i in range(frames):
            phase_array[i] = current_phase
            current_phase = (current_phase + phase_increments[i]) % (2 * np.pi)

        self.phase = current_phase

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

        # Apply LFO modulation to volume if target is 'volume'
        if config.lfo_target == 'volume':
            # Modulate amplitude using LFO (tremolo effect)
            # Ensure values stay positive
            env = env * (1.0 + lfo)
            # Clip to avoid extreme values
            np.clip(env, 0.0, 1.0, out=env)

        # Apply envelope to the wave (but don't filter individual oscillators)
        output = wave * env

        # Update LFO phase for next buffer
        self.lfo_phase += frames / config.sample_rate

        return output


def audio_callback(outdata, frames, time_info, status):
    """Audio callback function for the sounddevice output stream."""
    buffer = np.zeros(frames)
    unfiltered_buffer = np.zeros(frames)

    # If LFO is targeting filter cutoff, generate global LFO signal
    lfo_cutoff = None
    if config.lfo_target == 'cutoff':
        t = np.arange(frames) / config.sample_rate
        lfo = config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)
        # Map LFO to a range suitable for cutoff modulation (e.g., +/- 50% of current cutoff)
        lfo_cutoff = lfo

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
        buffer = filter.apply_filter(buffer, lfo_cutoff)

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
