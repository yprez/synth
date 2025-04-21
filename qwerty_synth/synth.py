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
        self.lfo_env_time = 0.0  # Time tracker for LFO attack envelope - reset for each new note
        self.done = False
        self.env_time = 0.0
        self.filter_env_time = 0.0  # Time tracker for filter envelope
        self.released = False
        self.last_env_level = 0.0
        self.last_filter_env_level = 0.0  # Last filter envelope level
        self.key = None  # Store the key used to activate this oscillator
        self.velocity = 1.0  # Default velocity (volume multiplier)

        # Precompute wave generation functions for performance
        self._wave_funcs = {
            'sine': lambda phase: np.sin(phase),
            'square': lambda phase: np.sign(np.sin(phase)),
            'triangle': lambda phase: 2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1,
            'sawtooth': lambda phase: 2 * ((phase / (2 * np.pi)) % 1) - 1
        }

    def generate(self, frames):
        """Generate audio samples with the current oscillator settings."""
        if self.done:
            return np.zeros(frames), np.zeros(frames)  # Return both amp and filter envs

        # Generate time array for LFO
        t = np.arange(frames) / config.sample_rate + self.lfo_phase

        # Compute LFO envelope with delay
        if self.lfo_env_time < config.lfo_delay_time:
            # In delay phase - no LFO
            lfo_env = 0.0
        elif config.lfo_attack_time > 0:
            # In attack phase - apply attack envelope after delay
            attack_time = self.lfo_env_time - config.lfo_delay_time
            lfo_env = np.clip(attack_time / config.lfo_attack_time, 0, 1.0)
        else:
            # No attack time - full envelope after delay
            lfo_env = 1.0

        lfo_env_array = np.full(frames, lfo_env)

        # Generate LFO signal with delay and attack envelope
        lfo = lfo_env_array * config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)

        # Increment LFO envelope time
        self.lfo_env_time += frames / config.sample_rate

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

        # Generate waveform using precomputed function
        wave_func = self._wave_funcs.get(self.waveform, self._wave_funcs['sine'])
        wave = wave_func(phase_array)

        # Generate amplitude envelope
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

        # Generate filter envelope
        filter_env = np.zeros(frames)
        for i in range(frames):
            time = self.filter_env_time + i / config.sample_rate
            if not self.released:
                if time < adsr.filter_adsr['attack']:
                    filter_env[i] = (time / adsr.filter_adsr['attack'])
                elif time < adsr.filter_adsr['attack'] + adsr.filter_adsr['decay']:
                    dt = time - adsr.filter_adsr['attack']
                    filter_env[i] = 1 - (1 - adsr.filter_adsr['sustain']) * (dt / adsr.filter_adsr['decay'])
                else:
                    filter_env[i] = adsr.filter_adsr['sustain']
            else:
                if time < adsr.filter_adsr['release']:
                    filter_env[i] = self.last_filter_env_level * (1 - time / adsr.filter_adsr['release'])
                else:
                    filter_env[i] = 0.0
                    # Note: we don't set self.done here as amplitude envelope controls that

        # Update envelope trackers
        self.env_time += frames / config.sample_rate
        self.last_env_level = env[-1]
        self.filter_env_time += frames / config.sample_rate
        self.last_filter_env_level = filter_env[-1]

        # Apply LFO modulation to volume if target is 'volume'
        if config.lfo_target == 'volume':
            # Modulate amplitude using LFO (tremolo effect)
            # Ensure values stay positive
            env = env * (1.0 + lfo)
            # Clip to avoid extreme values
            np.clip(env, 0.0, 1.0, out=env)

        # Apply envelope and velocity to the wave
        output = wave * env * self.velocity

        # Update LFO phase for next buffer
        self.lfo_phase += frames / config.sample_rate

        return output, filter_env


def audio_callback(outdata, frames, time_info, status):
    """Audio callback function for the sounddevice output stream."""
    if status:
        print(f"Audio callback status: {status}")

    buffer = np.zeros(frames)
    unfiltered_buffer = np.zeros(frames)
    filter_env_buffer = np.zeros(frames)  # Accumulate filter envelope values

    # If LFO is targeting filter cutoff, generate global LFO signal
    lfo_cutoff = None
    if config.lfo_target == 'cutoff':
        t = np.arange(frames) / config.sample_rate
        lfo = config.lfo_depth * np.sin(2 * np.pi * config.lfo_rate * t)
        # Map LFO to a range suitable for cutoff modulation (e.g., +/- 50% of current cutoff)
        lfo_cutoff = lfo

    num_active_notes = 0
    with config.notes_lock:
        finished_keys = []

        # Limit the number of active notes to prevent CPU overload
        active_note_items = list(config.active_notes.items())

        # Sort notes by newest (highest env_time means oldest)
        # We want to keep newer notes and release older ones if over the limit
        sorted_notes = sorted(active_note_items, key=lambda x: x[1].env_time)

        # If we have more notes than our limit, release the oldest ones
        if len(sorted_notes) > config.max_active_notes:
            for key, osc in sorted_notes[config.max_active_notes:]:
                if not osc.released:
                    osc.released = True
                    osc.last_env_level = osc.last_env_level  # Preserve the current envelope level

        # Process only the active notes up to our limit
        for key, osc in active_note_items:
            wave, osc_filter_env = osc.generate(frames)
            buffer += wave
            unfiltered_buffer += wave
            filter_env_buffer += osc_filter_env  # Add this oscillator's filter env to buffer
            if osc.done:
                finished_keys.append(key)
            else:
                num_active_notes += 1

        for key in finished_keys:
            del config.active_notes[key]

    # Apply normalization to prevent clipping with multiple notes
    if num_active_notes > 0:
        # Calculate RMS value of the buffer
        rms = np.sqrt(np.mean(buffer ** 2))

        # Apply dynamic scaling only if our signal is too loud
        # This is more efficient than normalizing all the time
        if rms > 0.3:  # If RMS is above threshold
            scaling_factor = 0.3 / rms
            buffer *= scaling_factor
            unfiltered_buffer *= scaling_factor

        # Store a copy of the unfiltered buffer for visualization
        unfiltered_buffer_copy = buffer.copy()

        # Apply filter only if needed (more efficient)
        # Normalize filter envelope if active notes > 0
        filter_env_buffer = filter_env_buffer / num_active_notes

        # Apply filter with envelope modulation
        # Only apply if we have a valid cutoff (below Nyquist)
        if filter.cutoff < config.sample_rate / 2:
            buffer = filter.apply_filter(buffer, lfo_cutoff, filter_env_buffer)
    else:
        unfiltered_buffer_copy = np.zeros(frames)

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
        blocksize=4096,
        latency='high'  # Use 'high' for more stability
    )
    return stream
