"""Synthesizer core functionality including Oscillator and audio processing."""

import numpy as np
import sounddevice as sd

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import filter
from qwerty_synth.delay import Delay
from qwerty_synth.chorus import Chorus
from qwerty_synth.lfo import LFO
from qwerty_synth.drive import apply_drive

# Initialize effects
delay = Delay(config.sample_rate, config.delay_time_ms)
chorus = Chorus(config.sample_rate)
global_lfo = LFO()  # Global LFO instance for filter modulation

# Parameter change flags
_chorus_params_changed = False
_delay_params_changed = False

def update_chorus_params():
    """Update chorus parameters if they've changed."""
    global _chorus_params_changed
    if _chorus_params_changed:
        chorus.set_rate(config.chorus_rate)
        chorus.set_depth(config.chorus_depth)
        chorus.set_mix(config.chorus_mix)
        chorus.set_voices(config.chorus_voices)
        _chorus_params_changed = False

def update_delay_params():
    """Update delay parameters if they've changed."""
    global _delay_params_changed
    if _delay_params_changed:
        delay.set_time(config.delay_time_ms)
        _delay_params_changed = False

def mark_chorus_params_changed():
    """Mark chorus parameters as changed."""
    global _chorus_params_changed
    _chorus_params_changed = True

def mark_delay_params_changed():
    """Mark delay parameters as changed."""
    global _delay_params_changed
    _delay_params_changed = True

class Oscillator:
    """Oscillator that generates waveforms with ADSR envelope."""

    def __init__(self, freq, waveform):
        """Initialize oscillator with frequency and waveform type."""
        self.freq = freq
        self.target_freq = freq  # Target frequency for glide
        self.waveform = waveform
        self.phase = 0.0
        self.lfo = LFO()
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

        # Generate LFO signal
        lfo = self.lfo.generate(frames)

        # Fast path for non-glide cases (most common scenario)
        if self.freq == self.target_freq:
            # No glide needed, use constant frequency
            freq_array = np.full(frames, self.freq)
        else:
            # Implement glide effect if target frequency differs from current frequency
            # Ensure glide_time is at least 0.001 seconds (1ms) to avoid division by zero
            safe_glide_time = max(0.001, config.glide_time)

            # Calculate step size based on glide time
            freq_step = (self.target_freq - self.freq) / (safe_glide_time * config.sample_rate)
            # Calculate how many frames we need for the glide
            glide_frames = min(frames, int(safe_glide_time * config.sample_rate))

            if glide_frames > 0:
                # Vectorized glide implementation
                idx = np.arange(frames)
                freq_array = np.where(
                    idx < glide_frames,
                    self.freq + freq_step * idx,
                    self.target_freq
                )
                self.freq = freq_array[-1]
            else:
                # If glide_time is zero, jump immediately to target frequency
                self.freq = self.target_freq
                freq_array = np.full(frames, self.target_freq)

        # Apply LFO modulation to pitch - only if LFO is actually doing something
        if not np.all(lfo == 0):
            mod_freq_array = self.lfo.apply_pitch_modulation(freq_array, lfo)
        else:
            mod_freq_array = freq_array

        # Calculate phase increments based on modulated frequency
        phase_increments = 2 * np.pi * mod_freq_array / config.sample_rate

        # Vectorized phase accumulation
        phase_array = (self.phase + np.cumsum(phase_increments)) % (2 * np.pi)
        self.phase = phase_array[-1]

        # Generate waveform using precomputed function
        wave_func = self._wave_funcs.get(self.waveform, self._wave_funcs['sine'])
        wave = wave_func(phase_array)

        # Check if we're in a simple case for envelope (no release, past attack/decay)
        simple_env_case = (not self.released and
                           self.env_time > adsr.adsr['attack'] + adsr.adsr['decay'])

        # Fast path for sustained notes
        if simple_env_case:
            env = np.full(frames, adsr.adsr['sustain'])
            self.env_time += frames / config.sample_rate
            self.last_env_level = adsr.adsr['sustain']
        else:
            # Vectorized ADSR envelope generation
            time_array = self.env_time + np.arange(frames) / config.sample_rate

            if not self.released:
                # Attack phase
                attack_mask = time_array < adsr.adsr['attack']
                env = np.where(attack_mask,
                               time_array / adsr.adsr['attack'],
                               0)

                # Decay phase
                decay_start = adsr.adsr['attack']
                decay_end = decay_start + adsr.adsr['decay']
                decay_mask = (time_array >= decay_start) & (time_array < decay_end)
                env = np.where(decay_mask,
                             1 - (1 - adsr.adsr['sustain']) * ((time_array - decay_start) / adsr.adsr['decay']),
                             env)

                # Sustain phase
                sustain_mask = time_array >= decay_end
                env = np.where(sustain_mask,
                               adsr.adsr['sustain'],
                               env)
            else:
                # Release phase
                release_mask = time_array < adsr.adsr['release']
                env = np.where(release_mask,
                             self.last_env_level * (1 - time_array / adsr.adsr['release']),
                             0)
                self.done = time_array[-1] >= adsr.adsr['release']

            # Update envelope trackers
            self.env_time += frames / config.sample_rate
            self.last_env_level = env[-1]

        # Similar approach for filter envelope
        simple_filter_env_case = (
            not self.released and
            self.filter_env_time > adsr.filter_adsr['attack'] + adsr.filter_adsr['decay']
        )

        if simple_filter_env_case:
            filter_env = np.full(frames, adsr.filter_adsr['sustain'])
            self.filter_env_time += frames / config.sample_rate
            self.last_filter_env_level = adsr.filter_adsr['sustain']
        else:
            # Vectorized filter ADSR envelope generation
            time_array = self.filter_env_time + np.arange(frames) / config.sample_rate

            if not self.released:
                # Attack phase
                attack_mask = time_array < adsr.filter_adsr['attack']
                filter_env = np.where(attack_mask,
                                      time_array / adsr.filter_adsr['attack'],
                                      0)

                # Decay phase
                decay_start = adsr.filter_adsr['attack']
                decay_end = decay_start + adsr.filter_adsr['decay']
                decay_mask = (time_array >= decay_start) & (time_array < decay_end)
                filter_env = np.where(
                    decay_mask,
                    1 - (1 - adsr.filter_adsr['sustain']) * ((time_array - decay_start) / adsr.filter_adsr['decay']),
                    filter_env
                )

                # Sustain phase
                sustain_mask = time_array >= decay_end
                filter_env = np.where(sustain_mask,
                                    adsr.filter_adsr['sustain'],
                                    filter_env)
            else:
                # Release phase
                release_mask = time_array < adsr.filter_adsr['release']
                filter_env = np.where(
                    release_mask,
                    self.last_filter_env_level * (1 - time_array / adsr.filter_adsr['release']),
                    0
                )

            # Update filter envelope trackers
            self.filter_env_time += frames / config.sample_rate
            self.last_filter_env_level = filter_env[-1]

        # Apply LFO modulation to volume - only if needed
        if not np.all(lfo == 0) and config.lfo_target == 'volume':
            env = self.lfo.apply_amplitude_modulation(env, lfo)

        # Apply envelope and velocity to the wave
        output = wave * env * self.velocity

        return output, filter_env


def audio_callback(outdata, frames, time_info, status):
    """Audio callback function for the sounddevice output stream."""
    if status:
        print(f"Audio callback status: {status}")

    buffer = np.zeros(frames)
    filter_env_buffer = np.zeros(frames)  # Accumulate filter envelope values

    # Use global LFO for filter cutoff modulation
    lfo_cutoff = global_lfo.get_cutoff_modulation(frames)

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

        # Process only the active notes up to our limit - only the most recent ones
        processing_notes = sorted_notes[:config.max_active_notes]
        if processing_notes:
            for key, osc in processing_notes:
                wave, osc_filter_env = osc.generate(frames)
                buffer += wave
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

        # Normalize filter envelope if active notes > 0
        filter_env_buffer = filter_env_buffer / num_active_notes

        # Apply filter with envelope modulation
        # Only apply if we have a valid cutoff (below Nyquist)
        if filter.cutoff < config.sample_rate / 2:
            buffer = filter.apply_filter(buffer, lfo_cutoff, filter_env_buffer)

        # Apply drive effect (wave folding/soft clipping) after filter
        buffer = apply_drive(buffer)

        # Optional safety clip to prevent extreme peaks after drive
        np.clip(buffer, -1.2, 1.2, out=buffer)

    # Create stereo buffers from mono buffer
    buffer_L = buffer
    buffer_R = buffer

    # Apply chorus effect if enabled - before the delay
    if config.chorus_enabled:
        # Update chorus parameters only if they've changed
        update_chorus_params()
        # Process audio through chorus
        buffer_L, buffer_R = chorus.process(buffer_L, buffer_R)

    # Apply delay effect if enabled - after all oscillator processing
    if config.delay_enabled:
        # Update delay parameters only if they've changed
        update_delay_params()
        if config.delay_pingpong:
            # Apply ping-pong stereo delay
            buffer_L, buffer_R = delay.pingpong(
                buffer_L, buffer_R,
                config.delay_mix,
                config.delay_feedback
            )
        else:
            # Apply traditional mono delay
            # Convert stereo to mono by averaging L and R if chorus has been applied
            mono_input = (buffer_L + buffer_R) / 2 if config.chorus_enabled else buffer
            mono_delayed = delay.process_block(
                mono_input,
                config.delay_feedback,
                config.delay_mix
            )
            buffer_L = buffer_R = mono_delayed

    # Output stereo audio - avoid unnecessary copy operations
    outdata[:, 0] = config.volume * buffer_L
    outdata[:, 1] = config.volume * buffer_R

    with config.buffer_lock:
        config.waveform_buffer = np.roll(config.waveform_buffer, -frames)
        config.waveform_buffer[-frames:] = buffer


def create_audio_stream(latency='high'):
    """Create and return a configured audio output stream.

    Args:
        latency: 'high' for stability or 'low' for reduced latency
    """
    stream = sd.OutputStream(
        samplerate=config.sample_rate,
        channels=2,  # Stereo output
        callback=audio_callback,
        blocksize=config.blocksize,
        latency=latency
    )
    return stream
