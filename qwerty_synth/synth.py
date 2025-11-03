"""Synthesizer core functionality including Oscillator and audio processing."""

import numpy as np
import sounddevice as sd

from qwerty_synth import config
from qwerty_synth import filter
from qwerty_synth.delay import Delay
from qwerty_synth.chorus import Chorus
from qwerty_synth.reverb import Reverb
from qwerty_synth.lfo import LFO
from qwerty_synth.drive import apply_drive
from qwerty_synth import record
from qwerty_synth.event_scheduler import global_scheduler


# Initialize effects
delay = Delay(config.sample_rate, config.delay_time_ms)
chorus = Chorus(config.sample_rate)
reverb = Reverb(config.sample_rate)
global_lfo = LFO()  # Global LFO instance for filter modulation

# Global audio stream for simple programmatic use
_global_audio_stream = None

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
            'square': lambda phase: np.sign(np.sin(phase)) * 0.95,  # Scale to avoid clipping
            'triangle': lambda phase: 2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1,
            'sawtooth': lambda phase: 2 * ((phase / (2 * np.pi)) % 1) - 1
        }

    def generate(self, frames):
        """Generate audio samples with the current oscillator settings."""
        if self.done:
            return np.zeros(frames), np.zeros(frames)  # Return both amp and filter envs

        # Handle zero frames case
        if frames == 0:
            return np.zeros(0), np.zeros(0)

        # Glide
        if self.freq == self.target_freq:
            # Fast path for non-glide cases (most common scenario)
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

        # Generate LFO signal
        lfo = self.lfo.generate(frames)

        # Apply LFO modulation to pitch - only if LFO is enabled and actually doing something
        if config.lfo_enabled and not np.all(lfo == 0):
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
                           self.env_time > config.adsr['attack'] + config.adsr['decay'])

        # Fast path for sustained notes
        if simple_env_case:
            env = np.full(frames, config.adsr['sustain'])
            self.env_time += frames / config.sample_rate
            self.last_env_level = config.adsr['sustain']
        else:
            # Vectorized ADSR envelope generation
            time_array = self.env_time + np.arange(frames) / config.sample_rate

            if not self.released:
                # Attack phase
                attack_mask = time_array < config.adsr['attack']
                if config.adsr['attack'] > 0:
                    env = np.where(attack_mask,
                                   time_array / config.adsr['attack'],
                                   0)
                else:
                    # If attack is zero, immediately jump to full level
                    env = np.where(attack_mask, 1.0, 0)

                # Decay phase
                decay_start = config.adsr['attack']
                decay_end = decay_start + config.adsr['decay']
                decay_mask = (time_array >= decay_start) & (time_array < decay_end)
                if config.adsr['decay'] > 0:
                    env = np.where(decay_mask,
                                 1 - (1 - config.adsr['sustain']) * ((time_array - decay_start) / config.adsr['decay']),
                                 env)
                else:
                    # If decay is zero, immediately jump to sustain level
                    env = np.where(decay_mask, config.adsr['sustain'], env)

                # Sustain phase
                sustain_mask = time_array >= decay_end
                env = np.where(sustain_mask,
                               config.adsr['sustain'],
                               env)
            else:
                # Release phase
                release_mask = time_array < config.adsr['release']
                if config.adsr['release'] > 0:
                    env = np.where(release_mask,
                                 self.last_env_level * (1 - time_array / config.adsr['release']),
                                 0)
                    self.done = time_array[-1] >= config.adsr['release']
                else:
                    # If release is zero, immediately silence the note
                    env = np.zeros_like(time_array)
                    self.done = True

            # Update envelope trackers
            self.env_time += frames / config.sample_rate
            self.last_env_level = env[-1]

        # Similar approach for filter envelope
        simple_filter_env_case = (
            not self.released and
            self.filter_env_time > config.filter_adsr['attack'] + config.filter_adsr['decay']
        )

        if simple_filter_env_case:
            filter_env = np.full(frames, config.filter_adsr['sustain'])
            self.filter_env_time += frames / config.sample_rate
            self.last_filter_env_level = config.filter_adsr['sustain']
        else:
            # Vectorized filter ADSR envelope generation
            time_array = self.filter_env_time + np.arange(frames) / config.sample_rate

            if not self.released:
                # Attack phase
                attack_mask = time_array < config.filter_adsr['attack']
                if config.filter_adsr['attack'] > 0:
                    filter_env = np.where(attack_mask,
                                          time_array / config.filter_adsr['attack'],
                                          0)
                else:
                    # If attack is zero, immediately jump to full level
                    filter_env = np.where(attack_mask, 1.0, 0)

                # Decay phase
                decay_start = config.filter_adsr['attack']
                decay_end = decay_start + config.filter_adsr['decay']
                decay_mask = (time_array >= decay_start) & (time_array < decay_end)
                if config.filter_adsr['decay'] > 0:
                    filter_env = np.where(
                        decay_mask,
                        1 - (1 - config.filter_adsr['sustain']) * ((time_array - decay_start) / config.filter_adsr['decay']),
                        filter_env
                    )
                else:
                    # If decay is zero, immediately jump to sustain level
                    filter_env = np.where(decay_mask, config.filter_adsr['sustain'], filter_env)

                # Sustain phase
                sustain_mask = time_array >= decay_end
                filter_env = np.where(sustain_mask,
                                    config.filter_adsr['sustain'],
                                    filter_env)
            else:
                # Release phase
                release_mask = time_array < config.filter_adsr['release']
                if config.filter_adsr['release'] > 0:
                    filter_env = np.where(
                        release_mask,
                        self.last_filter_env_level * (1 - time_array / config.filter_adsr['release']),
                        0
                    )
                else:
                    # If release is zero, immediately go to zero
                    filter_env = np.zeros_like(time_array)

            # Update filter envelope trackers
            self.filter_env_time += frames / config.sample_rate
            self.last_filter_env_level = filter_env[-1]

        # Apply LFO modulation to volume - only if needed
        if not np.all(lfo == 0) and config.lfo_target == 'volume':
            env = self.lfo.apply_amplitude_modulation(env, lfo)

        # Apply envelope and velocity to the wave
        output = wave * env * self.velocity

        return output, filter_env


# Counter for generating unique note keys
_scheduled_note_counter = 0


def _handle_scheduled_note_on(midi_note: int, velocity: float):
    """Handle a scheduled note_on event from the event scheduler.

    This function is called from the audio callback, so it must be thread-safe
    and efficient. It creates a new oscillator for the note.
    """
    global _scheduled_note_counter

    # Convert MIDI note to frequency
    freq = 440.0 * (2 ** ((midi_note - 69) / 12))

    with config.notes_lock:
        # Generate unique key for this note
        _scheduled_note_counter += 1
        key = f'scheduled_{_scheduled_note_counter}'

        # Create new oscillator
        osc = Oscillator(freq, config.waveform_type)
        osc.key = key
        osc.velocity = velocity

        # Add to active notes
        config.active_notes[key] = osc


def _handle_scheduled_note_off(midi_note: int):
    """Handle a scheduled note_off event from the event scheduler.

    This function is called from the audio callback. It finds and releases
    oscillators that were created by the scheduler (not keyboard notes).
    """
    # Convert MIDI note to frequency for matching
    target_freq = 440.0 * (2 ** ((midi_note - 69) / 12))

    with config.notes_lock:
        # Find all SCHEDULED oscillators playing this note and release them
        # Only affect notes with 'scheduled_' prefix to avoid releasing keyboard notes
        for key, osc in list(config.active_notes.items()):
            # Only release scheduled notes, not keyboard or programmatic notes
            if key.startswith('scheduled_'):
                # Check if this oscillator is playing the target frequency
                # Use a small tolerance for floating point comparison
                if abs(osc.freq - target_freq) < 0.01:
                    if not osc.released:
                        osc.released = True
                        osc.env_time = 0.0
                        osc.lfo_env_time = 0.0


def audio_callback(outdata, frames, time_info, status):
    """Audio callback function for the sounddevice output stream."""
    if status:
        print(f"Audio callback status: {status}")

    # Process scheduled events from the event scheduler
    scheduled_events = global_scheduler.process_events(frames)
    for frame_offset, event in scheduled_events:
        if event.event_type == 'note_on':
            _handle_scheduled_note_on(event.midi_note, event.velocity)
        elif event.event_type == 'note_off':
            _handle_scheduled_note_off(event.midi_note)
        elif event.event_type == 'callback' and event.callback:
            event.callback(*event.callback_args)

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
        if config.filter_cutoff < config.sample_rate / 2:
            buffer = filter.apply_filter(buffer, lfo_cutoff, filter_env_buffer)

        # Optional safety clip to prevent extreme peaks after drive
        np.clip(buffer, -1.2, 1.2, out=buffer)

    # Apply drive effect (wave folding/soft clipping) after filter - always
    buffer = apply_drive(buffer)

    # Start with mono signal
    buffer_L = buffer.copy()  # Use copy to prevent unintended shared references
    buffer_R = buffer.copy()

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
            buffer_L = mono_delayed.copy()  # Use copy to prevent shared references
            buffer_R = mono_delayed.copy()

    # Apply reverb effect if enabled - last in the effects chain
    if config.reverb_enabled:
        buffer_L, buffer_R = reverb.process(buffer_L, buffer_R)

    # Apply volume and output to the audio device
    outdata[:, 0] = config.volume * buffer_L
    outdata[:, 1] = config.volume * buffer_R

    # Add audio to recording buffer if recording is enabled
    if record.is_recording():
        # Create a copy of the stereo output for recording
        stereo_frame = np.column_stack((outdata[:, 0], outdata[:, 1]))
        record.add_audio_block(stereo_frame)

    # Update visualization buffer - use mono version for simplicity
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


def start_audio(latency='high'):
    """Start the global audio stream for simple programmatic use.

    Args:
        latency: 'high' for stability or 'low' for reduced latency

    Returns:
        bool: True if audio started successfully, False otherwise
    """
    global _global_audio_stream

    if _global_audio_stream is not None:
        # Audio already running
        return True

    try:
        _global_audio_stream = create_audio_stream(latency)
        _global_audio_stream.start()
        return True
    except Exception as e:
        print(f"Failed to start audio: {e}")
        _global_audio_stream = None
        return False


def stop_audio():
    """Stop the global audio stream."""
    global _global_audio_stream

    if _global_audio_stream is not None:
        try:
            _global_audio_stream.stop()
            _global_audio_stream.close()
        except Exception as e:
            print(f"Error stopping audio: {e}")
        finally:
            _global_audio_stream = None


def is_audio_running():
    """Check if the global audio stream is running.

    Returns:
        bool: True if audio is running, False otherwise
    """
    global _global_audio_stream
    return _global_audio_stream is not None and _global_audio_stream.active


class SynthContext:
    """Context manager for easy synth usage.

    Example:
        with SynthContext():
            controller.play_midi_note(60, duration=1.0)
            time.sleep(1.2)
    """

    def __init__(self, latency='high'):
        self.latency = latency

    def __enter__(self):
        if not start_audio(self.latency):
            raise RuntimeError("Failed to start audio")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_audio()
