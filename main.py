import numpy as np
import sounddevice as sd
from pynput import keyboard
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sample_rate = 44100
volume = 0.3
fade_duration = 0.01

visual_buffer_size = int(sample_rate * 0.1)
waveform_buffer = np.zeros(visual_buffer_size)
buffer_lock = threading.Lock()

adsr = {
    'attack': 0.05,
    'decay': 0.1,
    'sustain': 0.7,
    'release': 0.3
}

adsr_curve = np.zeros(512)

key_note_map = {
    'a': 261.63, 'w': 277.18, 's': 293.66, 'e': 311.13,
    'd': 329.63, 'f': 349.23, 't': 369.99, 'g': 392.00,
    'y': 415.30, 'h': 440.00, 'u': 466.16, 'j': 493.88,
    'k': 523.25, 'o': 554.37, 'l': 587.33, 'p': 622.25,
    ';': 659.25, "'": 698.46
}

active_notes = {}
notes_lock = threading.Lock()

octave_offset = 0
octave_min = -2
octave_max = 3

waveform_type = 'sine'


class Oscillator:
    def __init__(self, freq, waveform):
        self.freq = freq
        self.waveform = waveform
        self.phase = 0.0
        self.done = False
        self.env_time = 0.0
        self.released = False
        self.last_env_level = 0.0

    def generate(self, frames):
        if self.done:
            return np.zeros(frames)

        t = np.arange(frames) / sample_rate
        phase_increment = 2 * np.pi * self.freq / sample_rate
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
            time = self.env_time + i / sample_rate
            if not self.released:
                if time < adsr['attack']:
                    env[i] = (time / adsr['attack'])
                elif time < adsr['attack'] + adsr['decay']:
                    dt = time - adsr['attack']
                    env[i] = 1 - (1 - adsr['sustain']) * (dt / adsr['decay'])
                else:
                    env[i] = adsr['sustain']
            else:
                if time < adsr['release']:
                    env[i] = self.last_env_level * (1 - time / adsr['release'])
                else:
                    env[i] = 0.0
                    self.done = True

        self.env_time += frames / sample_rate
        self.last_env_level = env[-1]
        wave *= env

        return wave


def audio_callback(outdata, frames, time_info, status):
    buffer = np.zeros(frames)

    with notes_lock:
        finished_keys = []

        for key, osc in active_notes.items():
            wave = osc.generate(frames)
            buffer += wave
            if osc.done:
                finished_keys.append(key)

        for key in finished_keys:
            del active_notes[key]

    if len(active_notes) > 0:
        buffer /= len(active_notes)

    outdata[:] = (volume * buffer).reshape(-1, 1)

    with buffer_lock:
        global waveform_buffer
        waveform_buffer = np.roll(waveform_buffer, -frames)
        waveform_buffer[-frames:] = buffer


def on_press(key):
    global octave_offset, waveform_type

    try:
        k = key.char.lower()

        if k in key_note_map:
            base_freq = key_note_map[k]
            freq = base_freq * (2 ** (octave_offset / 12))
            with notes_lock:
                if k not in active_notes:
                    osc = Oscillator(freq, waveform_type)
                    active_notes[k] = osc

        elif k == 'z' and octave_offset > 12 * octave_min:
            octave_offset -= 12
            print(f'Octave down: {octave_offset // 12:+}')
        elif k == 'x' and octave_offset < 12 * octave_max:
            octave_offset += 12
            print(f'Octave up: {octave_offset // 12:+}')

        elif k == '1':
            waveform_type = 'sine'
            print('Waveform: sine')
        elif k == '2':
            waveform_type = 'square'
            print('Waveform: square')
        elif k == '3':
            waveform_type = 'triangle'
            print('Waveform: triangle')
        elif k == '4':
            waveform_type = 'sawtooth'
            print('Waveform: sawtooth')

        step = {
            'attack': 0.01,
            'decay': 0.1,
            'sustain': 0.1,
            'release': 0.1
        }

        if k == '5':
            adsr['attack'] += step['attack']
        elif k == '6':
            adsr['attack'] = max(0.001, adsr['attack'] - step['attack'])
        elif k == '7':
            adsr['decay'] += step['decay']
        elif k == '8':
            adsr['decay'] = max(0.01, adsr['decay'] - step['decay'])
        elif k == '9':
            adsr['sustain'] = min(1.0, adsr['sustain'] + step['sustain'])
        elif k == '0':
            adsr['sustain'] = max(0.0, adsr['sustain'] - step['sustain'])
        elif k == '-':
            adsr['release'] += step['release']
        elif k == '=':
            adsr['release'] = max(0.01, adsr['release'] - step['release'])

        update_adsr_curve()

    except AttributeError:
        if key == keyboard.Key.esc:
            print('Exiting...')
            sd.stop()
            plt.close('all')
            return False


def update_adsr_curve():
    global adsr_curve
    total_time = adsr['attack'] + adsr['decay'] + adsr['release']
    t = np.linspace(0, total_time, len(adsr_curve))
    curve = np.zeros_like(t)

    for i, time in enumerate(t):
        if time < adsr['attack']:
            curve[i] = time / adsr['attack']
        elif time < adsr['attack'] + adsr['decay']:
            dt = time - adsr['attack']
            curve[i] = 1 - (1 - adsr['sustain']) * (dt / adsr['decay'])
        elif time < total_time:
            rt = time - (adsr['attack'] + adsr['decay'])
            curve[i] = adsr['sustain'] * (1 - rt / adsr['release'])
        else:
            curve[i] = 0

    adsr_curve = curve


def on_release(key):
    try:
        k = key.char.lower()
        with notes_lock:
            if k in active_notes:
                active_notes[k].released = True
                active_notes[k].env_time = 0.0
    except AttributeError:
        pass


def run_keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def plot_waveform():
    window = 512
    fft_size = 2048

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.tight_layout(pad=4.0)

    ax_wave, ax_fft, ax_env = axs

    wave_line, = ax_wave.plot(np.zeros(window))
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_xlim(0, window)
    ax_wave.set_title('Synth Output Waveform')
    ax_wave.set_xlabel('Samples')
    ax_wave.set_ylabel('Amplitude')

    freqs = np.fft.rfftfreq(fft_size, d=1 / sample_rate)
    fft_line, = ax_fft.semilogx(freqs, np.zeros_like(freqs))
    ax_fft.set_xlim(20, sample_rate / 2)
    ax_fft.set_ylim(0, 1)
    ax_fft.set_title('Frequency Spectrum')
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Magnitude')

    env_line, = ax_env.plot(adsr_curve)
    ax_env.set_ylim(0, 1.1)
    ax_env.set_xlim(0, len(adsr_curve))
    ax_env.set_title('ADSR Envelope Curve')
    ax_env.set_xlabel('Time (normalized)')
    ax_env.set_ylabel('Amplitude')

    def update(_):
        with buffer_lock:
            data = waveform_buffer.copy()

        for i in range(len(data) - 1):
            if data[i] < 0 <= data[i + 1]:
                start = i
                break
        else:
            start = 0

        end = start + window
        if end > len(data):
            start = len(data) - window
            end = len(data)

        segment = data[start:end]
        if len(segment) < window:
            segment = np.pad(segment, (0, window - len(segment)))

        wave_line.set_ydata(segment)

        fft_data = data[-fft_size:] * np.hanning(fft_size)
        spectrum = np.abs(np.fft.rfft(fft_data)) / fft_size
        fft_line.set_ydata(spectrum)

        env_line.set_ydata(adsr_curve)
        return wave_line, fft_line, env_line

    update_adsr_curve()
    ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
    plt.show()


print('QWERTY Synth: Play keys A–K, W,E,T,Y,U,O,P etc.')
print('Use Z / X to shift octave down / up')
print('Use 1–4 to switch waveform: 1=sine, 2=square, 3=triangle, 4=sawtooth')
print('Use 5/6 for Attack, 7/8 for Decay, 9/0 for Sustain, -/= for Release')
print('Press ESC to quit')

stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=1024
)
stream.start()

threading.Thread(target=run_keyboard_listener, daemon=True).start()
plot_waveform()
