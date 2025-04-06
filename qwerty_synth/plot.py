"""Visualization for waveform, frequency spectrum, and ADSR envelope."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from qwerty_synth import config
from qwerty_synth import adsr


def plot_waveform():
    """Create and display interactive plots for waveform, spectrum, and ADSR."""
    window = 512
    fft_size = 2048

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.tight_layout(pad=4.0)

    ax_wave, ax_fft, ax_env = axs

    # Waveform plot
    wave_line, = ax_wave.plot(np.zeros(window))
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_xlim(0, window)
    ax_wave.set_title('Synth Output Waveform')
    ax_wave.set_xlabel('Samples')
    ax_wave.set_ylabel('Amplitude')

    # Frequency spectrum plot
    freqs = np.fft.rfftfreq(fft_size, d=1 / config.sample_rate)
    fft_line, = ax_fft.semilogx(freqs, np.zeros_like(freqs))
    ax_fft.set_xlim(20, config.sample_rate / 2)
    ax_fft.set_ylim(0, 1)
    ax_fft.set_title('Frequency Spectrum')
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Magnitude')

    # ADSR envelope plot
    env_line, = ax_env.plot(adsr.adsr_curve)
    ax_env.set_ylim(0, 1.1)
    ax_env.set_xlim(0, len(adsr.adsr_curve))
    ax_env.set_title('ADSR Envelope Curve')
    ax_env.set_xlabel('Time (normalized)')
    ax_env.set_ylabel('Amplitude')

    def update(_):
        """Update function for animation."""
        with config.buffer_lock:
            data = config.waveform_buffer.copy()

        # Find zero crossing to start display at
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

        # Update FFT display
        fft_data = data[-fft_size:] * np.hanning(fft_size)
        spectrum = np.abs(np.fft.rfft(fft_data)) / fft_size
        fft_line.set_ydata(spectrum)

        # Update ADSR curve
        env_line.set_ydata(adsr.adsr_curve)
        return wave_line, fft_line, env_line

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
    plt.show()


def init_visualization():
    """Initialize and prepare visualization."""
    adsr.update_adsr_curve()
