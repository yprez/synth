"""Tkinter GUI for QWERTY Synth providing controls for waveform and ADSR."""

import tkinter as tk
from tkinter import ttk
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth
from qwerty_synth import input as kb_input


class SynthGUI:
    """GUI for controlling QWERTY Synth parameters."""

    def __init__(self, root):
        """Initialize the GUI with the root Tkinter window."""
        self.root = root
        self.root.title("QWERTY Synth")
        self.root.geometry("1200x800")  # Increased size to accommodate all plots
        self.root.resizable(True, True)

        # Animation control variables
        self.animation_running = True
        self.ani = None  # Will hold the animation object

        # Keep track of the last GUI update time
        self.last_update_time = time.time()

        self.setup_ui()
        self.running = True

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start animation after UI is set up
        self.start_animation()

    def setup_ui(self):
        """Set up the user interface components."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Waveform selection and Volume control in the same row
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        # Waveform selection
        wave_frame = ttk.LabelFrame(control_frame, text="Waveform", padding="10")
        wave_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.waveform_var = tk.StringVar(value=config.waveform_type)
        waveforms = [
            ("Sine", "sine"),
            ("Square", "square"),
            ("Triangle", "triangle"),
            ("Sawtooth", "sawtooth"),
        ]

        for i, (text, value) in enumerate(waveforms):
            ttk.Radiobutton(
                wave_frame,
                text=text,
                value=value,
                variable=self.waveform_var,
                command=self.update_waveform
            ).grid(row=0, column=i, padx=20)

        # Volume Control
        volume_frame = ttk.LabelFrame(control_frame, text="Volume Control", padding="10")
        volume_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Label(volume_frame, text="Volume").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.volume_var = tk.DoubleVar(value=config.volume)
        volume_slider = ttk.Scale(
            volume_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            command=self.update_volume
        )
        volume_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        volume_frame.columnconfigure(1, weight=1)
        self.volume_label = ttk.Label(volume_frame, text=f"{config.volume:.2f}")
        self.volume_label.grid(row=0, column=2, padx=5, pady=5)

        # Create visualization frames
        # A container for waveform and spectrum visualization
        viz_container = ttk.Frame(main_frame)
        viz_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Waveform visualization
        wave_viz_frame = ttk.LabelFrame(viz_container, text="Waveform Display", padding="10")
        wave_viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.wave_fig = Figure(figsize=(4, 3), dpi=100)
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=wave_viz_frame)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize waveform plot
        self.window_size = 512
        self.wave_line, = self.wave_ax.plot(np.zeros(self.window_size))
        self.wave_ax.set_ylim(-1, 1)
        self.wave_ax.set_xlim(0, self.window_size)
        self.wave_ax.set_title('Synth Output Waveform')
        self.wave_ax.set_xlabel('Samples')
        self.wave_ax.set_ylabel('Amplitude')
        self.wave_ax.grid(True, linestyle='--', alpha=0.7)

        # Spectrum visualization
        spec_viz_frame = ttk.LabelFrame(viz_container, text="Frequency Spectrum", padding="10")
        spec_viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.spec_fig = Figure(figsize=(4, 3), dpi=100)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=spec_viz_frame)
        self.spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize spectrum plot
        self.fft_size = 2048
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1 / config.sample_rate)
        self.spec_line, = self.spec_ax.semilogx(self.freqs, np.zeros_like(self.freqs))
        self.spec_ax.set_xlim(20, config.sample_rate / 2)
        self.spec_ax.set_ylim(0, 1)
        self.spec_ax.set_title('Frequency Spectrum')
        self.spec_ax.set_xlabel('Frequency (Hz)')
        self.spec_ax.set_ylabel('Magnitude')
        self.spec_ax.grid(True, linestyle='--', alpha=0.7)

        # Create a frame to hold ADSR controls and visualization side by side
        adsr_container = ttk.Frame(main_frame)
        adsr_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # ADSR controls
        adsr_frame = ttk.LabelFrame(adsr_container, text="ADSR Envelope", padding="10")
        adsr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Attack slider
        ttk.Label(adsr_frame, text="Attack").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.attack_var = tk.DoubleVar(value=adsr.adsr['attack'])
        ttk.Scale(
            adsr_frame,
            from_=0.01,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.attack_var,
            command=lambda v: self.update_adsr('attack', float(v))
        ).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.attack_label = ttk.Label(adsr_frame, text=f"{adsr.adsr['attack']:.2f} s")
        self.attack_label.grid(row=0, column=2, padx=5, pady=5)

        # Decay slider
        ttk.Label(adsr_frame, text="Decay").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.decay_var = tk.DoubleVar(value=adsr.adsr['decay'])
        ttk.Scale(
            adsr_frame,
            from_=0.01,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.decay_var,
            command=lambda v: self.update_adsr('decay', float(v))
        ).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.decay_label = ttk.Label(adsr_frame, text=f"{adsr.adsr['decay']:.2f} s")
        self.decay_label.grid(row=1, column=2, padx=5, pady=5)

        # Sustain slider
        ttk.Label(adsr_frame, text="Sustain").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.sustain_var = tk.DoubleVar(value=adsr.adsr['sustain'])
        ttk.Scale(
            adsr_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.sustain_var,
            command=lambda v: self.update_adsr('sustain', float(v))
        ).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.sustain_label = ttk.Label(adsr_frame, text=f"{adsr.adsr['sustain']:.2f}")
        self.sustain_label.grid(row=2, column=2, padx=5, pady=5)

        # Release slider
        ttk.Label(adsr_frame, text="Release").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.release_var = tk.DoubleVar(value=adsr.adsr['release'])
        ttk.Scale(
            adsr_frame,
            from_=0.01,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.release_var,
            command=lambda v: self.update_adsr('release', float(v))
        ).grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.release_label = ttk.Label(adsr_frame, text=f"{adsr.adsr['release']:.2f} s")
        self.release_label.grid(row=3, column=2, padx=5, pady=5)

        adsr_frame.columnconfigure(1, weight=1)

        # ADSR visualization
        viz_frame = ttk.LabelFrame(adsr_container, text="ADSR Curve", padding="10")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create the matplotlib figure and canvas
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial ADSR curve plot
        self.plot_adsr_curve()

        # Exit button
        exit_frame = ttk.Frame(main_frame)
        exit_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            exit_frame,
            text="Exit",
            command=self.on_closing
        ).pack(side=tk.RIGHT)

        # Instructions
        instruction_text = (
            "Play notes using your keyboard (A-K, W,E,T,Y,U,O,P).\n"
            "Z/X: Change octave down/up.\n"
            "1-4: Change waveform (sine, square, triangle, sawtooth).\n"
            "5-0, -, =: Adjust ADSR parameters.\n"
            "[/]: Decrease/Increase volume."
        )
        ttk.Label(main_frame, text=instruction_text).pack(anchor=tk.W, pady=10)

    def start_animation(self):
        """Start the animation for waveform and spectrum plots."""
        self.ani = animation.FuncAnimation(
            self.wave_fig,
            self.update_plots,
            interval=30,
            blit=True,
            cache_frame_data=False
        )

    def update_plots(self, _):
        """Update function for waveform and spectrum plots animation."""
        if not self.animation_running:
            return self.wave_line, self.spec_line

        # Check if waveform type has changed and update GUI if needed
        if self.waveform_var.get() != config.waveform_type:
            self.waveform_var.set(config.waveform_type)

        # Check if ADSR parameters have changed and update GUI if needed
        adsr_changed = False
        if self.attack_var.get() != adsr.adsr['attack']:
            self.attack_var.set(adsr.adsr['attack'])
            self.attack_label.config(text=f"{adsr.adsr['attack']:.2f} s")
            adsr_changed = True

        if self.decay_var.get() != adsr.adsr['decay']:
            self.decay_var.set(adsr.adsr['decay'])
            self.decay_label.config(text=f"{adsr.adsr['decay']:.2f} s")
            adsr_changed = True

        if self.sustain_var.get() != adsr.adsr['sustain']:
            self.sustain_var.set(adsr.adsr['sustain'])
            self.sustain_label.config(text=f"{adsr.adsr['sustain']:.2f}")
            adsr_changed = True

        if self.release_var.get() != adsr.adsr['release']:
            self.release_var.set(adsr.adsr['release'])
            self.release_label.config(text=f"{adsr.adsr['release']:.2f} s")
            adsr_changed = True

        # Check if volume has changed
        if self.volume_var.get() != config.volume:
            self.volume_var.set(config.volume)
            self.volume_label.config(text=f"{config.volume:.2f}")

        # Update ADSR curve if any parameters changed
        if adsr_changed:
            self.plot_adsr_curve()

        # Get current audio buffer data
        with config.buffer_lock:
            data = config.waveform_buffer.copy()

        if len(data) == 0:
            return self.wave_line, self.spec_line

        # Update waveform plot
        # Find zero crossing for clean waveform display
        for i in range(len(data) - 1):
            if data[i] < 0 <= data[i + 1]:
                start = i
                break
        else:
            start = 0

        end = start + self.window_size
        if end > len(data):
            start = max(0, len(data) - self.window_size)
            end = len(data)

        segment = data[start:end]
        if len(segment) < self.window_size:
            segment = np.pad(segment, (0, self.window_size - len(segment)))

        self.wave_line.set_ydata(segment)

        # Update spectrum plot
        fft_data = data[-self.fft_size:] if len(data) >= self.fft_size else np.pad(data, (0, self.fft_size - len(data)))
        fft_data = fft_data * np.hanning(self.fft_size)
        spectrum = np.abs(np.fft.rfft(fft_data)) / self.fft_size
        self.spec_line.set_ydata(spectrum)

        # Draw canvases
        self.wave_canvas.draw_idle()
        self.spec_canvas.draw_idle()

        return self.wave_line, self.spec_line

    def plot_adsr_curve(self):
        """Plot the ADSR curve in the matplotlib figure."""
        self.ax.clear()
        x = np.arange(len(adsr.adsr_curve))
        self.ax.plot(x, adsr.adsr_curve, 'b-')
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xlim(0, len(adsr.adsr_curve))
        self.ax.set_xlabel('Time (normalized)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_waveform(self):
        """Update the waveform type in the config."""
        config.waveform_type = self.waveform_var.get()

    def update_adsr(self, param, value):
        """Update the specified ADSR parameter."""
        adsr.adsr[param] = value
        adsr.update_adsr_curve()

        # Update the ADSR curve visualization
        self.plot_adsr_curve()

        # Update label
        if param == 'attack':
            self.attack_label.config(text=f"{value:.2f} s")
        elif param == 'decay':
            self.decay_label.config(text=f"{value:.2f} s")
        elif param == 'sustain':
            self.sustain_label.config(text=f"{value:.2f}")
        elif param == 'release':
            self.release_label.config(text=f"{value:.2f} s")

    def update_volume(self, value):
        """Update the master volume."""
        volume = float(value)
        config.volume = volume
        self.volume_label.config(text=f"{volume:.2f}")

    def on_closing(self):
        """Handle window close event."""
        self.animation_running = False  # Stop the animation
        if self.ani is not None:
            self.ani.event_source.stop()  # Stop the animation event source
        self.running = False
        self.root.destroy()


def start_gui():
    """Start the GUI and synth components."""
    root = tk.Tk()
    app = SynthGUI(root)

    # Share the GUI instance with the input module
    kb_input.gui_instance = app

    # Create and start audio stream
    stream = synth.create_audio_stream()
    stream.start()

    # Start keyboard input handling in a separate thread
    kb_input.start_keyboard_input()

    # Start Tkinter event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C by properly closing the GUI
        print("Keyboard interrupt detected, exiting...")
        app.on_closing()
    finally:
        # Clean up
        stream.stop()
        stream.close()
