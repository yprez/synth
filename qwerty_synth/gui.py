"""Tkinter GUI for QWERTY Synth providing controls for waveform and ADSR."""

import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.setup_ui()
        self.running = True

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Set up the user interface components."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Waveform selection
        wave_frame = ttk.LabelFrame(main_frame, text="Waveform", padding="10")
        wave_frame.pack(fill=tk.X, pady=10)

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
            "Z/X: Change octave down/up."
        )
        ttk.Label(main_frame, text=instruction_text).pack(anchor=tk.W, pady=10)

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

    def on_closing(self):
        """Handle window close event."""
        self.running = False
        self.root.destroy()


def start_gui():
    """Start the GUI and synth components."""
    root = tk.Tk()
    app = SynthGUI(root)

    # Create and start audio stream
    stream = synth.create_audio_stream()
    stream.start()

    # Start keyboard input handling in a separate thread
    input_thread = threading.Thread(target=kb_input.start_keyboard_input)
    input_thread.daemon = True
    input_thread.start()

    # Start Tkinter event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        stream.stop()
        stream.close()


if __name__ == "__main__":
    start_gui()
