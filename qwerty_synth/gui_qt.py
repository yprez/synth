"""PyQt GUI for QWERTY Synth providing controls for waveform and ADSR."""

import sys
import time
import signal
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QRadioButton, QPushButton, QGroupBox, QGridLayout
)
import pyqtgraph as pg

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth
from qwerty_synth import input as kb_input
from qwerty_synth import filter

# Global variable to hold reference to the GUI instance
gui_instance = None

class SynthGUI(QMainWindow):
    """GUI for controlling QWERTY Synth parameters using PyQt."""

    def __init__(self):
        """Initialize the GUI with PyQt5."""
        super().__init__()

        self.setWindowTitle("QWERTY Synth")
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height

        # Animation control variables
        self.animation_running = True

        # Keep track of the last GUI update time
        self.last_update_time = time.time()

        # Set up the user interface
        self.setup_ui()
        self.running = True

        # Start animation timer after UI is set up
        self.start_animation()

        # Handle window close event
        self.show()

    def setup_ui(self):
        """Set up the user interface components."""
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Waveform selection and Volume control in the same row
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        # Waveform selection
        wave_group = QGroupBox("Waveform")
        wave_layout = QHBoxLayout(wave_group)
        control_layout.addWidget(wave_group, stretch=1)

        self.waveform_buttons = []
        waveforms = [
            ("Sine", "sine"),
            ("Square", "square"),
            ("Triangle", "triangle"),
            ("Sawtooth", "sawtooth"),
        ]

        for text, value in waveforms:
            radio = QRadioButton(text)
            radio.setProperty("value", value)
            radio.toggled.connect(self.update_waveform)
            wave_layout.addWidget(radio)
            self.waveform_buttons.append(radio)
            if value == config.waveform_type:
                radio.setChecked(True)

        # Volume Control
        volume_group = QGroupBox("Volume Control")
        volume_layout = QHBoxLayout(volume_group)
        control_layout.addWidget(volume_group, stretch=1)

        volume_layout.addWidget(QLabel("Volume"))

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(config.volume * 100))
        self.volume_slider.valueChanged.connect(self.update_volume)
        volume_layout.addWidget(self.volume_slider, stretch=1)

        self.volume_label = QLabel(f"{config.volume:.2f}")
        volume_layout.addWidget(self.volume_label)

        # Filter Control
        filter_group = QGroupBox("Filter Control")
        filter_layout = QHBoxLayout(filter_group)
        main_layout.addWidget(filter_group)

        filter_layout.addWidget(QLabel("Cutoff Frequency"))

        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(100, 10000)
        self.cutoff_slider.setValue(int(filter.cutoff))
        self.cutoff_slider.valueChanged.connect(self.update_filter_cutoff)
        filter_layout.addWidget(self.cutoff_slider, stretch=1)

        self.cutoff_label = QLabel(f"{filter.cutoff:.0f} Hz")
        filter_layout.addWidget(self.cutoff_label)

        # Create a layout for visualization frames
        viz_layout = QHBoxLayout()
        main_layout.addLayout(viz_layout, stretch=1)

        # Waveform visualization
        wave_viz_group = QGroupBox("Waveform Display")
        wave_viz_layout = QVBoxLayout(wave_viz_group)
        viz_layout.addWidget(wave_viz_group)

        # Set up pyqtgraph plot for waveform
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setLabel('left', 'Amplitude')
        self.wave_plot.setLabel('bottom', 'Samples')
        self.wave_plot.setTitle('Synth Output Waveform')
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        wave_viz_layout.addWidget(self.wave_plot)

        # Initialize waveform plot
        self.window_size = 512

        # Add legend before creating plot items
        self.wave_plot.addLegend()

        self.wave_curve = self.wave_plot.plot(
            np.zeros(self.window_size),
            pen=pg.mkPen('b', width=2),
            name='Filtered'
        )
        self.unfiltered_curve = self.wave_plot.plot(
            np.zeros(self.window_size),
            pen=pg.mkPen('r', width=1, style=Qt.DashLine),
            name='Unfiltered'
        )
        self.wave_plot.setYRange(-1, 1)
        self.wave_plot.setXRange(0, 500)

        # Spectrum visualization
        spec_viz_group = QGroupBox("Frequency Spectrum")
        spec_viz_layout = QVBoxLayout(spec_viz_group)
        viz_layout.addWidget(spec_viz_group)

        # Set up pyqtgraph plot for spectrum
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setLabel('left', 'Magnitude')
        self.spec_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spec_plot.setTitle('Frequency Spectrum')
        self.spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spec_plot.setLogMode(x=False, y=False)
        spec_viz_layout.addWidget(self.spec_plot)

        # Initialize spectrum plot
        self.fft_size = 2048
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1 / config.sample_rate)
        self.spec_curve = self.spec_plot.plot(
            self.freqs,
            np.zeros_like(self.freqs),
            pen=pg.mkPen('g', width=2)
        )
        self.spec_plot.setXRange(50, 5000)
        self.spec_plot.setYRange(0, 0.25)

        # Create a frame to hold ADSR controls and visualization side by side
        adsr_layout = QHBoxLayout()
        main_layout.addLayout(adsr_layout, stretch=1)

        # ADSR controls
        adsr_group = QGroupBox("ADSR Envelope")
        adsr_controls = QGridLayout(adsr_group)
        adsr_layout.addWidget(adsr_group)

        # Attack slider
        adsr_controls.addWidget(QLabel("Attack"), 0, 0)
        self.attack_slider = QSlider(Qt.Horizontal)
        self.attack_slider.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.attack_slider.setValue(int(adsr.adsr['attack'] * 100))
        self.attack_slider.valueChanged.connect(
            lambda v: self.update_adsr('attack', v/100.0)
        )
        adsr_controls.addWidget(self.attack_slider, 0, 1)
        self.attack_label = QLabel(f"{adsr.adsr['attack']:.2f} s")
        adsr_controls.addWidget(self.attack_label, 0, 2)

        # Decay slider
        adsr_controls.addWidget(QLabel("Decay"), 1, 0)
        self.decay_slider = QSlider(Qt.Horizontal)
        self.decay_slider.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.decay_slider.setValue(int(adsr.adsr['decay'] * 100))
        self.decay_slider.valueChanged.connect(
            lambda v: self.update_adsr('decay', v/100.0)
        )
        adsr_controls.addWidget(self.decay_slider, 1, 1)
        self.decay_label = QLabel(f"{adsr.adsr['decay']:.2f} s")
        adsr_controls.addWidget(self.decay_label, 1, 2)

        # Sustain slider
        adsr_controls.addWidget(QLabel("Sustain"), 2, 0)
        self.sustain_slider = QSlider(Qt.Horizontal)
        self.sustain_slider.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.sustain_slider.setValue(int(adsr.adsr['sustain'] * 100))
        self.sustain_slider.valueChanged.connect(
            lambda v: self.update_adsr('sustain', v/100.0)
        )
        adsr_controls.addWidget(self.sustain_slider, 2, 1)
        self.sustain_label = QLabel(f"{adsr.adsr['sustain']:.2f}")
        adsr_controls.addWidget(self.sustain_label, 2, 2)

        # Release slider
        adsr_controls.addWidget(QLabel("Release"), 3, 0)
        self.release_slider = QSlider(Qt.Horizontal)
        self.release_slider.setRange(1, 300)  # 0.01 to 3.0 seconds (x100)
        self.release_slider.setValue(int(adsr.adsr['release'] * 100))
        self.release_slider.valueChanged.connect(
            lambda v: self.update_adsr('release', v/100.0)
        )
        adsr_controls.addWidget(self.release_slider, 3, 1)
        self.release_label = QLabel(f"{adsr.adsr['release']:.2f} s")
        adsr_controls.addWidget(self.release_label, 3, 2)

        # ADSR visualization
        adsr_viz_group = QGroupBox("ADSR Curve")
        adsr_viz_layout = QVBoxLayout(adsr_viz_group)
        adsr_layout.addWidget(adsr_viz_group)

        # Set up pyqtgraph plot for ADSR curve
        self.adsr_plot = pg.PlotWidget()
        self.adsr_plot.setLabel('left', 'Amplitude')
        self.adsr_plot.setLabel('bottom', 'Time (normalized)')
        self.adsr_plot.showGrid(x=True, y=True, alpha=0.3)
        adsr_viz_layout.addWidget(self.adsr_plot)

        # Initialize ADSR curve plot
        self.adsr_curve = self.adsr_plot.plot(
            np.arange(len(adsr.adsr_curve)),
            adsr.adsr_curve,
            pen=pg.mkPen('b', width=2)
        )
        self.adsr_plot.setYRange(0, 1.1)
        self.adsr_plot.setXRange(0, len(adsr.adsr_curve))

        # Instructions and Exit button
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # Instructions
        instruction_text = (
            "Play notes using your keyboard (A-K, W,E,T,Y,U,O,P).\n"
            "Z/X: Change octave down/up.\n"
            "1-4: Change waveform (sine, square, triangle, sawtooth).\n"
            "5-0, -, =: Adjust ADSR parameters.\n"
            "[/]: Decrease/Increase volume."
        )
        instructions = QLabel(instruction_text)
        bottom_layout.addWidget(instructions, alignment=Qt.AlignLeft)

        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        bottom_layout.addWidget(exit_button, alignment=Qt.AlignRight)

    def start_animation(self):
        """Start the QTimer for updating plots."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)  # Update every 30ms

    def update_plots(self):
        """Update function for waveform and spectrum plots animation."""
        if not self.animation_running:
            return

        # Check if waveform type has changed and update GUI if needed
        for button in self.waveform_buttons:
            if button.property("value") == config.waveform_type and not button.isChecked():
                button.setChecked(True)
                break

        # Check if filter cutoff has changed and update GUI if needed
        if self.cutoff_slider.value() != filter.cutoff:
            self.cutoff_slider.setValue(int(filter.cutoff))
            self.cutoff_label.setText(f"{filter.cutoff:.0f} Hz")

        # Check if ADSR parameters have changed and update GUI if needed
        adsr_changed = False

        if self.attack_slider.value() != int(adsr.adsr['attack'] * 100):
            self.attack_slider.setValue(int(adsr.adsr['attack'] * 100))
            self.attack_label.setText(f"{adsr.adsr['attack']:.2f} s")
            adsr_changed = True

        if self.decay_slider.value() != int(adsr.adsr['decay'] * 100):
            self.decay_slider.setValue(int(adsr.adsr['decay'] * 100))
            self.decay_label.setText(f"{adsr.adsr['decay']:.2f} s")
            adsr_changed = True

        if self.sustain_slider.value() != int(adsr.adsr['sustain'] * 100):
            self.sustain_slider.setValue(int(adsr.adsr['sustain'] * 100))
            self.sustain_label.setText(f"{adsr.adsr['sustain']:.2f}")
            adsr_changed = True

        if self.release_slider.value() != int(adsr.adsr['release'] * 100):
            self.release_slider.setValue(int(adsr.adsr['release'] * 100))
            self.release_label.setText(f"{adsr.adsr['release']:.2f} s")
            adsr_changed = True

        # Check if volume has changed
        if self.volume_slider.value() != int(config.volume * 100):
            self.volume_slider.setValue(int(config.volume * 100))
            self.volume_label.setText(f"{config.volume:.2f}")

        # Update ADSR curve if any parameters changed
        if adsr_changed:
            self.plot_adsr_curve()

        # Get current audio buffer data
        with config.buffer_lock:
            filtered_data = config.waveform_buffer.copy()
            unfiltered_data = config.unfiltered_buffer.copy()

        if len(filtered_data) == 0:
            return

        # Update waveform plot
        # Find zero crossing for clean waveform display
        for i in range(len(filtered_data) - 1):
            if filtered_data[i] < 0 <= filtered_data[i + 1]:
                start = i
                break
        else:
            start = 0

        end = start + self.window_size
        if end > len(filtered_data):
            start = max(0, len(filtered_data) - self.window_size)
            end = len(filtered_data)

        filtered_segment = filtered_data[start:end]
        if len(filtered_segment) < self.window_size:
            filtered_segment = np.pad(filtered_segment,
                                      (0, self.window_size - len(filtered_segment)))

        unfiltered_segment = unfiltered_data[start:end]
        if len(unfiltered_segment) < self.window_size:
            unfiltered_segment = np.pad(unfiltered_segment,
                                        (0, self.window_size - len(unfiltered_segment)))

        self.wave_curve.setData(np.arange(len(filtered_segment)), filtered_segment)
        self.unfiltered_curve.setData(np.arange(len(unfiltered_segment)), unfiltered_segment)

        # Update spectrum plot
        fft_data = (
            filtered_data[-self.fft_size:] if len(filtered_data) >= self.fft_size else
            np.pad(filtered_data, (0, self.fft_size - len(filtered_data)))
        )
        fft_data = fft_data * np.hanning(self.fft_size)
        spectrum = np.abs(np.fft.rfft(fft_data)) / self.fft_size
        self.spec_curve.setData(self.freqs, spectrum)

    def plot_adsr_curve(self):
        """Update the ADSR curve in the plot."""
        self.adsr_curve.setData(
            np.arange(len(adsr.adsr_curve)),
            adsr.adsr_curve
        )

    def update_waveform(self):
        """Update the waveform type in the config."""
        # Find which radio button is checked
        for button in self.waveform_buttons:
            if button.isChecked():
                config.waveform_type = button.property("value")
                break

    def update_adsr(self, param, value):
        """Update the specified ADSR parameter."""
        adsr.adsr[param] = value
        adsr.update_adsr_curve()

        # Update the ADSR curve visualization
        self.plot_adsr_curve()

        # Update label
        if param == 'attack':
            self.attack_label.setText(f"{value:.2f} s")
        elif param == 'decay':
            self.decay_label.setText(f"{value:.2f} s")
        elif param == 'sustain':
            self.sustain_label.setText(f"{value:.2f}")
        elif param == 'release':
            self.release_label.setText(f"{value:.2f} s")

    def update_volume(self, value):
        """Update the master volume."""
        volume = value / 100.0
        config.volume = volume
        self.volume_label.setText(f"{volume:.2f}")

    def update_filter_cutoff(self, value):
        """Update the low-pass filter cutoff frequency."""
        cutoff = float(value)
        filter.cutoff = cutoff
        self.cutoff_label.setText(f"{cutoff:.0f} Hz")

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Set animation state to stopped
            self.animation_running = False

            # Stop the animation timer if it exists
            if hasattr(self, 'timer') and self.timer is not None:
                self.timer.stop()

            # Set running flag to False
            self.running = False

            # Accept the close event
            if event is not None:
                event.accept()
        except Exception as e:
            print(f"Error during close: {e}")


def start_gui():
    """Start the GUI and synth components."""
    app = QApplication(sys.argv)

    # Apply white background theme for charts
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    # Create GUI
    gui = SynthGUI()

    # Store global reference to GUI for signal handling
    global gui_instance
    gui_instance = gui

    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\nKeyboard interrupt detected, exiting...")
        if gui_instance is not None:
            gui_instance.close()
        app.quit()

    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Install event filter to process signals during Qt event loop
    timer = QTimer()
    timer.start(500)  # Check for signals every 500ms
    timer.timeout.connect(lambda: None)  # Wake up Python interpreter regularly

    # Share the GUI instance with the input module
    kb_input.gui_instance = gui

    # Create and start audio stream
    stream = synth.create_audio_stream()
    stream.start()

    # Start keyboard input handling in a separate thread
    kb_input.start_keyboard_input()

    # Start Qt event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        # This might still catch some KeyboardInterrupts
        print("Keyboard interrupt detected, exiting...")
        gui.close()
    finally:
        # Clean up
        stream.stop()
        stream.close()


if __name__ == "__main__":
    start_gui()
