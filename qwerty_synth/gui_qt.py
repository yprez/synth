"""PyQt GUI for QWERTY Synth providing controls for waveform and ADSR."""

import sys
import time
import signal
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QRadioButton, QPushButton, QGroupBox, QGridLayout,
    QCheckBox, QDoubleSpinBox, QComboBox, QTabWidget, QSpinBox, QFrame
)
import pyqtgraph as pg

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth
from qwerty_synth import input as kb_input
from qwerty_synth import filter
from qwerty_synth import delay
from qwerty_synth.step_sequencer import StepSequencer

# Global variable to hold reference to the GUI instance
gui_instance = None

class SynthGUI(QMainWindow):
    """GUI for controlling QWERTY Synth parameters using PyQt."""

    def __init__(self):
        """Initialize the GUI with PyQt5."""
        super().__init__()

        self.setWindowTitle("QWERTY Synth")
        self.setGeometry(100, 100, 1200, 900)  # x, y, width, height - increased height for sequencer

        # Animation control variables
        self.animation_running = True

        # Keep track of the last GUI update time
        self.last_update_time = time.time()

        # Create sequencer instance
        self.sequencer = None  # Will be initialized in setup_ui

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

        # Octave Control
        octave_group = QGroupBox("Octave Control")
        octave_layout = QVBoxLayout(octave_group)
        control_layout.addWidget(octave_group, stretch=1)

        # Buttons and display in a horizontal layout
        octave_btn_layout = QHBoxLayout()
        octave_layout.addLayout(octave_btn_layout)

        # Decrease octave button
        self.octave_down_btn = QPushButton("−")  # Unicode minus sign
        self.octave_down_btn.setToolTip("Decrease octave (Z key)")
        self.octave_down_btn.clicked.connect(self.decrease_octave)
        octave_btn_layout.addWidget(self.octave_down_btn)

        # Octave display
        self.octave_label = QLabel(f"{config.octave_offset // 12:+d}")
        self.octave_label.setAlignment(Qt.AlignCenter)
        octave_btn_layout.addWidget(self.octave_label, stretch=1)

        # Increase octave button
        self.octave_up_btn = QPushButton("+")
        self.octave_up_btn.setToolTip("Increase octave (X key)")
        self.octave_up_btn.clicked.connect(self.increase_octave)
        octave_btn_layout.addWidget(self.octave_up_btn)

        # Octave slider in a second row
        self.octave_slider = QSlider(Qt.Horizontal)
        self.octave_slider.setRange(config.octave_min, config.octave_max)
        self.octave_slider.setValue(config.octave_offset // 12)
        self.octave_slider.setTickPosition(QSlider.TicksBelow)
        self.octave_slider.setTickInterval(1)
        self.octave_slider.valueChanged.connect(self.update_octave)
        octave_layout.addWidget(self.octave_slider)

        # Mono Mode and Portamento Controls
        mono_group = QGroupBox("Mono Mode")
        mono_layout = QHBoxLayout(mono_group)
        control_layout.addWidget(mono_group, stretch=1)

        # Mono mode checkbox
        self.mono_checkbox = QCheckBox("Mono Mode")
        self.mono_checkbox.setChecked(config.mono_mode)
        self.mono_checkbox.stateChanged.connect(self.update_mono_mode)
        mono_layout.addWidget(self.mono_checkbox)

        # Glide time slider
        mono_layout.addWidget(QLabel("Glide"))
        self.glide_slider = QSlider(Qt.Horizontal)
        self.glide_slider.setRange(1, 500)  # 1ms to 500ms (was 0-500)
        self.glide_slider.setValue(int(max(1, config.glide_time * 1000)))
        self.glide_slider.valueChanged.connect(self.update_glide_time)
        mono_layout.addWidget(self.glide_slider, stretch=1)

        self.glide_label = QLabel(f"{config.glide_time*1000:.0f} ms")
        mono_layout.addWidget(self.glide_label)

        # Filter Control
        filter_group = QGroupBox("Filter Control")
        filter_layout = QHBoxLayout(filter_group)
        main_layout.addWidget(filter_group)

        # Filter enable checkbox
        self.filter_enable_checkbox = QCheckBox("Enable Filter")
        self.filter_enable_checkbox.setChecked(filter.filter_enabled)
        self.filter_enable_checkbox.stateChanged.connect(self.update_filter_enabled)
        filter_layout.addWidget(self.filter_enable_checkbox)

        filter_layout.addWidget(QLabel("Cutoff Frequency"))

        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setRange(100, 10000)
        self.cutoff_slider.setValue(int(filter.cutoff))
        self.cutoff_slider.valueChanged.connect(self.update_filter_cutoff)
        filter_layout.addWidget(self.cutoff_slider, stretch=1)

        self.cutoff_label = QLabel(f"{filter.cutoff:.0f} Hz")
        filter_layout.addWidget(self.cutoff_label)

        # Add resonance control
        filter_layout.addWidget(QLabel("Resonance"))
        self.resonance_slider = QSlider(Qt.Horizontal)
        self.resonance_slider.setRange(0, 95)  # 0.0 to 0.95 (x100)
        self.resonance_slider.setValue(int(filter.resonance * 100))
        self.resonance_slider.valueChanged.connect(self.update_filter_resonance)
        filter_layout.addWidget(self.resonance_slider, stretch=1)

        self.resonance_label = QLabel(f"{filter.resonance:.2f}")
        filter_layout.addWidget(self.resonance_label)

        # Add filter envelope amount control
        filter_layout.addWidget(QLabel("Envelope Amount"))
        self.filter_env_amount_slider = QSlider(Qt.Horizontal)
        self.filter_env_amount_slider.setRange(0, 10000)
        self.filter_env_amount_slider.setValue(int(adsr.filter_env_amount))
        self.filter_env_amount_slider.valueChanged.connect(self.update_filter_env_amount)
        filter_layout.addWidget(self.filter_env_amount_slider, stretch=1)

        self.filter_env_amount_label = QLabel(f"{adsr.filter_env_amount:.0f} Hz")
        filter_layout.addWidget(self.filter_env_amount_label)

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

        # Create a tabbed widget for amp ADSR and filter ADSR
        envelope_tabs = QTabWidget()
        main_layout.addWidget(envelope_tabs, stretch=1)

        # Create the amplitude ADSR tab
        amp_env_widget = QWidget()
        amp_env_layout = QHBoxLayout(amp_env_widget)
        envelope_tabs.addTab(amp_env_widget, "Amplitude Envelope")

        # Create the filter ADSR tab
        filter_env_widget = QWidget()
        filter_env_layout = QHBoxLayout(filter_env_widget)
        envelope_tabs.addTab(filter_env_widget, "Filter Envelope")

        # Create the LFO tab
        lfo_tab_widget = QWidget()
        lfo_tab_layout = QHBoxLayout(lfo_tab_widget)
        envelope_tabs.addTab(lfo_tab_widget, "LFO Control")

        # Create the delay effect tab
        delay_tab_widget = QWidget()
        delay_tab_layout = QHBoxLayout(delay_tab_widget)
        envelope_tabs.addTab(delay_tab_widget, "Delay Effect")

        # Create the sequencer tab
        self.sequencer = StepSequencer()
        envelope_tabs.addTab(self.sequencer, "Step Sequencer")

        # Connect sequencer BPM to the global BPM
        if hasattr(self.sequencer, 'bpm_spinbox'):
            self.sequencer.bpm_spinbox.valueChanged.connect(self.sync_sequencer_bpm)

        # ADSR controls for amplitude
        adsr_group = QGroupBox("ADSR Envelope")
        adsr_controls = QGridLayout(adsr_group)
        amp_env_layout.addWidget(adsr_group)

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

        # ADSR visualization for amplitude
        adsr_viz_group = QGroupBox("ADSR Curve")
        adsr_viz_layout = QVBoxLayout(adsr_viz_group)
        amp_env_layout.addWidget(adsr_viz_group)

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

        # ADSR controls for filter envelope
        filter_adsr_group = QGroupBox("Filter ADSR Envelope")
        filter_adsr_controls = QGridLayout(filter_adsr_group)
        filter_env_layout.addWidget(filter_adsr_group)

        # Filter Attack slider
        filter_adsr_controls.addWidget(QLabel("Attack"), 0, 0)
        self.filter_attack_slider = QSlider(Qt.Horizontal)
        self.filter_attack_slider.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.filter_attack_slider.setValue(int(adsr.filter_adsr['attack'] * 100))
        self.filter_attack_slider.valueChanged.connect(
            lambda v: self.update_filter_adsr('attack', v/100.0)
        )
        filter_adsr_controls.addWidget(self.filter_attack_slider, 0, 1)
        self.filter_attack_label = QLabel(f"{adsr.filter_adsr['attack']:.2f} s")
        filter_adsr_controls.addWidget(self.filter_attack_label, 0, 2)

        # Filter Decay slider
        filter_adsr_controls.addWidget(QLabel("Decay"), 1, 0)
        self.filter_decay_slider = QSlider(Qt.Horizontal)
        self.filter_decay_slider.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.filter_decay_slider.setValue(int(adsr.filter_adsr['decay'] * 100))
        self.filter_decay_slider.valueChanged.connect(
            lambda v: self.update_filter_adsr('decay', v/100.0)
        )
        filter_adsr_controls.addWidget(self.filter_decay_slider, 1, 1)
        self.filter_decay_label = QLabel(f"{adsr.filter_adsr['decay']:.2f} s")
        filter_adsr_controls.addWidget(self.filter_decay_label, 1, 2)

        # Filter Sustain slider
        filter_adsr_controls.addWidget(QLabel("Sustain"), 2, 0)
        self.filter_sustain_slider = QSlider(Qt.Horizontal)
        self.filter_sustain_slider.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.filter_sustain_slider.setValue(int(adsr.filter_adsr['sustain'] * 100))
        self.filter_sustain_slider.valueChanged.connect(
            lambda v: self.update_filter_adsr('sustain', v/100.0)
        )
        filter_adsr_controls.addWidget(self.filter_sustain_slider, 2, 1)
        self.filter_sustain_label = QLabel(f"{adsr.filter_adsr['sustain']:.2f}")
        filter_adsr_controls.addWidget(self.filter_sustain_label, 2, 2)

        # Filter Release slider
        filter_adsr_controls.addWidget(QLabel("Release"), 3, 0)
        self.filter_release_slider = QSlider(Qt.Horizontal)
        self.filter_release_slider.setRange(1, 300)  # 0.01 to 3.0 seconds (x100)
        self.filter_release_slider.setValue(int(adsr.filter_adsr['release'] * 100))
        self.filter_release_slider.valueChanged.connect(
            lambda v: self.update_filter_adsr('release', v/100.0)
        )
        filter_adsr_controls.addWidget(self.filter_release_slider, 3, 1)
        self.filter_release_label = QLabel(f"{adsr.filter_adsr['release']:.2f} s")
        filter_adsr_controls.addWidget(self.filter_release_label, 3, 2)

        # Filter ADSR visualization
        filter_adsr_viz_group = QGroupBox("Filter ADSR Curve")
        filter_adsr_viz_layout = QVBoxLayout(filter_adsr_viz_group)
        filter_env_layout.addWidget(filter_adsr_viz_group)

        # Set up pyqtgraph plot for filter ADSR curve
        self.filter_adsr_plot = pg.PlotWidget()
        self.filter_adsr_plot.setLabel('left', 'Amplitude')
        self.filter_adsr_plot.setLabel('bottom', 'Time (normalized)')
        self.filter_adsr_plot.showGrid(x=True, y=True, alpha=0.3)
        filter_adsr_viz_layout.addWidget(self.filter_adsr_plot)

        # Initialize filter ADSR curve plot
        self.filter_adsr_curve = self.filter_adsr_plot.plot(
            np.arange(len(adsr.filter_adsr_curve)),
            adsr.filter_adsr_curve,
            pen=pg.mkPen('r', width=2)
        )
        self.filter_adsr_plot.setYRange(0, 1.1)
        self.filter_adsr_plot.setXRange(0, len(adsr.filter_adsr_curve))

        # LFO Controls
        lfo_group = QGroupBox("LFO Parameters")
        lfo_layout = QGridLayout(lfo_group)
        lfo_tab_layout.addWidget(lfo_group)

        # LFO Enable checkbox
        self.lfo_enable_checkbox = QCheckBox("Enable LFO")
        self.lfo_enable_checkbox.setChecked(config.lfo_enabled)
        self.lfo_enable_checkbox.stateChanged.connect(self.update_lfo_enabled)
        lfo_layout.addWidget(self.lfo_enable_checkbox, 0, 0, 1, 1)

        # LFO Rate - changed from QDoubleSpinBox to QSlider
        lfo_layout.addWidget(QLabel("Rate (Hz)"), 1, 0)
        self.lfo_rate_slider = QSlider(Qt.Horizontal)
        self.lfo_rate_slider.setRange(1, 200)  # 0.1 to 20.0 Hz (x10)
        self.lfo_rate_slider.setValue(int(config.lfo_rate * 10))
        self.lfo_rate_slider.valueChanged.connect(self.update_lfo_rate)
        lfo_layout.addWidget(self.lfo_rate_slider, 1, 1)
        self.lfo_rate_label = QLabel(f"{config.lfo_rate:.1f} Hz")
        lfo_layout.addWidget(self.lfo_rate_label, 1, 2)

        # LFO Depth
        lfo_layout.addWidget(QLabel("Depth"), 1, 3)
        self.lfo_depth_slider = QSlider(Qt.Horizontal)
        self.lfo_depth_slider.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.lfo_depth_slider.setValue(int(config.lfo_depth * 100))
        self.lfo_depth_slider.valueChanged.connect(self.update_lfo_depth)
        lfo_layout.addWidget(self.lfo_depth_slider, 1, 4)
        self.lfo_depth_label = QLabel(f"{config.lfo_depth:.2f}")
        lfo_layout.addWidget(self.lfo_depth_label, 1, 5)

        # LFO Attack Time
        lfo_layout.addWidget(QLabel("Attack (s)"), 2, 0)
        self.lfo_attack_slider = QSlider(Qt.Horizontal)
        self.lfo_attack_slider.setRange(0, 20)  # 0.0 to 2.0 seconds (x10)
        self.lfo_attack_slider.setValue(int(config.lfo_attack_time * 10))
        self.lfo_attack_slider.valueChanged.connect(self.update_lfo_attack_time)
        lfo_layout.addWidget(self.lfo_attack_slider, 2, 1)
        self.lfo_attack_label = QLabel(f"{config.lfo_attack_time:.1f} s")
        lfo_layout.addWidget(self.lfo_attack_label, 2, 2)

        # LFO Delay Time
        lfo_layout.addWidget(QLabel("Delay (s)"), 3, 0)
        self.lfo_delay_slider = QSlider(Qt.Horizontal)
        self.lfo_delay_slider.setRange(0, 20)  # 0.0 to 2.0 seconds (x10)
        self.lfo_delay_slider.setValue(int(config.lfo_delay_time * 10))
        self.lfo_delay_slider.valueChanged.connect(self.update_lfo_delay_time)
        lfo_layout.addWidget(self.lfo_delay_slider, 3, 1)
        self.lfo_delay_label = QLabel(f"{config.lfo_delay_time:.1f} s")
        lfo_layout.addWidget(self.lfo_delay_label, 3, 2)

        # LFO Target
        lfo_layout.addWidget(QLabel("Target"), 2, 3)
        self.lfo_target_combo = QComboBox()
        self.lfo_target_combo.addItems(["pitch", "volume", "cutoff"])
        self.lfo_target_combo.setCurrentText(config.lfo_target)
        self.lfo_target_combo.currentTextChanged.connect(self.update_lfo_target)
        lfo_layout.addWidget(self.lfo_target_combo, 2, 3, 1, 3)

        # Delay effect controls
        delay_group = QGroupBox("Delay Parameters")
        delay_layout = QGridLayout(delay_group)
        delay_tab_layout.addWidget(delay_group)

        # Delay enable checkbox
        self.delay_enable_checkbox = QCheckBox("Enable Delay")
        self.delay_enable_checkbox.setChecked(config.delay_enabled)
        self.delay_enable_checkbox.stateChanged.connect(self.update_delay_enabled)
        delay_layout.addWidget(self.delay_enable_checkbox, 0, 0, 1, 3)

        # Tempo sync checkbox
        self.delay_sync_checkbox = QCheckBox("Sync to Tempo")
        self.delay_sync_checkbox.setChecked(config.delay_sync_enabled)
        self.delay_sync_checkbox.stateChanged.connect(self.update_delay_sync)
        delay_layout.addWidget(self.delay_sync_checkbox, 0, 3, 1, 3)

        # Create frame for tempo-sync controls
        self.tempo_sync_frame = QFrame()
        tempo_sync_hlayout = QHBoxLayout(self.tempo_sync_frame)
        tempo_sync_hlayout.setContentsMargins(0, 0, 0, 0)
        tempo_sync_hlayout.setSpacing(8)
        delay_layout.addWidget(self.tempo_sync_frame, 1, 0, 1, 6)

        # Division
        tempo_sync_hlayout.addWidget(QLabel("Division:"), stretch=0)
        self.delay_division_combo = QComboBox()
        self.delay_division_combo.addItems(list(delay.DIV2MULT.keys()))
        self.delay_division_combo.setCurrentText(config.delay_division)
        self.delay_division_combo.currentTextChanged.connect(self.update_delay_division)
        tempo_sync_hlayout.addWidget(self.delay_division_combo, stretch=1)

        # BPM
        tempo_sync_hlayout.addWidget(QLabel("BPM:"), stretch=0)
        self.delay_bpm_spinbox = QSpinBox()
        self.delay_bpm_spinbox.setRange(40, 300)
        self.delay_bpm_spinbox.setValue(config.bpm)
        self.delay_bpm_spinbox.valueChanged.connect(self.update_delay_bpm)
        tempo_sync_hlayout.addWidget(self.delay_bpm_spinbox, stretch=1)

        # Computed delay time
        self.delay_ms_label = QLabel(f"{config.delay_time_ms:.1f} ms")
        tempo_sync_hlayout.addWidget(self.delay_ms_label, stretch=0)
        tempo_sync_hlayout.addStretch(1)
        self.tempo_sync_frame.setMaximumHeight(40)

        # Delay time manual control
        delay_layout.addWidget(QLabel("Time (ms):"), 2, 0)
        self.delay_time_slider = QSlider(Qt.Horizontal)
        self.delay_time_slider.setRange(10, 1000)
        self.delay_time_slider.setValue(int(config.delay_time_ms))
        self.delay_time_slider.valueChanged.connect(self.update_delay_time)
        delay_layout.addWidget(self.delay_time_slider, 2, 1, 1, 4)
        self.delay_time_label = QLabel(f"{config.delay_time_ms:.0f} ms")
        delay_layout.addWidget(self.delay_time_label, 2, 5)
        delay_layout.setRowStretch(2, 1)

        # Delay feedback control
        delay_layout.addWidget(QLabel("Feedback:"), 3, 0)
        self.delay_feedback_slider = QSlider(Qt.Horizontal)
        self.delay_feedback_slider.setRange(0, 95)  # 0.0 to 0.95 (x100)
        self.delay_feedback_slider.setValue(int(config.delay_feedback * 100))
        self.delay_feedback_slider.valueChanged.connect(self.update_delay_feedback)
        delay_layout.addWidget(self.delay_feedback_slider, 3, 1, 1, 4)
        self.delay_feedback_label = QLabel(f"{config.delay_feedback:.2f}")
        delay_layout.addWidget(self.delay_feedback_label, 3, 5)
        delay_layout.setRowStretch(3, 1)

        # Delay mix control
        delay_layout.addWidget(QLabel("Mix (Dry ↔ Wet):"), 4, 0)
        self.delay_mix_slider = QSlider(Qt.Horizontal)
        self.delay_mix_slider.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.delay_mix_slider.setValue(int(config.delay_mix * 100))
        self.delay_mix_slider.valueChanged.connect(self.update_delay_mix)
        delay_layout.addWidget(self.delay_mix_slider, 4, 1, 1, 4)
        self.delay_mix_label = QLabel(f"{config.delay_mix:.2f}")
        delay_layout.addWidget(self.delay_mix_label, 4, 5)
        delay_layout.setRowStretch(4, 1)

        # Update the visibility of sync controls based on initial state
        self.update_delay_sync_controls()

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

        # Initialize delay time from BPM
        delay.update_delay_from_bpm()

    def update_plots(self):
        """Update function for waveform and spectrum plots animation."""
        if not self.animation_running:
            return

        # Check if waveform type has changed and update GUI if needed
        for button in self.waveform_buttons:
            if button.property("value") == config.waveform_type and not button.isChecked():
                button.setChecked(True)
                break

        # Check if octave has changed and update GUI if needed
        current_octave_text = f"{config.octave_offset // 12:+d}"
        if self.octave_label.text() != current_octave_text:
            self.octave_label.setText(current_octave_text)

        # Check if filter cutoff has changed and update GUI if needed
        if self.cutoff_slider.value() != filter.cutoff:
            self.cutoff_slider.setValue(int(filter.cutoff))
            self.cutoff_label.setText(f"{filter.cutoff:.0f} Hz")

        # Check if filter resonance has changed and update GUI if needed
        if self.resonance_slider.value() != int(filter.resonance * 100):
            self.resonance_slider.setValue(int(filter.resonance * 100))
            self.resonance_label.setText(f"{filter.resonance:.2f}")

        # Check if filter envelope amount has changed and update GUI
        if self.filter_env_amount_slider.value() != adsr.filter_env_amount:
            self.filter_env_amount_slider.setValue(int(adsr.filter_env_amount))
            self.filter_env_amount_label.setText(f"{adsr.filter_env_amount:.0f} Hz")

        # Check if LFO attack time has changed and update GUI if needed
        if self.lfo_attack_slider.value() != int(config.lfo_attack_time * 10):
            self.lfo_attack_slider.setValue(int(config.lfo_attack_time * 10))
            self.lfo_attack_label.setText(f"{config.lfo_attack_time:.1f} s")

        # Check if LFO rate has changed and update GUI if needed
        if self.lfo_rate_slider.value() != int(config.lfo_rate * 10):
            self.lfo_rate_slider.setValue(int(config.lfo_rate * 10))
            self.lfo_rate_label.setText(f"{config.lfo_rate:.1f} Hz")

        # Check if LFO delay time has changed and update GUI if needed
        if self.lfo_delay_slider.value() != int(config.lfo_delay_time * 10):
            self.lfo_delay_slider.setValue(int(config.lfo_delay_time * 10))
            self.lfo_delay_label.setText(f"{config.lfo_delay_time:.1f} s")

        # Check if LFO depth has changed and update GUI if needed
        if self.lfo_depth_slider.value() != int(config.lfo_depth * 100):
            self.lfo_depth_slider.setValue(int(config.lfo_depth * 100))
            self.lfo_depth_label.setText(f"{config.lfo_depth:.2f}")

        # Check if LFO enabled status has changed and update GUI if needed
        if self.lfo_enable_checkbox.isChecked() != config.lfo_enabled:
            self.lfo_enable_checkbox.setChecked(config.lfo_enabled)

        # Check if filter enabled status has changed and update GUI if needed
        if self.filter_enable_checkbox.isChecked() != filter.filter_enabled:
            self.filter_enable_checkbox.setChecked(filter.filter_enabled)

        # Check if delay parameters have changed and update GUI if needed
        if self.delay_enable_checkbox.isChecked() != config.delay_enabled:
            self.delay_enable_checkbox.setChecked(config.delay_enabled)

        if self.delay_sync_checkbox.isChecked() != config.delay_sync_enabled:
            self.delay_sync_checkbox.setChecked(config.delay_sync_enabled)
            self.update_delay_sync_controls()

        if self.delay_time_slider.value() != int(config.delay_time_ms):
            self.delay_time_slider.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

        if self.delay_feedback_slider.value() != int(config.delay_feedback * 100):
            self.delay_feedback_slider.setValue(int(config.delay_feedback * 100))
            self.delay_feedback_label.setText(f"{config.delay_feedback:.2f}")

        if self.delay_mix_slider.value() != int(config.delay_mix * 100):
            self.delay_mix_slider.setValue(int(config.delay_mix * 100))
            self.delay_mix_label.setText(f"{config.delay_mix:.2f}")

        if self.delay_division_combo.currentText() != config.delay_division:
            self.delay_division_combo.setCurrentText(config.delay_division)

        if self.delay_bpm_spinbox.value() != config.bpm:
            self.delay_bpm_spinbox.setValue(config.bpm)

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

        # Check if filter ADSR parameters have changed
        filter_adsr_changed = False

        if self.filter_attack_slider.value() != int(adsr.filter_adsr['attack'] * 100):
            self.filter_attack_slider.setValue(int(adsr.filter_adsr['attack'] * 100))
            self.filter_attack_label.setText(f"{adsr.filter_adsr['attack']:.2f} s")
            filter_adsr_changed = True

        if self.filter_decay_slider.value() != int(adsr.filter_adsr['decay'] * 100):
            self.filter_decay_slider.setValue(int(adsr.filter_adsr['decay'] * 100))
            self.filter_decay_label.setText(f"{adsr.filter_adsr['decay']:.2f} s")
            filter_adsr_changed = True

        if self.filter_sustain_slider.value() != int(adsr.filter_adsr['sustain'] * 100):
            self.filter_sustain_slider.setValue(int(adsr.filter_adsr['sustain'] * 100))
            self.filter_sustain_label.setText(f"{adsr.filter_adsr['sustain']:.2f}")
            filter_adsr_changed = True

        if self.filter_release_slider.value() != int(adsr.filter_adsr['release'] * 100):
            self.filter_release_slider.setValue(int(adsr.filter_adsr['release'] * 100))
            self.filter_release_label.setText(f"{adsr.filter_adsr['release']:.2f} s")
            filter_adsr_changed = True

        # Check if volume has changed
        if self.volume_slider.value() != int(config.volume * 100):
            self.volume_slider.setValue(int(config.volume * 100))
            self.volume_label.setText(f"{config.volume:.2f}")

        # Update ADSR curves if parameters changed
        if adsr_changed:
            self.plot_adsr_curve()

        if filter_adsr_changed:
            self.plot_filter_adsr_curve()

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

    def plot_filter_adsr_curve(self):
        """Update the filter ADSR curve in the plot."""
        self.filter_adsr_curve.setData(
            np.arange(len(adsr.filter_adsr_curve)),
            adsr.filter_adsr_curve
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

    def update_filter_adsr(self, param, value):
        """Update the specified filter ADSR parameter."""
        adsr.filter_adsr[param] = value
        adsr.update_filter_adsr_curve()

        # Update the filter ADSR curve visualization
        self.plot_filter_adsr_curve()

        # Update label
        if param == 'attack':
            self.filter_attack_label.setText(f"{value:.2f} s")
        elif param == 'decay':
            self.filter_decay_label.setText(f"{value:.2f} s")
        elif param == 'sustain':
            self.filter_sustain_label.setText(f"{value:.2f}")
        elif param == 'release':
            self.filter_release_label.setText(f"{value:.2f} s")

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

    def update_filter_resonance(self, value):
        """Update the filter resonance."""
        resonance = value / 100.0
        filter.resonance = resonance
        self.resonance_label.setText(f"{resonance:.2f}")

    def update_filter_env_amount(self, value):
        """Update the filter envelope amount."""
        amount = float(value)
        adsr.filter_env_amount = amount
        self.filter_env_amount_label.setText(f"{amount:.0f} Hz")

    def update_mono_mode(self, state):
        """Update the mono mode setting."""
        config.mono_mode = (state == Qt.Checked)
        # Clear active notes to prevent stuck notes when switching modes
        with config.notes_lock:
            config.active_notes.clear()
            config.mono_pressed_keys.clear()

    def update_glide_time(self, value):
        """Update the glide time setting."""
        # Ensure minimum value of 1ms to avoid division by zero
        glide_ms = max(1, value)
        glide_time = glide_ms / 1000.0  # Convert from ms to seconds
        config.glide_time = glide_time
        self.glide_label.setText(f"{glide_ms:.0f} ms")

    def update_lfo_rate(self, value):
        """Update the LFO rate setting."""
        config.lfo_rate = value / 10.0
        self.lfo_rate_label.setText(f"{config.lfo_rate:.1f} Hz")

    def update_lfo_depth(self, value):
        """Update the LFO depth setting."""
        depth = value / 100.0
        config.lfo_depth = depth
        self.lfo_depth_label.setText(f"{depth:.2f}")

    def update_lfo_target(self, value):
        """Update the LFO target setting."""
        config.lfo_target = value

    def update_lfo_attack_time(self, value):
        """Update the LFO attack time setting."""
        config.lfo_attack_time = value / 10.0
        self.lfo_attack_label.setText(f"{config.lfo_attack_time:.1f} s")

    def update_lfo_delay_time(self, value):
        """Update the LFO delay time setting."""
        config.lfo_delay_time = value / 10.0
        self.lfo_delay_label.setText(f"{config.lfo_delay_time:.1f} s")

    def update_lfo_enabled(self, state):
        """Update the LFO enabled setting."""
        config.lfo_enabled = (state == Qt.Checked)

    def update_filter_enabled(self, state):
        """Update the filter enabled setting."""
        filter.filter_enabled = (state == Qt.Checked)

    def update_delay_enabled(self, state):
        """Update the delay enabled setting."""
        config.delay_enabled = (state == Qt.Checked)
        if not config.delay_enabled:
            delay.clear_cache()

    def update_delay_sync(self, state):
        """Update whether delay time is synced to BPM."""
        config.delay_sync_enabled = (state == Qt.Checked)
        self.update_delay_sync_controls()

        # If enabling sync, update delay time from BPM
        if config.delay_sync_enabled:
            delay.update_delay_from_bpm()
            self.delay_time_slider.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

    def update_delay_sync_controls(self):
        """Show/hide appropriate controls based on sync setting."""
        # Show/hide sync controls based on sync checkbox state
        self.tempo_sync_frame.setVisible(config.delay_sync_enabled)

        # Enable/disable manual time slider based on sync setting
        self.delay_time_slider.setEnabled(not config.delay_sync_enabled)

    def update_delay_time(self, value):
        """Update the delay time setting."""
        # Only update if we're not in sync mode
        if not config.delay_sync_enabled:
            config.delay_time_ms = float(value)
            delay.set_time(value)
            self.delay_time_label.setText(f"{value:.0f} ms")
            self.delay_ms_label.setText(f"{value:.1f} ms")

    def update_delay_feedback(self, value):
        """Update the delay feedback setting."""
        feedback = value / 100.0
        config.delay_feedback = feedback
        self.delay_feedback_label.setText(f"{feedback:.2f}")

    def update_delay_mix(self, value):
        """Update the delay mix setting."""
        mix = value / 100.0
        config.delay_mix = mix
        self.delay_mix_label.setText(f"{mix:.2f}")

    def update_delay_division(self, division):
        """Update the delay division setting and recalculate time."""
        config.delay_division = division
        if config.delay_sync_enabled:
            delay.update_delay_from_bpm()

            # Update displayed values
            self.delay_time_slider.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

    def update_delay_bpm(self, bpm):
        """Update the global BPM and recalculate delay time."""
        config.bpm = bpm

        if config.delay_sync_enabled:
            delay.update_delay_from_bpm()

            # Update displayed values
            self.delay_time_slider.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

        # Also update the sequencer BPM if it exists
        if hasattr(self, 'sequencer') and self.sequencer:
            self.sequencer.bpm_spinbox.setValue(bpm)

    def sync_sequencer_bpm(self, bpm):
        """Sync the global BPM when sequencer BPM changes."""
        config.bpm = bpm
        self.delay_bpm_spinbox.setValue(bpm)

        if config.delay_sync_enabled:
            delay.update_delay_from_bpm()

            # Update displayed delay time values
            self.delay_time_slider.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

    def decrease_octave(self):
        """Decrease the octave by one."""
        if config.octave_offset > 12 * config.octave_min:
            config.octave_offset -= 12
            self.octave_label.setText(f"{config.octave_offset // 12:+d}")
            # Update slider without triggering the signal
            self.octave_slider.blockSignals(True)
            self.octave_slider.setValue(config.octave_offset // 12)
            self.octave_slider.blockSignals(False)

    def increase_octave(self):
        """Increase the octave by one."""
        if config.octave_offset < 12 * config.octave_max:
            config.octave_offset += 12
            self.octave_label.setText(f"{config.octave_offset // 12:+d}")
            # Update slider without triggering the signal
            self.octave_slider.blockSignals(True)
            self.octave_slider.setValue(config.octave_offset // 12)
            self.octave_slider.blockSignals(False)

    def update_octave(self, value):
        """Update the octave from slider value."""
        # Convert slider value to offset (x12 semitones per octave)
        config.octave_offset = value * 12
        self.octave_label.setText(f"{value:+d}")

    def closeEvent(self, event):
        """Clean up when window is closed."""
        self.animation_running = False
        if hasattr(self, 'timer'):
            self.timer.stop()

        # Stop sequencer if running
        if self.sequencer:
            self.sequencer.stop()

        event.accept()


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
