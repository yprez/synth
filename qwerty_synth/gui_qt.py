"""PyQt GUI for QWERTY Synth providing controls for waveform and ADSR."""

import sys
import time
import signal
import os
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QRadioButton, QPushButton, QGroupBox, QGridLayout,
    QCheckBox, QComboBox, QTabWidget, QSpinBox, QFrame,
    QFileDialog, QProgressBar, QDial
)
import pyqtgraph as pg
import qdarkstyle

from qwerty_synth import config
from qwerty_synth import adsr
from qwerty_synth import synth
from qwerty_synth import input as kb_input
from qwerty_synth import filter
from qwerty_synth.delay import DIV2MULT
from qwerty_synth.step_sequencer import StepSequencer
from qwerty_synth.controller import play_midi_file

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
        self.visualization_enabled = False

        # Keep track of the last GUI update time
        self.last_update_time = time.time()

        # Create sequencer instance
        self.sequencer = None  # Will be initialized in setup_ui

        # MIDI player state
        self.midi_player_active = False
        self.midi_file_path = None
        self.midi_playback_thread = None
        self.midi_paused = False

        # Set up the user interface
        self.setup_ui()
        self.running = True

        # Start animation timer after UI is set up
        self.start_animation()

        # Handle window close event
        self.showMaximized()  # Start maximized but with window controls

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
        volume_layout = QGridLayout(volume_group)
        control_layout.addWidget(volume_group, stretch=1)

        volume_label = QLabel("Volume")
        volume_label.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(volume_label, 0, 0)

        self.volume_dial = QDial()
        self.volume_dial.setRange(0, 100)
        self.volume_dial.setValue(int(config.volume * 100))
        self.volume_dial.valueChanged.connect(self.update_volume)
        self.volume_dial.setNotchesVisible(True)
        volume_layout.addWidget(self.volume_dial, 1, 0)

        self.volume_label = QLabel(f"{config.volume:.2f}")
        self.volume_label.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(self.volume_label, 2, 0)

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

        # Create a horizontal layout for filter and drive controls
        filter_drive_layout = QHBoxLayout()
        main_layout.addLayout(filter_drive_layout)

        # Filter Control
        filter_group = QGroupBox("Filter Control")
        filter_layout = QGridLayout(filter_group)
        filter_drive_layout.addWidget(filter_group)

        # Filter enable checkbox
        self.filter_enable_checkbox = QCheckBox("Enable Filter")
        self.filter_enable_checkbox.setChecked(filter.filter_enabled)
        self.filter_enable_checkbox.stateChanged.connect(self.update_filter_enabled)
        filter_layout.addWidget(self.filter_enable_checkbox, 0, 0, 1, 3)

        # Cutoff control
        cutoff_label = QLabel("Cutoff")
        cutoff_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(cutoff_label, 1, 0)

        self.cutoff_dial = QDial()
        self.cutoff_dial.setRange(100, 10000)
        self.cutoff_dial.setValue(int(filter.cutoff))
        self.cutoff_dial.valueChanged.connect(self.update_filter_cutoff)
        self.cutoff_dial.setNotchesVisible(True)
        filter_layout.addWidget(self.cutoff_dial, 2, 0)

        self.cutoff_label = QLabel(f"{filter.cutoff:.0f} Hz")
        self.cutoff_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.cutoff_label, 3, 0)

        # Resonance control
        res_label = QLabel("Resonance")
        res_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(res_label, 1, 1)

        self.resonance_dial = QDial()
        self.resonance_dial.setRange(0, 95)  # 0.0 to 0.95 (x100)
        self.resonance_dial.setValue(int(filter.resonance * 100))
        self.resonance_dial.valueChanged.connect(self.update_filter_resonance)
        self.resonance_dial.setNotchesVisible(True)
        filter_layout.addWidget(self.resonance_dial, 2, 1)

        self.resonance_label = QLabel(f"{filter.resonance:.2f}")
        self.resonance_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.resonance_label, 3, 1)

        # Filter envelope amount control
        env_label = QLabel("Env Amount")
        env_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(env_label, 1, 2)

        self.filter_env_amount_dial = QDial()
        self.filter_env_amount_dial.setRange(0, 10000)
        self.filter_env_amount_dial.setValue(int(adsr.filter_env_amount))
        self.filter_env_amount_dial.valueChanged.connect(self.update_filter_env_amount)
        self.filter_env_amount_dial.setNotchesVisible(True)
        filter_layout.addWidget(self.filter_env_amount_dial, 2, 2)

        self.filter_env_amount_label = QLabel(f"{adsr.filter_env_amount:.0f} Hz")
        self.filter_env_amount_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.filter_env_amount_label, 3, 2)

        # Drive Control
        drive_group = QGroupBox("Drive Control")
        drive_layout = QGridLayout(drive_group)
        filter_drive_layout.addWidget(drive_group)

        # Drive enable checkbox
        self.drive_enable_checkbox = QCheckBox("Enable Drive")
        self.drive_enable_checkbox.setChecked(config.drive_on)
        self.drive_enable_checkbox.stateChanged.connect(self.update_drive_enabled)
        drive_layout.addWidget(self.drive_enable_checkbox, 0, 0, 1, 2)

        # Drive type selector
        drive_layout.addWidget(QLabel("Type:"), 0, 2)
        self.drive_type_combo = QComboBox()
        self.drive_type_combo.addItems(["tanh", "arctan", "cubic", "fuzz", "asymmetric"])
        self.drive_type_combo.setCurrentText(config.drive_type)
        self.drive_type_combo.currentTextChanged.connect(self.update_drive_type)
        drive_layout.addWidget(self.drive_type_combo, 0, 3)

        # Drive gain control
        drive_gain_label = QLabel("Drive Gain")
        drive_gain_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(drive_gain_label, 1, 0)

        self.drive_dial = QDial()
        self.drive_dial.setRange(0, 300)  # 0.0 to 3.0 (x100)
        self.drive_dial.setValue(int(config.drive_gain * 100))
        self.drive_dial.valueChanged.connect(self.update_drive_gain)
        self.drive_dial.setNotchesVisible(True)
        drive_layout.addWidget(self.drive_dial, 2, 0)

        self.drive_label = QLabel(f"{config.drive_gain:.1f}")
        self.drive_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(self.drive_label, 3, 0)

        # Drive tone control
        tone_label = QLabel("Tone")
        tone_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(tone_label, 1, 1)

        self.drive_tone_dial = QDial()
        self.drive_tone_dial.setRange(-95, 95)  # -0.95 to 0.95 (x100)
        self.drive_tone_dial.setValue(int(config.drive_tone * 100))
        self.drive_tone_dial.valueChanged.connect(self.update_drive_tone)
        self.drive_tone_dial.setNotchesVisible(True)
        drive_layout.addWidget(self.drive_tone_dial, 2, 1)

        self.drive_tone_label = QLabel(f"{config.drive_tone:.2f}")
        self.drive_tone_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(self.drive_tone_label, 3, 1)

        # Drive mix control
        mix_label = QLabel("Mix")
        mix_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(mix_label, 1, 2)

        self.drive_mix_dial = QDial()
        self.drive_mix_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.drive_mix_dial.setValue(int(config.drive_mix * 100))
        self.drive_mix_dial.valueChanged.connect(self.update_drive_mix)
        self.drive_mix_dial.setNotchesVisible(True)
        drive_layout.addWidget(self.drive_mix_dial, 2, 2)

        self.drive_mix_label = QLabel(f"{config.drive_mix:.2f}")
        self.drive_mix_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(self.drive_mix_label, 3, 2)

        # Drive asymmetry control (only for asymmetric mode)
        asym_label = QLabel("Asymmetry")
        asym_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(asym_label, 1, 3)

        self.drive_asym_dial = QDial()
        self.drive_asym_dial.setRange(0, 90)  # 0.0 to 0.9 (x100)
        self.drive_asym_dial.setValue(int(config.drive_asymmetry * 100))
        self.drive_asym_dial.valueChanged.connect(self.update_drive_asymmetry)
        self.drive_asym_dial.setNotchesVisible(True)
        drive_layout.addWidget(self.drive_asym_dial, 2, 3)

        self.drive_asym_label = QLabel(f"{config.drive_asymmetry:.2f}")
        self.drive_asym_label.setAlignment(Qt.AlignCenter)
        drive_layout.addWidget(self.drive_asym_label, 3, 3)

        # Create a tabbed widget for amp ADSR and filter ADSR
        envelope_tabs = QTabWidget()
        main_layout.addWidget(envelope_tabs, stretch=1)

        # Create a combined envelopes tab (instead of separate amp and filter tabs)
        envelopes_widget = QWidget()
        envelopes_layout = QHBoxLayout(envelopes_widget)
        envelope_tabs.addTab(envelopes_widget, "Envelopes")

        # Create amp and filter envelope sections that will be placed side by side
        amp_env_widget = QWidget()
        amp_env_layout = QHBoxLayout(amp_env_widget)
        envelopes_layout.addWidget(amp_env_widget)

        filter_env_widget = QWidget()
        filter_env_layout = QHBoxLayout(filter_env_widget)
        envelopes_layout.addWidget(filter_env_widget)

        # Create the LFO tab
        lfo_tab_widget = QWidget()
        lfo_tab_layout = QHBoxLayout(lfo_tab_widget)
        envelope_tabs.addTab(lfo_tab_widget, "LFO Control")

        # Create the delay effect tab
        delay_tab_widget = QWidget()
        delay_tab_layout = QHBoxLayout(delay_tab_widget)
        envelope_tabs.addTab(delay_tab_widget, "Delay")

        # Create the chorus effect tab
        chorus_tab_widget = QWidget()
        chorus_tab_layout = QHBoxLayout(chorus_tab_widget)
        envelope_tabs.addTab(chorus_tab_widget, "Chorus")

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

        # Attack control
        attack_label = QLabel("Attack")
        attack_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(attack_label, 0, 0)

        self.attack_dial = QDial()
        self.attack_dial.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.attack_dial.setValue(int(adsr.adsr['attack'] * 100))
        self.attack_dial.valueChanged.connect(
            lambda v: self.update_adsr('attack', v/100.0)
        )
        self.attack_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.attack_dial, 1, 0)

        self.attack_label = QLabel(f"{adsr.adsr['attack']:.2f} s")
        self.attack_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.attack_label, 2, 0)

        # Decay control
        decay_label = QLabel("Decay")
        decay_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(decay_label, 0, 1)

        self.decay_dial = QDial()
        self.decay_dial.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.decay_dial.setValue(int(adsr.adsr['decay'] * 100))
        self.decay_dial.valueChanged.connect(
            lambda v: self.update_adsr('decay', v/100.0)
        )
        self.decay_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.decay_dial, 1, 1)

        self.decay_label = QLabel(f"{adsr.adsr['decay']:.2f} s")
        self.decay_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.decay_label, 2, 1)

        # Sustain control
        sustain_label = QLabel("Sustain")
        sustain_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(sustain_label, 0, 2)

        self.sustain_dial = QDial()
        self.sustain_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.sustain_dial.setValue(int(adsr.adsr['sustain'] * 100))
        self.sustain_dial.valueChanged.connect(
            lambda v: self.update_adsr('sustain', v/100.0)
        )
        self.sustain_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.sustain_dial, 1, 2)

        self.sustain_label = QLabel(f"{adsr.adsr['sustain']:.2f}")
        self.sustain_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.sustain_label, 2, 2)

        # Release control
        release_label = QLabel("Release")
        release_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(release_label, 0, 3)

        self.release_dial = QDial()
        self.release_dial.setRange(1, 300)  # 0.01 to 3.0 seconds (x100)
        self.release_dial.setValue(int(adsr.adsr['release'] * 100))
        self.release_dial.valueChanged.connect(
            lambda v: self.update_adsr('release', v/100.0)
        )
        self.release_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.release_dial, 1, 3)

        self.release_label = QLabel(f"{adsr.adsr['release']:.2f} s")
        self.release_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.release_label, 2, 3)

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
            pen=pg.mkPen((100, 200, 255), width=2)
        )
        self.adsr_plot.setYRange(0, 1.1)
        self.adsr_plot.setXRange(0, len(adsr.adsr_curve))

        # ADSR controls for filter envelope
        filter_adsr_group = QGroupBox("Filter ADSR Envelope")
        filter_adsr_controls = QGridLayout(filter_adsr_group)
        filter_env_layout.addWidget(filter_adsr_group)

        # Filter Attack control
        filter_attack_label = QLabel("Attack")
        filter_attack_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_attack_label, 0, 0)

        self.filter_attack_dial = QDial()
        self.filter_attack_dial.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.filter_attack_dial.setValue(int(adsr.filter_adsr['attack'] * 100))
        self.filter_attack_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('attack', v/100.0)
        )
        self.filter_attack_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_attack_dial, 1, 0)

        self.filter_attack_label = QLabel(f"{adsr.filter_adsr['attack']:.2f} s")
        self.filter_attack_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_attack_label, 2, 0)

        # Filter Decay control
        filter_decay_label = QLabel("Decay")
        filter_decay_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_decay_label, 0, 1)

        self.filter_decay_dial = QDial()
        self.filter_decay_dial.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.filter_decay_dial.setValue(int(adsr.filter_adsr['decay'] * 100))
        self.filter_decay_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('decay', v/100.0)
        )
        self.filter_decay_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_decay_dial, 1, 1)

        self.filter_decay_label = QLabel(f"{adsr.filter_adsr['decay']:.2f} s")
        self.filter_decay_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_decay_label, 2, 1)

        # Filter Sustain control
        filter_sustain_label = QLabel("Sustain")
        filter_sustain_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_sustain_label, 0, 2)

        self.filter_sustain_dial = QDial()
        self.filter_sustain_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.filter_sustain_dial.setValue(int(adsr.filter_adsr['sustain'] * 100))
        self.filter_sustain_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('sustain', v/100.0)
        )
        self.filter_sustain_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_sustain_dial, 1, 2)

        self.filter_sustain_label = QLabel(f"{adsr.filter_adsr['sustain']:.2f}")
        self.filter_sustain_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_sustain_label, 2, 2)

        # Filter Release control
        filter_release_label = QLabel("Release")
        filter_release_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_release_label, 0, 3)

        self.filter_release_dial = QDial()
        self.filter_release_dial.setRange(1, 300)  # 0.01 to 3.0 seconds (x100)
        self.filter_release_dial.setValue(int(adsr.filter_adsr['release'] * 100))
        self.filter_release_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('release', v/100.0)
        )
        self.filter_release_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_release_dial, 1, 3)

        self.filter_release_label = QLabel(f"{adsr.filter_adsr['release']:.2f} s")
        self.filter_release_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_release_label, 2, 3)

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

        # LFO Enable checkbox and Target selector in top row
        control_row = QHBoxLayout()
        lfo_layout.addLayout(control_row, 0, 0, 1, 4)

        self.lfo_enable_checkbox = QCheckBox("Enable LFO")
        self.lfo_enable_checkbox.setChecked(config.lfo_enabled)
        self.lfo_enable_checkbox.stateChanged.connect(self.update_lfo_enabled)
        control_row.addWidget(self.lfo_enable_checkbox)

        control_row.addWidget(QLabel("Target:"))
        self.lfo_target_combo = QComboBox()
        self.lfo_target_combo.addItems(["pitch", "volume", "cutoff"])
        self.lfo_target_combo.setCurrentText(config.lfo_target)
        self.lfo_target_combo.currentTextChanged.connect(self.update_lfo_target)
        control_row.addWidget(self.lfo_target_combo)

        # LFO Rate control
        rate_label = QLabel("Rate")
        rate_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(rate_label, 1, 0)

        self.lfo_rate_dial = QDial()
        self.lfo_rate_dial.setRange(1, 200)  # 0.1 to 20.0 Hz (x10)
        self.lfo_rate_dial.setValue(int(config.lfo_rate * 10))
        self.lfo_rate_dial.valueChanged.connect(self.update_lfo_rate)
        self.lfo_rate_dial.setNotchesVisible(True)
        lfo_layout.addWidget(self.lfo_rate_dial, 2, 0)

        self.lfo_rate_label = QLabel(f"{config.lfo_rate:.1f} Hz")
        self.lfo_rate_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(self.lfo_rate_label, 3, 0)

        # LFO Depth control
        depth_label = QLabel("Depth")
        depth_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(depth_label, 1, 1)

        self.lfo_depth_dial = QDial()
        self.lfo_depth_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.lfo_depth_dial.setValue(int(config.lfo_depth * 100))
        self.lfo_depth_dial.valueChanged.connect(self.update_lfo_depth)
        self.lfo_depth_dial.setNotchesVisible(True)
        lfo_layout.addWidget(self.lfo_depth_dial, 2, 1)

        self.lfo_depth_label = QLabel(f"{config.lfo_depth:.2f}")
        self.lfo_depth_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(self.lfo_depth_label, 3, 1)

        # LFO Attack Time control
        attack_label = QLabel("Attack")
        attack_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(attack_label, 1, 2)

        self.lfo_attack_dial = QDial()
        self.lfo_attack_dial.setRange(0, 20)  # 0.0 to 2.0 seconds (x10)
        self.lfo_attack_dial.setValue(int(config.lfo_attack_time * 10))
        self.lfo_attack_dial.valueChanged.connect(self.update_lfo_attack_time)
        self.lfo_attack_dial.setNotchesVisible(True)
        lfo_layout.addWidget(self.lfo_attack_dial, 2, 2)

        self.lfo_attack_label = QLabel(f"{config.lfo_attack_time:.1f} s")
        self.lfo_attack_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(self.lfo_attack_label, 3, 2)

        # LFO Delay Time control
        delay_label = QLabel("Delay")
        delay_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(delay_label, 1, 3)

        self.lfo_delay_dial = QDial()
        self.lfo_delay_dial.setRange(0, 20)  # 0.0 to 2.0 seconds (x10)
        self.lfo_delay_dial.setValue(int(config.lfo_delay_time * 10))
        self.lfo_delay_dial.valueChanged.connect(self.update_lfo_delay_time)
        self.lfo_delay_dial.setNotchesVisible(True)
        lfo_layout.addWidget(self.lfo_delay_dial, 2, 3)

        self.lfo_delay_label = QLabel(f"{config.lfo_delay_time:.1f} s")
        self.lfo_delay_label.setAlignment(Qt.AlignCenter)
        lfo_layout.addWidget(self.lfo_delay_label, 3, 3)

        # Delay effect controls
        delay_group = QGroupBox("Delay Parameters")
        delay_layout = QGridLayout(delay_group)
        delay_tab_layout.addWidget(delay_group)

        # Top row controls
        control_row = QHBoxLayout()
        delay_layout.addLayout(control_row, 0, 0, 1, 4)

        # Delay enable checkbox
        self.delay_enable_checkbox = QCheckBox("Enable Delay")
        self.delay_enable_checkbox.setChecked(config.delay_enabled)
        self.delay_enable_checkbox.stateChanged.connect(self.update_delay_enabled)
        control_row.addWidget(self.delay_enable_checkbox)

        # Ping-pong checkbox
        self.pingpong_checkbox = QCheckBox("Ping-Pong (Stereo)")
        self.pingpong_checkbox.setChecked(config.delay_pingpong)
        self.pingpong_checkbox.stateChanged.connect(self.update_delay_pingpong)
        control_row.addWidget(self.pingpong_checkbox)

        # Tempo sync checkbox
        self.delay_sync_checkbox = QCheckBox("Sync to Tempo")
        self.delay_sync_checkbox.setChecked(config.delay_sync_enabled)
        self.delay_sync_checkbox.stateChanged.connect(self.update_delay_sync)
        control_row.addWidget(self.delay_sync_checkbox)

        # Create frame for tempo-sync controls
        self.tempo_sync_frame = QFrame()
        tempo_sync_layout = QHBoxLayout(self.tempo_sync_frame)
        tempo_sync_layout.setContentsMargins(0, 0, 0, 0)
        tempo_sync_layout.setSpacing(8)
        delay_layout.addWidget(self.tempo_sync_frame, 1, 0, 1, 4)

        # Division
        tempo_sync_layout.addWidget(QLabel("Division:"))
        self.delay_division_combo = QComboBox()
        self.delay_division_combo.addItems(list(DIV2MULT.keys()))
        self.delay_division_combo.setCurrentText(config.delay_division)
        self.delay_division_combo.currentTextChanged.connect(self.update_delay_division)
        tempo_sync_layout.addWidget(self.delay_division_combo)

        # BPM
        tempo_sync_layout.addWidget(QLabel("BPM:"))
        self.delay_bpm_spinbox = QSpinBox()
        self.delay_bpm_spinbox.setRange(40, 300)
        self.delay_bpm_spinbox.setValue(config.bpm)
        self.delay_bpm_spinbox.valueChanged.connect(self.update_delay_bpm)
        tempo_sync_layout.addWidget(self.delay_bpm_spinbox)

        # Computed delay time
        self.delay_ms_label = QLabel(f"{config.delay_time_ms:.1f} ms")
        tempo_sync_layout.addWidget(self.delay_ms_label)
        tempo_sync_layout.addStretch(1)

        # Delay time control
        time_label = QLabel("Time")
        time_label.setAlignment(Qt.AlignCenter)
        delay_layout.addWidget(time_label, 2, 0)

        self.delay_time_dial = QDial()
        self.delay_time_dial.setRange(10, 1000)
        self.delay_time_dial.setValue(int(config.delay_time_ms))
        self.delay_time_dial.valueChanged.connect(self.update_delay_time)
        self.delay_time_dial.setNotchesVisible(True)
        delay_layout.addWidget(self.delay_time_dial, 3, 0)

        self.delay_time_label = QLabel(f"{config.delay_time_ms:.0f} ms")
        self.delay_time_label.setAlignment(Qt.AlignCenter)
        delay_layout.addWidget(self.delay_time_label, 4, 0)

        # Delay feedback control
        feedback_label = QLabel("Feedback")
        feedback_label.setAlignment(Qt.AlignCenter)
        delay_layout.addWidget(feedback_label, 2, 1)

        self.delay_feedback_dial = QDial()
        self.delay_feedback_dial.setRange(0, 95)  # 0.0 to 0.95 (x100)
        self.delay_feedback_dial.setValue(int(config.delay_feedback * 100))
        self.delay_feedback_dial.valueChanged.connect(self.update_delay_feedback)
        self.delay_feedback_dial.setNotchesVisible(True)
        delay_layout.addWidget(self.delay_feedback_dial, 3, 1)

        self.delay_feedback_label = QLabel(f"{config.delay_feedback:.2f}")
        self.delay_feedback_label.setAlignment(Qt.AlignCenter)
        delay_layout.addWidget(self.delay_feedback_label, 4, 1)

        # Delay mix control
        mix_label = QLabel("Mix")
        mix_label.setAlignment(Qt.AlignCenter)
        delay_layout.addWidget(mix_label, 2, 2)

        self.delay_mix_dial = QDial()
        self.delay_mix_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.delay_mix_dial.setValue(int(config.delay_mix * 100))
        self.delay_mix_dial.valueChanged.connect(self.update_delay_mix)
        self.delay_mix_dial.setNotchesVisible(True)
        delay_layout.addWidget(self.delay_mix_dial, 3, 2)

        self.delay_mix_label = QLabel(f"{config.delay_mix:.2f}")
        self.delay_mix_label.setAlignment(Qt.AlignCenter)
        delay_layout.addWidget(self.delay_mix_label, 4, 2)

        # Update the visibility of sync controls based on initial state
        self.update_delay_sync_controls()

        # Chorus effect controls
        chorus_group = QGroupBox("Chorus Parameters")
        chorus_layout = QGridLayout(chorus_group)
        chorus_tab_layout.addWidget(chorus_group)

        # Chorus enable checkbox in top row
        control_row = QHBoxLayout()
        chorus_layout.addLayout(control_row, 0, 0, 1, 3)

        self.chorus_enable_checkbox = QCheckBox("Enable Chorus")
        self.chorus_enable_checkbox.setChecked(config.chorus_enabled)
        self.chorus_enable_checkbox.stateChanged.connect(self.update_chorus_enabled)
        control_row.addWidget(self.chorus_enable_checkbox)

        # Chorus rate control
        rate_label = QLabel("Rate")
        rate_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(rate_label, 1, 0)

        self.chorus_rate_dial = QDial()
        self.chorus_rate_dial.setRange(1, 100)  # 0.1 to 10.0 Hz (x10)
        self.chorus_rate_dial.setValue(int(config.chorus_rate * 10))
        self.chorus_rate_dial.valueChanged.connect(self.update_chorus_rate)
        self.chorus_rate_dial.setNotchesVisible(True)
        chorus_layout.addWidget(self.chorus_rate_dial, 2, 0)

        self.chorus_rate_label = QLabel(f"{config.chorus_rate:.1f} Hz")
        self.chorus_rate_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(self.chorus_rate_label, 3, 0)

        # Chorus depth control
        depth_label = QLabel("Depth")
        depth_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(depth_label, 1, 1)

        self.chorus_depth_dial = QDial()
        self.chorus_depth_dial.setRange(1, 30)  # 1 to 30 ms
        self.chorus_depth_dial.setValue(int(config.chorus_depth * 1000))
        self.chorus_depth_dial.valueChanged.connect(self.update_chorus_depth)
        self.chorus_depth_dial.setNotchesVisible(True)
        chorus_layout.addWidget(self.chorus_depth_dial, 2, 1)

        self.chorus_depth_label = QLabel(f"{config.chorus_depth * 1000:.1f} ms")
        self.chorus_depth_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(self.chorus_depth_label, 3, 1)

        # Chorus mix control
        mix_label = QLabel("Mix")
        mix_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(mix_label, 1, 2)

        self.chorus_mix_dial = QDial()
        self.chorus_mix_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.chorus_mix_dial.setValue(int(config.chorus_mix * 100))
        self.chorus_mix_dial.valueChanged.connect(self.update_chorus_mix)
        self.chorus_mix_dial.setNotchesVisible(True)
        chorus_layout.addWidget(self.chorus_mix_dial, 2, 2)

        self.chorus_mix_label = QLabel(f"{config.chorus_mix:.2f}")
        self.chorus_mix_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(self.chorus_mix_label, 3, 2)

        # Visualization toggle and visualizations at the bottom
        viz_toggle_layout = QHBoxLayout()
        main_layout.addLayout(viz_toggle_layout)

        self.viz_enable_checkbox = QCheckBox("Enable Waveform and Spectrum Visualization (requires more CPU)")
        self.viz_enable_checkbox.setChecked(self.visualization_enabled)
        self.viz_enable_checkbox.stateChanged.connect(self.update_visualization_enabled)
        viz_toggle_layout.addWidget(self.viz_enable_checkbox)

        # Create a layout for visualization frames
        viz_layout = QHBoxLayout()
        main_layout.addLayout(viz_layout, stretch=1)

        # Waveform visualization
        self.wave_viz_group = QGroupBox("Waveform Display")
        wave_viz_layout = QVBoxLayout(self.wave_viz_group)
        viz_layout.addWidget(self.wave_viz_group)

        # Set up pyqtgraph plot for waveform
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setLabel('left', 'Amplitude')
        self.wave_plot.setLabel('bottom', 'Samples')
        self.wave_plot.setTitle('Synth Output Waveform')
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        wave_viz_layout.addWidget(self.wave_plot)

        # Initialize waveform plot
        self.window_size = 512

        # Remove legend and unfiltered curve
        self.wave_curve = self.wave_plot.plot(
            np.zeros(self.window_size),
            pen=pg.mkPen('g', width=2)
        )
        self.wave_plot.setYRange(-1, 1)
        self.wave_plot.setXRange(0, 500)

        # Spectrum visualization
        self.spec_viz_group = QGroupBox("Frequency Spectrum")
        spec_viz_layout = QVBoxLayout(self.spec_viz_group)
        viz_layout.addWidget(self.spec_viz_group)

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

        # Update visibility based on initial state
        self.update_visualization_visibility()

        # Create the MIDI player tab
        midi_player_widget = QWidget()
        midi_player_layout = QVBoxLayout(midi_player_widget)
        envelope_tabs.addTab(midi_player_widget, "MIDI Player")

        # MIDI file selection
        midi_file_group = QGroupBox("MIDI File")
        midi_file_layout = QHBoxLayout(midi_file_group)
        midi_player_layout.addWidget(midi_file_group)

        self.midi_file_label = QLabel("No file selected")
        midi_file_layout.addWidget(self.midi_file_label, stretch=1)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_midi_file)
        midi_file_layout.addWidget(browse_button)

        # Playback controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout(playback_group)
        midi_player_layout.addWidget(playback_group)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_midi)
        self.play_button.setEnabled(False)
        playback_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_midi)
        self.pause_button.setEnabled(False)
        playback_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_midi)
        self.stop_button.setEnabled(False)
        playback_layout.addWidget(self.stop_button)

        # Tempo control
        tempo_group = QGroupBox("Tempo")
        tempo_layout = QHBoxLayout(tempo_group)
        midi_player_layout.addWidget(tempo_group)

        tempo_layout.addWidget(QLabel("Tempo Scale:"))

        self.tempo_slider = QSlider(Qt.Horizontal)
        self.tempo_slider.setRange(50, 200)  # 0.5x to 2.0x
        self.tempo_slider.setValue(100)      # Default 1.0x
        self.tempo_slider.valueChanged.connect(self.update_midi_tempo)
        tempo_layout.addWidget(self.tempo_slider, stretch=1)

        self.tempo_label = QLabel("1.00x")
        tempo_layout.addWidget(self.tempo_label)

        # Progress display
        progress_group = QGroupBox("Playback Progress")
        progress_layout = QVBoxLayout(progress_group)
        midi_player_layout.addWidget(progress_group)

        self.midi_progress_bar = QProgressBar()
        self.midi_progress_bar.setRange(0, 100)
        self.midi_progress_bar.setValue(0)
        progress_layout.addWidget(self.midi_progress_bar)

        # Status label
        self.midi_status_label = QLabel("Ready")
        progress_layout.addWidget(self.midi_status_label)

        # Add spacer to push controls to the top
        midi_player_layout.addStretch(1)

    def start_animation(self):
        """Start the QTimer for updating plots."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)  # Update every 30ms

        # Initialize delay time from BPM
        self.update_delay_from_bpm()

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
        if self.cutoff_dial.value() != filter.cutoff:
            self.cutoff_dial.setValue(int(filter.cutoff))
            self.cutoff_label.setText(f"{filter.cutoff:.0f} Hz")

        # Check if filter resonance has changed and update GUI if needed
        if self.resonance_dial.value() != int(filter.resonance * 100):
            self.resonance_dial.setValue(int(filter.resonance * 100))
            self.resonance_label.setText(f"{filter.resonance:.2f}")

        # Check if filter envelope amount has changed and update GUI
        if self.filter_env_amount_dial.value() != adsr.filter_env_amount:
            self.filter_env_amount_dial.setValue(int(adsr.filter_env_amount))
            self.filter_env_amount_label.setText(f"{adsr.filter_env_amount:.0f} Hz")

        # Check if LFO parameters have changed and update GUI if needed
        if self.lfo_rate_dial.value() != int(config.lfo_rate * 10):
            self.lfo_rate_dial.setValue(int(config.lfo_rate * 10))
            self.lfo_rate_label.setText(f"{config.lfo_rate:.1f} Hz")

        if self.lfo_depth_dial.value() != int(config.lfo_depth * 100):
            self.lfo_depth_dial.setValue(int(config.lfo_depth * 100))
            self.lfo_depth_label.setText(f"{config.lfo_depth:.2f}")

        if self.lfo_attack_dial.value() != int(config.lfo_attack_time * 10):
            self.lfo_attack_dial.setValue(int(config.lfo_attack_time * 10))
            self.lfo_attack_label.setText(f"{config.lfo_attack_time:.1f} s")

        if self.lfo_delay_dial.value() != int(config.lfo_delay_time * 10):
            self.lfo_delay_dial.setValue(int(config.lfo_delay_time * 10))
            self.lfo_delay_label.setText(f"{config.lfo_delay_time:.1f} s")

        # Check if LFO enabled status has changed and update GUI if needed
        if self.lfo_enable_checkbox.isChecked() != config.lfo_enabled:
            self.lfo_enable_checkbox.setChecked(config.lfo_enabled)

        # Check if filter enabled status has changed and update GUI if needed
        if self.filter_enable_checkbox.isChecked() != filter.filter_enabled:
            self.filter_enable_checkbox.setChecked(filter.filter_enabled)

        # Check if drive settings have changed and update GUI if needed
        if self.drive_enable_checkbox.isChecked() != config.drive_on:
            self.drive_enable_checkbox.setChecked(config.drive_on)

        if self.drive_dial.value() != int(config.drive_gain * 100):
            self.drive_dial.setValue(int(config.drive_gain * 100))
            self.drive_label.setText(f"{config.drive_gain:.1f}")

        if self.drive_type_combo.currentText() != config.drive_type:
            self.drive_type_combo.setCurrentText(config.drive_type)

        if self.drive_tone_dial.value() != int(config.drive_tone * 100):
            self.drive_tone_dial.setValue(int(config.drive_tone * 100))
            self.drive_tone_label.setText(f"{config.drive_tone:.2f}")

        if self.drive_mix_dial.value() != int(config.drive_mix * 100):
            self.drive_mix_dial.setValue(int(config.drive_mix * 100))
            self.drive_mix_label.setText(f"{config.drive_mix:.2f}")

        if self.drive_asym_dial.value() != int(config.drive_asymmetry * 100):
            self.drive_asym_dial.setValue(int(config.drive_asymmetry * 100))
            self.drive_asym_label.setText(f"{config.drive_asymmetry:.2f}")

        # Check if delay parameters have changed and update GUI if needed
        if self.delay_enable_checkbox.isChecked() != config.delay_enabled:
            self.delay_enable_checkbox.setChecked(config.delay_enabled)
        if self.delay_sync_checkbox.isChecked() != config.delay_sync_enabled:
            self.delay_sync_checkbox.setChecked(config.delay_sync_enabled)
            self.update_delay_sync_controls()

        if self.delay_time_dial.value() != int(config.delay_time_ms):
            self.delay_time_dial.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

        if self.delay_feedback_dial.value() != int(config.delay_feedback * 100):
            self.delay_feedback_dial.setValue(int(config.delay_feedback * 100))
            self.delay_feedback_label.setText(f"{config.delay_feedback:.2f}")

        if self.delay_mix_dial.value() != int(config.delay_mix * 100):
            self.delay_mix_dial.setValue(int(config.delay_mix * 100))
            self.delay_mix_label.setText(f"{config.delay_mix:.2f}")

        if self.delay_division_combo.currentText() != config.delay_division:
            self.delay_division_combo.setCurrentText(config.delay_division)

        if self.delay_bpm_spinbox.value() != config.bpm:
            self.delay_bpm_spinbox.setValue(config.bpm)

        # Check if ADSR parameters have changed and update GUI if needed
        adsr_changed = False

        if self.attack_dial.value() != int(adsr.adsr['attack'] * 100):
            self.attack_dial.setValue(int(adsr.adsr['attack'] * 100))
            self.attack_label.setText(f"{adsr.adsr['attack']:.2f} s")
            adsr_changed = True

        if self.decay_dial.value() != int(adsr.adsr['decay'] * 100):
            self.decay_dial.setValue(int(adsr.adsr['decay'] * 100))
            self.decay_label.setText(f"{adsr.adsr['decay']:.2f} s")
            adsr_changed = True

        if self.sustain_dial.value() != int(adsr.adsr['sustain'] * 100):
            self.sustain_dial.setValue(int(adsr.adsr['sustain'] * 100))
            self.sustain_label.setText(f"{adsr.adsr['sustain']:.2f}")
            adsr_changed = True

        if self.release_dial.value() != int(adsr.adsr['release'] * 100):
            self.release_dial.setValue(int(adsr.adsr['release'] * 100))
            self.release_label.setText(f"{adsr.adsr['release']:.2f} s")
            adsr_changed = True

        # Check if filter ADSR parameters have changed
        filter_adsr_changed = False

        if self.filter_attack_dial.value() != int(adsr.filter_adsr['attack'] * 100):
            self.filter_attack_dial.setValue(int(adsr.filter_adsr['attack'] * 100))
            self.filter_attack_label.setText(f"{adsr.filter_adsr['attack']:.2f} s")
            filter_adsr_changed = True

        if self.filter_decay_dial.value() != int(adsr.filter_adsr['decay'] * 100):
            self.filter_decay_dial.setValue(int(adsr.filter_adsr['decay'] * 100))
            self.filter_decay_label.setText(f"{adsr.filter_adsr['decay']:.2f} s")
            filter_adsr_changed = True

        if self.filter_sustain_dial.value() != int(adsr.filter_adsr['sustain'] * 100):
            self.filter_sustain_dial.setValue(int(adsr.filter_adsr['sustain'] * 100))
            self.filter_sustain_label.setText(f"{adsr.filter_adsr['sustain']:.2f}")
            filter_adsr_changed = True

        if self.filter_release_dial.value() != int(adsr.filter_adsr['release'] * 100):
            self.filter_release_dial.setValue(int(adsr.filter_adsr['release'] * 100))
            self.filter_release_label.setText(f"{adsr.filter_adsr['release']:.2f} s")
            filter_adsr_changed = True

        # Check if volume has changed
        if self.volume_dial.value() != int(config.volume * 100):
            self.volume_dial.setValue(int(config.volume * 100))
            self.volume_label.setText(f"{config.volume:.2f}")

        # Check if delay ping-pong status has changed and update GUI if needed
        if self.pingpong_checkbox.isChecked() != config.delay_pingpong:
            self.pingpong_checkbox.setChecked(config.delay_pingpong)

        # Check if chorus parameters have changed and update GUI if needed
        if self.chorus_enable_checkbox.isChecked() != config.chorus_enabled:
            self.chorus_enable_checkbox.setChecked(config.chorus_enabled)

        if self.chorus_rate_dial.value() != int(config.chorus_rate * 10):
            self.chorus_rate_dial.setValue(int(config.chorus_rate * 10))
            self.chorus_rate_label.setText(f"{config.chorus_rate:.1f} Hz")

        if self.chorus_depth_dial.value() != int(config.chorus_depth * 1000):
            self.chorus_depth_dial.setValue(int(config.chorus_depth * 1000))
            self.chorus_depth_label.setText(f"{config.chorus_depth * 1000:.1f} ms")

        if self.chorus_mix_dial.value() != int(config.chorus_mix * 100):
            self.chorus_mix_dial.setValue(int(config.chorus_mix * 100))
            self.chorus_mix_label.setText(f"{config.chorus_mix:.2f}")

        # Update ADSR curves if parameters changed
        if adsr_changed:
            self.plot_adsr_curve()

        if filter_adsr_changed:
            self.plot_filter_adsr_curve()

        # Skip waveform and spectrum processing if visualization is disabled
        if not self.visualization_enabled:
            return

        # Get current audio buffer data
        with config.buffer_lock:
            filtered_data = config.waveform_buffer.copy()

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

        self.wave_curve.setData(np.arange(len(filtered_segment)), filtered_segment)

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

    def update_drive_enabled(self, state):
        """Update the drive enabled setting."""
        config.drive_on = (state == Qt.Checked)

    def update_drive_gain(self, value):
        """Update the drive gain setting."""
        gain = value / 100.0
        config.drive_gain = gain
        self.drive_label.setText(f"{gain:.1f}")

    def update_drive_type(self, drive_type):
        """Update the drive type setting."""
        config.drive_type = drive_type

    def update_drive_tone(self, value):
        """Update the drive tone setting."""
        tone = value / 100.0
        config.drive_tone = tone
        self.drive_tone_label.setText(f"{tone:.2f}")

    def update_drive_mix(self, value):
        """Update the drive mix setting."""
        mix = value / 100.0
        config.drive_mix = mix
        self.drive_mix_label.setText(f"{mix:.2f}")

    def update_drive_asymmetry(self, value):
        """Update the drive asymmetry setting."""
        asymmetry = value / 100.0
        config.drive_asymmetry = asymmetry
        self.drive_asym_label.setText(f"{asymmetry:.2f}")

    def update_delay_enabled(self, state):
        """Update the delay enabled setting."""
        config.delay_enabled = (state == Qt.Checked)
        if not config.delay_enabled:
            synth.delay.clear_cache()

    def update_delay_sync(self, state):
        """Update whether delay time is synced to BPM."""
        config.delay_sync_enabled = (state == Qt.Checked)
        self.update_delay_sync_controls()

        # If enabling sync, update delay time from BPM
        if config.delay_sync_enabled:
            self.update_delay_from_bpm()
            self.delay_time_dial.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

    def update_delay_sync_controls(self):
        """Show/hide appropriate controls based on sync setting."""
        # Show/hide sync controls based on sync checkbox state
        self.tempo_sync_frame.setVisible(config.delay_sync_enabled)

        # Enable/disable manual time dial based on sync setting
        self.delay_time_dial.setEnabled(not config.delay_sync_enabled)

    def update_delay_time(self, value):
        """Update the delay time setting."""
        # Only update if we're not in sync mode
        if not config.delay_sync_enabled:
            config.delay_time_ms = float(value)
            synth.delay.set_time(value)
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
            self.update_delay_from_bpm()

            # Update displayed values
            self.delay_time_dial.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

    def update_delay_from_bpm(self):
        """Update delay time based on BPM and selected division."""
        synth.delay.update_delay_from_bpm(config.bpm, config.delay_division)

    def update_delay_bpm(self, bpm):
        """Update the global BPM and recalculate delay time."""
        config.bpm = bpm

        if config.delay_sync_enabled:
            self.update_delay_from_bpm()

            # Update displayed values
            self.delay_time_dial.setValue(int(config.delay_time_ms))
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
            self.update_delay_from_bpm()

            # Update displayed delay time values
            self.delay_time_dial.setValue(int(config.delay_time_ms))
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

    def update_delay_pingpong(self, state):
        """Update the delay ping-pong setting."""
        config.delay_pingpong = (state == Qt.Checked)
        if config.delay_enabled:
            synth.delay.clear_cache()  # Clear buffers to avoid artifacts when switching modes

    def update_chorus_enabled(self, state):
        """Update the chorus enabled setting."""
        config.chorus_enabled = (state == Qt.Checked)
        if not config.chorus_enabled:
            synth.chorus.clear_cache()  # Clear buffers when disabling

    def update_chorus_rate(self, value):
        """Update the chorus rate setting."""
        rate = value / 10.0  # Convert from slider value to Hz
        config.chorus_rate = rate
        synth.chorus.set_rate(rate)
        self.chorus_rate_label.setText(f"{rate:.1f} Hz")

    def update_chorus_depth(self, value):
        """Update the chorus depth setting."""
        depth = value / 1000.0  # Convert from ms to seconds
        config.chorus_depth = depth
        synth.chorus.set_depth(depth)
        self.chorus_depth_label.setText(f"{value:.1f} ms")

    def update_chorus_mix(self, value):
        """Update the chorus mix setting."""
        mix = value / 100.0
        config.chorus_mix = mix
        synth.chorus.set_mix(mix)
        self.chorus_mix_label.setText(f"{config.chorus_mix:.2f}")

    def update_visualization_enabled(self, state):
        """Update visualization enabled status and visibility."""
        self.visualization_enabled = (state == Qt.Checked)
        self.update_visualization_visibility()

    def update_visualization_visibility(self):
        """Show or hide visualization based on enabled status."""
        self.wave_viz_group.setVisible(self.visualization_enabled)
        self.spec_viz_group.setVisible(self.visualization_enabled)

    def browse_midi_file(self):
        """Open a file dialog to select a MIDI file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI File", "", "MIDI Files (*.mid *.midi);;All Files (*)"
        )

        if file_path:
            self.midi_file_path = file_path
            # Display just the filename, not the full path
            self.midi_file_label.setText(os.path.basename(file_path))
            self.play_button.setEnabled(True)
            self.midi_status_label.setText("Ready to play")
            self.midi_progress_bar.setValue(0)

            # If already playing, stop
            if self.midi_player_active:
                self.stop_midi()

    def play_midi(self):
        """Play the selected MIDI file."""
        if not self.midi_file_path:
            return

        # If paused, resume
        if self.midi_paused:
            config.midi_paused = False
            self.midi_paused = False
            self.midi_status_label.setText("Playing")
            return

        # Start new playback
        self.stop_midi()  # Ensure any previous playback is stopped

        # Get tempo scale from slider
        tempo_scale = self.tempo_slider.value() / 100.0

        try:
            # Set up playback state
            self.midi_player_active = True
            self.midi_paused = False
            config.midi_paused = False

            # Update UI
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.midi_status_label.setText("Playing")

            # Start MIDI playback (non-blocking)
            import threading
            self.midi_playback_thread = threading.Thread(
                target=self._play_midi_thread,
                args=(self.midi_file_path, tempo_scale),
                daemon=True
            )
            self.midi_playback_thread.start()

            # Start progress update timer
            self.midi_progress_timer = QTimer()
            self.midi_progress_timer.timeout.connect(self.update_midi_progress)
            self.midi_progress_timer.start(100)  # Update every 100ms

        except Exception as e:
            self.midi_status_label.setText(f"Error: {str(e)}")
            self.midi_player_active = False
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def _play_midi_thread(self, file_path, tempo_scale):
        """Background thread for MIDI playback."""
        try:
            # Store start time for progress calculation
            config.midi_playback_start_time = time.time()
            config.midi_playback_active = True

            # Play the MIDI file
            play_midi_file(file_path, tempo_scale)

            # Don't immediately reset state as the playback runs asynchronously
            # It will be reset when the playback completes or is stopped
        except Exception as e:
            print(f"MIDI playback error: {e}")
            config.midi_playback_active = False

    def pause_midi(self):
        """Pause the MIDI playback."""
        if self.midi_player_active and not self.midi_paused:
            self.midi_paused = True
            config.midi_paused = True
            self.midi_status_label.setText("Paused")
            self.play_button.setEnabled(True)

    def stop_midi(self):
        """Stop the MIDI playback."""
        if self.midi_player_active:
            # Signal to stop playback
            config.midi_playback_active = False
            self.midi_player_active = False
            self.midi_paused = False
            config.midi_paused = False

            # Reset UI
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.midi_status_label.setText("Stopped")
            self.midi_progress_bar.setValue(0)

            # Stop progress timer if it's running
            if hasattr(self, 'midi_progress_timer') and self.midi_progress_timer.isActive():
                self.midi_progress_timer.stop()

    def update_midi_tempo(self, value):
        """Update the MIDI playback tempo."""
        tempo_scale = value / 100.0
        self.tempo_label.setText(f"{tempo_scale:.2f}x")
        config.midi_tempo_scale = tempo_scale

    def update_midi_progress(self):
        """Update the MIDI playback progress bar."""
        # Check if playback is still active
        if not self.midi_player_active:
            self.midi_progress_timer.stop()
            return

        # If playback has completed naturally
        if not config.midi_playback_active:
            self.stop_midi()
            self.midi_status_label.setText("Completed")
            return

        # Calculate rough progress based on time (this is approximate)
        # For a better implementation, the controller would need to report actual progress
        if hasattr(config, 'midi_playback_duration') and config.midi_playback_duration > 0:
            elapsed = time.time() - config.midi_playback_start_time
            if self.midi_paused:
                # Don't update progress while paused
                return

            progress = min(100, int((elapsed / config.midi_playback_duration) * 100))
            self.midi_progress_bar.setValue(progress)

    def closeEvent(self, event):
        """Clean up when window is closed."""
        self.animation_running = False
        if hasattr(self, 'timer'):
            self.timer.stop()

        # Stop MIDI playback if active
        if self.midi_player_active:
            self.stop_midi()

        # Stop sequencer if running
        if self.sequencer:
            self.sequencer.stop()

        event.accept()


def start_gui():
    """Start the GUI and synth components."""
    app = QApplication(sys.argv)

    # Apply QDarkStyle dark theme
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Apply dark theme for pyqtgraph
    pg.setConfigOption('background', '#353535')  # Dark gray background
    pg.setConfigOption('foreground', 'w')  # White foreground
    pg.setConfigOption('antialias', True)

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
