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
    QFileDialog, QProgressBar, QDial, QListWidget, QListWidgetItem, QLineEdit,
    QMessageBox, QInputDialog, QShortcut
)
from PyQt5.QtGui import QKeySequence
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
from qwerty_synth import record
from qwerty_synth import patch
from qwerty_synth.arpeggiator import Arpeggiator
from qwerty_synth import arpeggiator

# Global variable to hold reference to the GUI instance
gui_instance = None

class SynthGUI(QMainWindow):
    """GUI for controlling QWERTY Synth parameters using PyQt."""

    # Stylesheet for toggle buttons
    TOGGLE_BUTTON_STYLE = """
        QPushButton {
            background-color: #353535;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:checked {
            background-color: #2d7d32;
            border: 1px solid #4caf50;
        }
        QPushButton:hover {
            border: 1px solid #777777;
        }
        QPushButton:checked:hover {
            border: 1px solid #66bb6a;
        }
    """

    def __init__(self):
        """Initialize the GUI with PyQt5."""
        super().__init__()

        self.setWindowTitle("QWERTY Synth")
        self.setGeometry(100, 100, 1200, 900)  # x, y, width, height - increased height for sequencer

        # Animation control variables
        self.animation_running = True
        self.visualization_enabled = True

        # Keep track of the last GUI update time
        self.last_update_time = time.time()

        # Create sequencer instance
        self.sequencer = None  # Will be initialized in setup_ui

        # MIDI player state
        self.midi_player_active = False
        self.midi_file_path = None
        self.midi_playback_thread = None
        self.midi_paused = False

        # Current patch name
        self.current_patch_name = "Untitled"
        self.current_patch_path = None

        # Set up the user interface
        self.setup_ui()
        self.running = True

        # Start animation timer after UI is set up
        self.start_animation()

        # Handle window close event
        self.showMaximized()  # Start maximized but with window controls

        # Set up keyboard shortcuts
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for patch management."""
        # Ctrl+S for quick save
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.quick_save_patch)

        # Ctrl+L for load
        self.load_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.load_shortcut.activated.connect(self.show_load_patch_dialog)

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

        # Transpose Control (Octave + Semitone)
        transpose_group = QGroupBox("Transpose Control")
        transpose_layout = QGridLayout(transpose_group)
        control_layout.addWidget(transpose_group, stretch=1)

        # Octave control
        octave_label = QLabel("Octave")
        octave_label.setAlignment(Qt.AlignCenter)
        transpose_layout.addWidget(octave_label, 0, 0)

        self.octave_dial = QDial()
        self.octave_dial.setRange(config.octave_min, config.octave_max)
        self.octave_dial.setValue(config.octave_offset // 12)
        self.octave_dial.valueChanged.connect(self.update_octave)
        self.octave_dial.setNotchesVisible(True)
        self.octave_dial.setToolTip("Transpose by octaves (Z/X keys)")
        transpose_layout.addWidget(self.octave_dial, 1, 0)

        self.octave_label = QLabel(f"{config.octave_offset // 12:+d}")
        self.octave_label.setAlignment(Qt.AlignCenter)
        transpose_layout.addWidget(self.octave_label, 2, 0)

        # Semitone control
        semitone_label = QLabel("Semitone")
        semitone_label.setAlignment(Qt.AlignCenter)
        transpose_layout.addWidget(semitone_label, 0, 1)

        self.semitone_dial = QDial()
        self.semitone_dial.setRange(config.semitone_min, config.semitone_max)
        self.semitone_dial.setValue(config.semitone_offset)
        self.semitone_dial.valueChanged.connect(self.update_semitone)
        self.semitone_dial.setNotchesVisible(True)
        self.semitone_dial.setToolTip("Transpose by semitones")
        transpose_layout.addWidget(self.semitone_dial, 1, 1)

        self.semitone_label = QLabel(f"{config.semitone_offset:+d}")
        self.semitone_label.setAlignment(Qt.AlignCenter)
        transpose_layout.addWidget(self.semitone_label, 2, 1)

        # Mono Mode and Portamento Controls
        mono_group = QGroupBox("Mono Mode")
        mono_layout = QHBoxLayout(mono_group)
        control_layout.addWidget(mono_group, stretch=1)

        # Mono mode toggle button
        self.mono_button = QPushButton("Mono Mode")
        self.mono_button.setCheckable(True)
        self.mono_button.setChecked(config.mono_mode)
        self.mono_button.clicked.connect(self.update_mono_mode)
        self.mono_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        mono_layout.addWidget(self.mono_button)

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

        # Filter enable toggle button
        self.filter_enable_button = QPushButton("Enable Filter")
        self.filter_enable_button.setCheckable(True)
        self.filter_enable_button.setChecked(config.filter_enabled)
        self.filter_enable_button.clicked.connect(self.update_filter_enabled)
        self.filter_enable_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        filter_layout.addWidget(self.filter_enable_button, 0, 0, 1, 1)

        # Filter type selector
        filter_layout.addWidget(QLabel("Type:"), 0, 1)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["lowpass", "highpass", "bandpass", "notch"])
        self.filter_type_combo.setCurrentText(config.filter_type)
        self.filter_type_combo.currentTextChanged.connect(self.update_filter_type)
        filter_layout.addWidget(self.filter_type_combo, 0, 2)

        # Filter topology selector
        filter_layout.addWidget(QLabel("Topology:"), 1, 3)
        self.filter_topology_combo = QComboBox()
        self.filter_topology_combo.addItems(["svf", "biquad"])
        self.filter_topology_combo.setCurrentText(config.filter_topology)
        self.filter_topology_combo.currentTextChanged.connect(self.update_filter_topology)
        filter_layout.addWidget(self.filter_topology_combo, 1, 4)

        # Filter slope selector (only for biquad)
        filter_layout.addWidget(QLabel("Slope:"), 2, 3)
        self.filter_slope_combo = QComboBox()
        self.filter_slope_combo.addItems(["12 dB/oct", "24 dB/oct"])
        self.filter_slope_combo.setCurrentText("24 dB/oct" if config.filter_slope == 24 else "12 dB/oct")
        self.filter_slope_combo.currentTextChanged.connect(self.update_filter_slope)
        filter_layout.addWidget(self.filter_slope_combo, 2, 4)

        # Enable/disable slope control based on initial topology
        self.filter_slope_combo.setEnabled(config.filter_topology == 'biquad')

        # Cutoff control
        cutoff_label = QLabel("Cutoff")
        cutoff_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(cutoff_label, 1, 0)

        self.cutoff_dial = QDial()
        self.cutoff_dial.setRange(20, 20000)
        self.cutoff_dial.setValue(int(config.filter_cutoff))
        self.cutoff_dial.valueChanged.connect(self.update_filter_cutoff)
        self.cutoff_dial.setNotchesVisible(True)
        filter_layout.addWidget(self.cutoff_dial, 2, 0)

        self.cutoff_label = QLabel(f"{config.filter_cutoff:.0f} Hz")
        self.cutoff_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.cutoff_label, 3, 0)

        # Resonance control
        res_label = QLabel("Resonance")
        res_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(res_label, 1, 1)

        self.resonance_dial = QDial()
        self.resonance_dial.setRange(0, 95)  # 0.0 to 0.95 (x100)
        self.resonance_dial.setValue(int(config.filter_resonance * 100))
        self.resonance_dial.valueChanged.connect(self.update_filter_resonance)
        self.resonance_dial.setNotchesVisible(True)
        filter_layout.addWidget(self.resonance_dial, 2, 1)

        self.resonance_label = QLabel(f"{config.filter_resonance:.2f}")
        self.resonance_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.resonance_label, 3, 1)

        # Filter envelope amount control
        env_label = QLabel("Env Amount")
        env_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(env_label, 1, 2)

        self.filter_env_amount_dial = QDial()
        self.filter_env_amount_dial.setRange(0, 10000)
        self.filter_env_amount_dial.setValue(int(config.filter_env_amount))
        self.filter_env_amount_dial.valueChanged.connect(self.update_filter_env_amount)
        self.filter_env_amount_dial.setNotchesVisible(True)
        filter_layout.addWidget(self.filter_env_amount_dial, 2, 2)

        self.filter_env_amount_label = QLabel(f"{config.filter_env_amount:.0f} Hz")
        self.filter_env_amount_label.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.filter_env_amount_label, 3, 2)

        # Drive Control
        drive_group = QGroupBox("Drive Control")
        drive_layout = QGridLayout(drive_group)
        filter_drive_layout.addWidget(drive_group)

        # Drive enable toggle button
        self.drive_enable_button = QPushButton("Enable Drive")
        self.drive_enable_button.setCheckable(True)
        self.drive_enable_button.setChecked(config.drive_on)
        self.drive_enable_button.clicked.connect(self.update_drive_enabled)
        self.drive_enable_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        drive_layout.addWidget(self.drive_enable_button, 0, 0, 1, 1)

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

        # Create the arpeggiator tab
        self.arpeggiator = Arpeggiator()
        envelope_tabs.addTab(self.arpeggiator, "Arpeggiator")

        # Set the global arpeggiator instance for input handling
        arpeggiator.arpeggiator_instance = self.arpeggiator

        # Create the patch management tab - add it after Arpeggiator
        patches_widget = QWidget()
        patches_layout = QVBoxLayout(patches_widget)
        envelope_tabs.addTab(patches_widget, "Patches")

        # Create the patch list
        patch_list_group = QGroupBox("Available Patches")
        patch_list_layout = QVBoxLayout(patch_list_group)
        patches_layout.addWidget(patch_list_group)

        # Search box
        search_layout = QHBoxLayout()
        patch_list_layout.addLayout(search_layout)
        search_layout.addWidget(QLabel("Search:"))
        self.patch_search = QLineEdit()
        self.patch_search.setPlaceholderText("Filter patches...")
        self.patch_search.textChanged.connect(self.filter_patches)
        search_layout.addWidget(self.patch_search)

        # List widget
        self.patch_list = QListWidget()
        self.patch_list.itemDoubleClicked.connect(self.load_selected_patch)
        patch_list_layout.addWidget(self.patch_list)

        # Current patch info
        current_patch_layout = QHBoxLayout()
        patch_list_layout.addLayout(current_patch_layout)
        current_patch_layout.addWidget(QLabel("Current Patch:"))
        self.current_patch_label = QLabel(self.current_patch_name)
        current_patch_layout.addWidget(self.current_patch_label, stretch=1)

        # Patch management buttons
        button_layout = QHBoxLayout()
        patch_list_layout.addLayout(button_layout)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_patch)
        button_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_selected_patch)
        button_layout.addWidget(self.load_button)

        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_patch)
        button_layout.addWidget(self.rename_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_patch)
        button_layout.addWidget(self.delete_button)

        # Populate the patch list
        self.refresh_patch_list()

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
        self.attack_dial.setValue(int(config.adsr['attack'] * 100))
        self.attack_dial.valueChanged.connect(
            lambda v: self.update_adsr('attack', v/100.0)
        )
        self.attack_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.attack_dial, 1, 0)

        self.attack_label = QLabel(f"{config.adsr['attack']:.2f} s")
        self.attack_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.attack_label, 2, 0)

        # Decay control
        decay_label = QLabel("Decay")
        decay_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(decay_label, 0, 1)

        self.decay_dial = QDial()
        self.decay_dial.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.decay_dial.setValue(int(config.adsr['decay'] * 100))
        self.decay_dial.valueChanged.connect(
            lambda v: self.update_adsr('decay', v/100.0)
        )
        self.decay_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.decay_dial, 1, 1)

        self.decay_label = QLabel(f"{config.adsr['decay']:.2f} s")
        self.decay_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.decay_label, 2, 1)

        # Sustain control
        sustain_label = QLabel("Sustain")
        sustain_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(sustain_label, 0, 2)

        self.sustain_dial = QDial()
        self.sustain_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.sustain_dial.setValue(int(config.adsr['sustain'] * 100))
        self.sustain_dial.valueChanged.connect(
            lambda v: self.update_adsr('sustain', v/100.0)
        )
        self.sustain_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.sustain_dial, 1, 2)

        self.sustain_label = QLabel(f"{config.adsr['sustain']:.2f}")
        self.sustain_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.sustain_label, 2, 2)

        # Release control
        release_label = QLabel("Release")
        release_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(release_label, 0, 3)

        self.release_dial = QDial()
        self.release_dial.setRange(1, 300)  # 0.01 to 3.0 seconds (x100)
        self.release_dial.setValue(int(config.adsr['release'] * 100))
        self.release_dial.valueChanged.connect(
            lambda v: self.update_adsr('release', v/100.0)
        )
        self.release_dial.setNotchesVisible(True)
        adsr_controls.addWidget(self.release_dial, 1, 3)

        self.release_label = QLabel(f"{config.adsr['release']:.2f} s")
        self.release_label.setAlignment(Qt.AlignCenter)
        adsr_controls.addWidget(self.release_label, 2, 3)

        # ADSR visualization for amplitude
        adsr_viz_group = QGroupBox("ADSR Curves")
        adsr_viz_layout = QVBoxLayout(adsr_viz_group)
        amp_env_layout.addWidget(adsr_viz_group)

        # Set up pyqtgraph plot for ADSR curves (both amp and filter)
        self.adsr_plot = pg.PlotWidget()
        self.adsr_plot.setLabel('left', 'Amplitude')
        self.adsr_plot.setLabel('bottom', 'Time (normalized)')
        self.adsr_plot.showGrid(x=True, y=True, alpha=0.3)
        adsr_viz_layout.addWidget(self.adsr_plot)

        # Initialize ADSR curves plot
        self.adsr_curve = self.adsr_plot.plot(
            np.arange(len(adsr.adsr_curve)),
            adsr.adsr_curve,
            pen=pg.mkPen((100, 200, 255), width=2),
            name="Amp Envelope"
        )

        # Add filter ADSR curve to the same plot
        self.filter_adsr_curve = self.adsr_plot.plot(
            np.arange(len(adsr.filter_adsr_curve)),
            adsr.filter_adsr_curve,
            pen=pg.mkPen('r', width=2),
            name="Filter Envelope"
        )

        # Add legend to distinguish between curves
        self.adsr_plot.addLegend()

        self.adsr_plot.setYRange(0, 1.1)
        self.adsr_plot.setXRange(0, len(adsr.adsr_curve))

        # Filter ADSR visualization - Create an empty QFrame as a spacer
        filter_adsr_viz_group = QFrame()
        filter_adsr_viz_layout = QVBoxLayout(filter_adsr_viz_group)
        filter_env_layout.addWidget(filter_adsr_viz_group)

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
        self.filter_attack_dial.setValue(int(config.filter_adsr['attack'] * 100))
        self.filter_attack_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('attack', v/100.0)
        )
        self.filter_attack_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_attack_dial, 1, 0)

        self.filter_attack_label = QLabel(f"{config.filter_adsr['attack']:.2f} s")
        self.filter_attack_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_attack_label, 2, 0)

        # Filter Decay control
        filter_decay_label = QLabel("Decay")
        filter_decay_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_decay_label, 0, 1)

        self.filter_decay_dial = QDial()
        self.filter_decay_dial.setRange(1, 200)  # 0.01 to 2.0 seconds (x100)
        self.filter_decay_dial.setValue(int(config.filter_adsr['decay'] * 100))
        self.filter_decay_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('decay', v/100.0)
        )
        self.filter_decay_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_decay_dial, 1, 1)

        self.filter_decay_label = QLabel(f"{config.filter_adsr['decay']:.2f} s")
        self.filter_decay_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_decay_label, 2, 1)

        # Filter Sustain control
        filter_sustain_label = QLabel("Sustain")
        filter_sustain_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_sustain_label, 0, 2)

        self.filter_sustain_dial = QDial()
        self.filter_sustain_dial.setRange(0, 100)  # 0.0 to 1.0 (x100)
        self.filter_sustain_dial.setValue(int(config.filter_adsr['sustain'] * 100))
        self.filter_sustain_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('sustain', v/100.0)
        )
        self.filter_sustain_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_sustain_dial, 1, 2)

        self.filter_sustain_label = QLabel(f"{config.filter_adsr['sustain']:.2f}")
        self.filter_sustain_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_sustain_label, 2, 2)

        # Filter Release control
        filter_release_label = QLabel("Release")
        filter_release_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(filter_release_label, 0, 3)

        self.filter_release_dial = QDial()
        self.filter_release_dial.setRange(1, 300)  # 0.01 to 3.0 seconds (x100)
        self.filter_release_dial.setValue(int(config.filter_adsr['release'] * 100))
        self.filter_release_dial.valueChanged.connect(
            lambda v: self.update_filter_adsr('release', v/100.0)
        )
        self.filter_release_dial.setNotchesVisible(True)
        filter_adsr_controls.addWidget(self.filter_release_dial, 1, 3)

        self.filter_release_label = QLabel(f"{config.filter_adsr['release']:.2f} s")
        self.filter_release_label.setAlignment(Qt.AlignCenter)
        filter_adsr_controls.addWidget(self.filter_release_label, 2, 3)

        # LFO Controls
        lfo_group = QGroupBox("LFO Parameters")
        lfo_layout = QGridLayout(lfo_group)
        lfo_tab_layout.addWidget(lfo_group)

        # LFO Enable checkbox and Target selector in top row
        control_row = QHBoxLayout()
        lfo_layout.addLayout(control_row, 0, 0, 1, 4)

        self.lfo_enable_button = QPushButton("Enable LFO")
        self.lfo_enable_button.setCheckable(True)
        self.lfo_enable_button.setChecked(config.lfo_enabled)
        self.lfo_enable_button.clicked.connect(self.update_lfo_enabled)
        self.lfo_enable_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        control_row.addWidget(self.lfo_enable_button)

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

        # Delay enable toggle button
        self.delay_enable_button = QPushButton("Enable Delay")
        self.delay_enable_button.setCheckable(True)
        self.delay_enable_button.setChecked(config.delay_enabled)
        self.delay_enable_button.clicked.connect(self.update_delay_enabled)
        self.delay_enable_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        control_row.addWidget(self.delay_enable_button)

        # Ping-pong toggle button
        self.pingpong_button = QPushButton("Ping-Pong (Stereo)")
        self.pingpong_button.setCheckable(True)
        self.pingpong_button.setChecked(config.delay_pingpong)
        self.pingpong_button.clicked.connect(self.update_delay_pingpong)
        self.pingpong_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        control_row.addWidget(self.pingpong_button)

        # Tempo sync toggle button
        self.delay_sync_button = QPushButton("Sync to Tempo")
        self.delay_sync_button.setCheckable(True)
        self.delay_sync_button.setChecked(config.delay_sync_enabled)
        self.delay_sync_button.clicked.connect(self.update_delay_sync)
        self.delay_sync_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        control_row.addWidget(self.delay_sync_button)

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
        chorus_layout.addLayout(control_row, 0, 0, 1, 1)

        self.chorus_enable_button = QPushButton("Enable Chorus")
        self.chorus_enable_button.setCheckable(True)
        self.chorus_enable_button.setChecked(config.chorus_enabled)
        self.chorus_enable_button.clicked.connect(self.update_chorus_enabled)
        self.chorus_enable_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        control_row.addWidget(self.chorus_enable_button)

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

        # Chorus voices control
        voices_label = QLabel("Voices")
        voices_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(voices_label, 1, 3)

        self.chorus_voices_dial = QDial()
        self.chorus_voices_dial.setRange(1, 4)  # 1 to 4 voices
        self.chorus_voices_dial.setValue(config.chorus_voices)
        self.chorus_voices_dial.valueChanged.connect(self.update_chorus_voices)
        self.chorus_voices_dial.setNotchesVisible(True)
        chorus_layout.addWidget(self.chorus_voices_dial, 2, 3)

        self.chorus_voices_label = QLabel(f"{config.chorus_voices}")
        self.chorus_voices_label.setAlignment(Qt.AlignCenter)
        chorus_layout.addWidget(self.chorus_voices_label, 3, 3)

        # Visualization toggle and visualizations at the bottom
        viz_toggle_layout = QHBoxLayout()
        main_layout.addLayout(viz_toggle_layout)

        self.viz_enable_checkbox = QCheckBox("Enable Visualization (requires more CPU)")
        self.viz_enable_checkbox.setChecked(self.visualization_enabled)
        self.viz_enable_checkbox.toggled.connect(self.update_visualization_enabled)
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

        # Create the recording tab
        recording_widget = QWidget()
        recording_layout = QVBoxLayout(recording_widget)
        envelope_tabs.addTab(recording_widget, "Recording")

        # Recording controls
        recording_controls_group = QGroupBox("Recording Controls")
        recording_controls_layout = QHBoxLayout(recording_controls_group)
        recording_layout.addWidget(recording_controls_group)

        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        recording_controls_layout.addWidget(self.record_button)

        # Bit depth selector
        recording_controls_layout.addWidget(QLabel("Bit Depth:"))
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["16-bit", "24-bit"])
        self.bit_depth_combo.setCurrentIndex(1 if config.recording_bit_depth == 24 else 0)
        self.bit_depth_combo.currentTextChanged.connect(self.update_bit_depth)
        recording_controls_layout.addWidget(self.bit_depth_combo)

        # Save location button
        self.save_location_button = QPushButton("Choose Save Location...")
        self.save_location_button.clicked.connect(self.choose_recording_location)
        recording_controls_layout.addWidget(self.save_location_button)

        # Recording status
        recording_status_group = QGroupBox("Recording Status")
        recording_status_layout = QVBoxLayout(recording_status_group)
        recording_layout.addWidget(recording_status_group)

        # Status label
        self.recording_status_label = QLabel("Ready to record")
        recording_status_layout.addWidget(self.recording_status_label)

        # Current recording path
        self.recording_path_label = QLabel("Recording will be saved automatically")
        self.recording_path_label.setWordWrap(True)
        recording_status_layout.addWidget(self.recording_path_label)

        # Recording time
        time_layout = QHBoxLayout()
        recording_status_layout.addLayout(time_layout)

        time_layout.addWidget(QLabel("Recording Time:"))
        self.recording_time_label = QLabel("00:00")
        time_layout.addWidget(self.recording_time_label)
        time_layout.addStretch(1)

        # Add a timer to update recording time
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)
        self.recording_timer.setInterval(500)  # Update every 500ms

        # Recent recordings list
        recent_recordings_group = QGroupBox("Recent Recordings")
        recent_recordings_layout = QVBoxLayout(recent_recordings_group)
        recording_layout.addWidget(recent_recordings_group)

        self.recent_recordings_list = QLabel("No recordings yet")
        recent_recordings_layout.addWidget(self.recent_recordings_list)

        # Add spacer to push controls to the top
        recording_layout.addStretch(1)

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
            # Update dial without triggering signals
            self.octave_dial.blockSignals(True)
            self.octave_dial.setValue(config.octave_offset // 12)
            self.octave_dial.blockSignals(False)

        # Check if semitone has changed and update GUI if needed
        current_semitone_text = f"{config.semitone_offset:+d}"
        if self.semitone_label.text() != current_semitone_text:
            self.semitone_label.setText(current_semitone_text)
            # Update slider without triggering signals
            self.semitone_dial.blockSignals(True)
            self.semitone_dial.setValue(config.semitone_offset)
            self.semitone_dial.blockSignals(False)

        # Check if filter cutoff has changed and update GUI if needed
        if self.cutoff_dial.value() != config.filter_cutoff:
            self.cutoff_dial.setValue(int(config.filter_cutoff))
            self.cutoff_label.setText(f"{config.filter_cutoff:.0f} Hz")

        # Check if filter resonance has changed and update GUI if needed
        if self.resonance_dial.value() != int(config.filter_resonance * 100):
            self.resonance_dial.setValue(int(config.filter_resonance * 100))
            self.resonance_label.setText(f"{config.filter_resonance:.2f}")

        # Check if filter envelope amount has changed and update GUI
        if self.filter_env_amount_dial.value() != config.filter_env_amount:
            self.filter_env_amount_dial.setValue(int(config.filter_env_amount))
            self.filter_env_amount_label.setText(f"{config.filter_env_amount:.0f} Hz")

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
        if self.lfo_enable_button.isChecked() != config.lfo_enabled:
            self.lfo_enable_button.setChecked(config.lfo_enabled)

        # Check if filter enabled status has changed and update GUI if needed
        if self.filter_enable_button.isChecked() != config.filter_enabled:
            self.filter_enable_button.setChecked(config.filter_enabled)

        # Check if filter type has changed and update GUI if needed
        if self.filter_type_combo.currentText() != config.filter_type:
            self.filter_type_combo.setCurrentText(config.filter_type)

        # Check if filter topology has changed and update GUI if needed
        if self.filter_topology_combo.currentText() != config.filter_topology:
            self.filter_topology_combo.setCurrentText(config.filter_topology)
            # Enable/disable slope control based on topology
            self.filter_slope_combo.setEnabled(config.filter_topology == 'biquad')

        # Check if filter slope has changed and update GUI if needed
        expected_slope_text = "24 dB/oct" if config.filter_slope == 24 else "12 dB/oct"
        if self.filter_slope_combo.currentText() != expected_slope_text:
            self.filter_slope_combo.setCurrentText(expected_slope_text)

        # Check if drive settings have changed and update GUI if needed
        if self.drive_enable_button.isChecked() != config.drive_on:
            self.drive_enable_button.setChecked(config.drive_on)

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
        if self.delay_enable_button.isChecked() != config.delay_enabled:
            self.delay_enable_button.setChecked(config.delay_enabled)
        if self.delay_sync_button.isChecked() != config.delay_sync_enabled:
            self.delay_sync_button.setChecked(config.delay_sync_enabled)
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

        if self.attack_dial.value() != int(config.adsr['attack'] * 100):
            self.attack_dial.setValue(int(config.adsr['attack'] * 100))
            self.attack_label.setText(f"{config.adsr['attack']:.2f} s")
            adsr_changed = True

        if self.decay_dial.value() != int(config.adsr['decay'] * 100):
            self.decay_dial.setValue(int(config.adsr['decay'] * 100))
            self.decay_label.setText(f"{config.adsr['decay']:.2f} s")
            adsr_changed = True

        if self.sustain_dial.value() != int(config.adsr['sustain'] * 100):
            self.sustain_dial.setValue(int(config.adsr['sustain'] * 100))
            self.sustain_label.setText(f"{config.adsr['sustain']:.2f}")
            adsr_changed = True

        if self.release_dial.value() != int(config.adsr['release'] * 100):
            self.release_dial.setValue(int(config.adsr['release'] * 100))
            self.release_label.setText(f"{config.adsr['release']:.2f} s")
            adsr_changed = True

        # Check if filter ADSR parameters have changed
        filter_adsr_changed = False

        if self.filter_attack_dial.value() != int(config.filter_adsr['attack'] * 100):
            self.filter_attack_dial.setValue(int(config.filter_adsr['attack'] * 100))
            self.filter_attack_label.setText(f"{config.filter_adsr['attack']:.2f} s")
            filter_adsr_changed = True

        if self.filter_decay_dial.value() != int(config.filter_adsr['decay'] * 100):
            self.filter_decay_dial.setValue(int(config.filter_adsr['decay'] * 100))
            self.filter_decay_label.setText(f"{config.filter_adsr['decay']:.2f} s")
            filter_adsr_changed = True

        if self.filter_sustain_dial.value() != int(config.filter_adsr['sustain'] * 100):
            self.filter_sustain_dial.setValue(int(config.filter_adsr['sustain'] * 100))
            self.filter_sustain_label.setText(f"{config.filter_adsr['sustain']:.2f}")
            filter_adsr_changed = True

        if self.filter_release_dial.value() != int(config.filter_adsr['release'] * 100):
            self.filter_release_dial.setValue(int(config.filter_adsr['release'] * 100))
            self.filter_release_label.setText(f"{config.filter_adsr['release']:.2f} s")
            filter_adsr_changed = True

        # Check if volume has changed
        if self.volume_dial.value() != int(config.volume * 100):
            self.volume_dial.setValue(int(config.volume * 100))
            self.volume_label.setText(f"{config.volume:.2f}")

        # Check if delay ping-pong status has changed and update GUI if needed
        if self.pingpong_button.isChecked() != config.delay_pingpong:
            self.pingpong_button.setChecked(config.delay_pingpong)

        # Check if chorus parameters have changed and update GUI if needed
        if self.chorus_enable_button.isChecked() != config.chorus_enabled:
            self.chorus_enable_button.setChecked(config.chorus_enabled)

        if self.chorus_rate_dial.value() != int(config.chorus_rate * 10):
            self.chorus_rate_dial.setValue(int(config.chorus_rate * 10))
            self.chorus_rate_label.setText(f"{config.chorus_rate:.1f} Hz")

        if self.chorus_depth_dial.value() != int(config.chorus_depth * 1000):
            self.chorus_depth_dial.setValue(int(config.chorus_depth * 1000))
            self.chorus_depth_label.setText(f"{config.chorus_depth * 1000:.1f} ms")

        if self.chorus_mix_dial.value() != int(config.chorus_mix * 100):
            self.chorus_mix_dial.setValue(int(config.chorus_mix * 100))
            self.chorus_mix_label.setText(f"{config.chorus_mix:.2f}")

        if self.chorus_voices_dial.value() != config.chorus_voices:
            self.chorus_voices_dial.setValue(config.chorus_voices)
            self.chorus_voices_label.setText(f"{config.chorus_voices}")

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

        # Check if visualization enabled status has changed
        if self.viz_enable_checkbox.isChecked() != self.visualization_enabled:
            self.viz_enable_checkbox.setChecked(self.visualization_enabled)

        # Check if arpeggiator settings have changed and update GUI if needed
        if hasattr(self, 'arpeggiator') and self.arpeggiator:
            if self.arpeggiator.enabled != config.arpeggiator_enabled:
                self.arpeggiator.enabled = config.arpeggiator_enabled
                self.arpeggiator.enable_button.setChecked(config.arpeggiator_enabled)
                self.arpeggiator.enable_button.setText("Enable Arpeggiator")

            if self.arpeggiator.pattern != config.arpeggiator_pattern:
                self.arpeggiator.pattern = config.arpeggiator_pattern
                # Find the pattern name from the value
                pattern_name = next((k for k, v in self.arpeggiator.PATTERNS.items() if v == config.arpeggiator_pattern), "Up")
                self.arpeggiator.pattern_combo.setCurrentText(pattern_name)

            if self.arpeggiator.rate != config.arpeggiator_rate:
                self.arpeggiator.rate = config.arpeggiator_rate
                self.arpeggiator.rate_dial.setValue(int(config.arpeggiator_rate))
                self.arpeggiator.rate_label.setText(f"{config.arpeggiator_rate} BPM")

            if self.arpeggiator.gate != config.arpeggiator_gate:
                self.arpeggiator.gate = config.arpeggiator_gate
                self.arpeggiator.gate_dial.setValue(int(config.arpeggiator_gate * 100))
                self.arpeggiator.gate_label.setText(f"{config.arpeggiator_gate:.1f}")

            if self.arpeggiator.octave_range != config.arpeggiator_octave_range:
                self.arpeggiator.octave_range = config.arpeggiator_octave_range
                self.arpeggiator.octave_spinbox.setValue(config.arpeggiator_octave_range)

            if self.arpeggiator.sync_to_bpm != config.arpeggiator_sync_to_bpm:
                self.arpeggiator.sync_to_bpm = config.arpeggiator_sync_to_bpm
                self.arpeggiator.sync_button.setChecked(config.arpeggiator_sync_to_bpm)

            if self.arpeggiator.sustain_base != config.arpeggiator_sustain_base:
                self.arpeggiator.sustain_base = config.arpeggiator_sustain_base
                self.arpeggiator.sustain_button.setChecked(config.arpeggiator_sustain_base)

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
        config.adsr[param] = value
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
        config.filter_adsr[param] = value
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
        config.filter_cutoff = cutoff
        self.cutoff_label.setText(f"{cutoff:.0f} Hz")

    def update_filter_resonance(self, value):
        """Update the filter resonance."""
        resonance = value / 100.0
        config.filter_resonance = resonance
        self.resonance_label.setText(f"{resonance:.2f}")

    def update_filter_env_amount(self, value):
        """Update the filter envelope amount."""
        amount = float(value)
        config.filter_env_amount = amount
        self.filter_env_amount_label.setText(f"{amount:.0f} Hz")

    def update_mono_mode(self, state):
        """Update the mono mode setting."""
        config.mono_mode = state
        self.mono_button.setChecked(state)
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
        config.lfo_enabled = state
        self.lfo_enable_button.setChecked(state)

    def update_filter_enabled(self, state):
        """Update the filter enabled setting."""
        config.filter_enabled = state
        self.filter_enable_button.setChecked(state)

    def update_filter_type(self, filter_type):
        """Update the filter type setting."""
        config.filter_type = filter_type
        # Reset filter state when changing types to avoid artifacts
        filter.reset_filter_state()

    def update_filter_topology(self, topology):
        """Update the filter topology setting."""
        config.filter_topology = topology
        # Reset filter state when changing topology to avoid artifacts
        filter.reset_filter_state()
        # Enable/disable slope control based on topology
        self.filter_slope_combo.setEnabled(topology == 'biquad')

    def update_filter_slope(self, slope_text):
        """Update the filter slope setting."""
        config.filter_slope = 24 if "24" in slope_text else 12
        # Reset filter state when changing slope to avoid artifacts
        filter.reset_filter_state()

    def update_drive_enabled(self, state):
        """Update the drive enabled setting."""
        config.drive_on = state
        self.drive_enable_button.setChecked(state)

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
        config.delay_enabled = state
        self.delay_enable_button.setChecked(state)
        if not config.delay_enabled:
            synth.delay.clear_cache()

    def update_delay_sync(self, state):
        """Update whether delay time is synced to BPM."""
        config.delay_sync_enabled = state
        self.delay_sync_button.setChecked(state)
        self.update_delay_sync_controls()

        # If enabling sync, update delay time from BPM
        if config.delay_sync_enabled:
            self.update_delay_from_bpm()
            self.delay_time_dial.setValue(int(config.delay_time_ms))
            self.delay_time_label.setText(f"{config.delay_time_ms:.0f} ms")
            self.delay_ms_label.setText(f"{config.delay_time_ms:.1f} ms")

    def update_delay_sync_controls(self):
        """Show/hide appropriate controls based on sync setting."""
        # Show/hide sync controls based on sync button state
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

        # Also sync arpeggiator BPM if it exists and is synced
        if hasattr(self, 'arpeggiator') and self.arpeggiator:
            self.arpeggiator.sync_bpm_changed(bpm)

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

        # Also sync arpeggiator BPM if it exists and is synced
        if hasattr(self, 'arpeggiator') and self.arpeggiator:
            self.arpeggiator.sync_bpm_changed(bpm)

    def decrease_octave(self):
        """Decrease the octave by one."""
        if config.octave_offset > 12 * config.octave_min:
            config.octave_offset -= 12
            self.octave_label.setText(f"{config.octave_offset // 12:+d}")
            # Update slider without triggering the signal
            self.octave_dial.blockSignals(True)
            self.octave_dial.setValue(config.octave_offset // 12)
            self.octave_dial.blockSignals(False)

            # Clear arpeggiator when transpose changes
            if hasattr(self, 'arpeggiator') and self.arpeggiator:
                self.arpeggiator.clear_notes()

    def increase_octave(self):
        """Increase the octave by one."""
        if config.octave_offset < 12 * config.octave_max:
            config.octave_offset += 12
            self.octave_label.setText(f"{config.octave_offset // 12:+d}")
            # Update slider without triggering the signal
            self.octave_dial.blockSignals(True)
            self.octave_dial.setValue(config.octave_offset // 12)
            self.octave_dial.blockSignals(False)

            # Clear arpeggiator when transpose changes
            if hasattr(self, 'arpeggiator') and self.arpeggiator:
                self.arpeggiator.clear_notes()

    def update_octave(self, value):
        """Update the octave from dial value."""
        # Convert dial value to offset (x12 semitones per octave)
        config.octave_offset = value * 12
        self.octave_label.setText(f"{value:+d}")

        # Clear arpeggiator when transpose changes
        if hasattr(self, 'arpeggiator') and self.arpeggiator:
            self.arpeggiator.clear_notes()

    def update_semitone(self, value):
        """Update the semitone transpose from dial value."""
        config.semitone_offset = value
        self.semitone_label.setText(f"{value:+d}")

        # Clear arpeggiator when transpose changes
        if hasattr(self, 'arpeggiator') and self.arpeggiator:
            self.arpeggiator.clear_notes()

    def update_delay_pingpong(self, state):
        """Update the delay ping-pong setting."""
        config.delay_pingpong = state
        self.pingpong_button.setChecked(state)
        if config.delay_enabled:
            synth.delay.clear_cache()  # Clear buffers to avoid artifacts when switching modes

    def update_chorus_enabled(self, state):
        """Update the chorus enabled setting."""
        config.chorus_enabled = state
        self.chorus_enable_button.setChecked(state)
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

    def update_chorus_voices(self, value):
        """Update the chorus voices setting."""
        config.chorus_voices = value
        synth.chorus.set_voices(value)
        self.chorus_voices_label.setText(f"{value}")

    def update_visualization_enabled(self, state):
        """Update visualization enabled status and visibility."""
        self.visualization_enabled = state
        self.viz_enable_checkbox.setChecked(state)
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

    def toggle_recording(self):
        """Start or stop recording based on button state."""
        if self.record_button.isChecked():
            # Start recording
            bit_depth = 24 if "24" in self.bit_depth_combo.currentText() else 16

            if hasattr(self, 'selected_recording_path') and self.selected_recording_path:
                # Use user-selected path if available
                output_path = self.selected_recording_path
            else:
                # Let the record module generate a path
                output_path = None

            path = record.start_recording(output_path)

            # Update UI
            self.record_button.setText("Stop Recording")
            self.recording_status_label.setText("Recording in progress")
            self.recording_path_label.setText(f"Will be saved to: {path}")

            # Start timer to update recording time
            self.recording_timer.start()

            # Disable save location button while recording
            self.save_location_button.setEnabled(False)

            # Clear the selected path
            self.selected_recording_path = None
        else:
            # Stop recording
            bit_depth = 24 if "24" in self.bit_depth_combo.currentText() else 16
            saved_path = record.stop_recording(config.sample_rate, bit_depth)

            # Update UI
            self.record_button.setText("Start Recording")
            self.recording_status_label.setText("Recording saved")
            if saved_path:
                self.recording_path_label.setText(f"Saved to: {saved_path}")

                # Update recent recordings list
                current_text = self.recent_recordings_list.text()
                if current_text == "No recordings yet":
                    self.recent_recordings_list.setText(os.path.basename(saved_path))
                else:
                    lines = current_text.split('\n')
                    if len(lines) >= 5:  # Keep only the 5 most recent
                        lines.pop()
                    lines.insert(0, os.path.basename(saved_path))
                    self.recent_recordings_list.setText('\n'.join(lines))

            # Stop timer
            self.recording_timer.stop()
            self.recording_time_label.setText("00:00")

            # Re-enable save location button
            self.save_location_button.setEnabled(True)

    def update_bit_depth(self, text):
        """Update the recording bit depth setting."""
        config.recording_bit_depth = 24 if "24" in text else 16

    def choose_recording_location(self):
        """Open file dialog to choose where to save the next recording."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Recording As", "", "WAV Files (*.wav);;All Files (*)"
        )

        if file_path:
            # Add .wav extension if not present
            if not file_path.lower().endswith('.wav'):
                file_path += '.wav'

            self.selected_recording_path = file_path
            self.recording_path_label.setText(f"Will be saved to: {file_path}")

    def update_recording_time(self):
        """Update the recording time display."""
        if record.is_recording():
            # Get recording time in seconds
            seconds = int(record.get_recording_time())
            minutes = seconds // 60
            seconds = seconds % 60

            # Update label
            self.recording_time_label.setText(f"{minutes:02d}:{seconds:02d}")

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

        # Stop arpeggiator if running
        if hasattr(self, 'arpeggiator') and self.arpeggiator:
            self.arpeggiator.stop()

        # Stop recording if active
        if record.is_recording():
            record.stop_recording(config.sample_rate, config.recording_bit_depth)

        event.accept()

    def quick_save_patch(self):
        """Quick save with the current patch name (or prompt if untitled)."""
        if self.current_patch_name == "Untitled":
            self.save_patch()  # This will prompt for a name
        else:
            try:
                path = patch.save_patch(self.current_patch_name)
                self.current_patch_path = path
                self.refresh_patch_list()
                QMessageBox.information(self, "Patch Saved", f"Patch '{self.current_patch_name}' saved successfully.")
            except patch.PatchError as e:
                QMessageBox.warning(self, "Save Error", f"Error saving patch: {str(e)}")

    def save_patch(self):
        """Save current settings as a patch."""
        name, ok = QInputDialog.getText(
            self, "Save Patch", "Enter a name for this patch:",
            QLineEdit.Normal, self.current_patch_name
        )

        if ok and name:
            # Check if patch with this name already exists
            patches = patch.list_patches()
            for p in patches:
                if p["name"] == name and (not self.current_patch_path or p["path"] != self.current_patch_path):
                    # Ask for confirmation to overwrite
                    confirm = QMessageBox.question(
                        self, "Confirm Overwrite",
                        f"A patch named '{name}' already exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                    )
                    if confirm == QMessageBox.No:
                        return

            try:
                path = patch.save_patch(name)
                self.current_patch_name = name
                self.current_patch_path = path
                self.current_patch_label.setText(name)
                self.refresh_patch_list()
                QMessageBox.information(self, "Patch Saved", f"Patch '{name}' saved successfully.")
            except patch.PatchError as e:
                QMessageBox.warning(self, "Save Error", f"Error saving patch: {str(e)}")

    def load_selected_patch(self):
        """Load the currently selected patch."""
        selected_items = self.patch_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select a patch to load.")
            return

        selected_item = selected_items[0]
        patch_path = selected_item.data(Qt.UserRole)

        self.load_patch_from_path(patch_path)

    def load_patch_from_path(self, path):
        """Load a patch from the specified path."""
        try:
            patch_data = patch.load_patch(path)
            self.current_patch_name = patch_data.get("name", "Unnamed")
            self.current_patch_path = path
            self.current_patch_label.setText(self.current_patch_name)

            # Refresh all controls to reflect loaded settings
            # The update_plots method will automatically update the GUI controls
            # based on the new config values on the next animation frame
            # QMessageBox.information(self, "Patch Loaded", f"Patch '{self.current_patch_name}' loaded successfully.")
        except patch.PatchError as e:
            QMessageBox.warning(self, "Load Error", f"Error loading patch: {str(e)}")

    def show_load_patch_dialog(self):
        """Show a dialog to load a patch."""
        # Simply show the patches tab
        for i in range(self.findChild(QTabWidget).count()):
            if self.findChild(QTabWidget).tabText(i) == "Patches":
                self.findChild(QTabWidget).setCurrentIndex(i)
                break

    def rename_patch(self):
        """Rename the selected patch."""
        selected_items = self.patch_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select a patch to rename.")
            return

        selected_item = selected_items[0]
        patch_path = selected_item.data(Qt.UserRole)
        current_name = selected_item.text()

        new_name, ok = QInputDialog.getText(
            self, "Rename Patch", "Enter a new name for this patch:",
            QLineEdit.Normal, current_name
        )

        if ok and new_name:
            try:
                new_path = patch.rename_patch(patch_path, new_name)

                # Update current patch name if this is the loaded patch
                if self.current_patch_path == patch_path:
                    self.current_patch_name = new_name
                    self.current_patch_path = new_path
                    self.current_patch_label.setText(new_name)

                self.refresh_patch_list()
                QMessageBox.information(self, "Patch Renamed", f"Patch renamed to '{new_name}' successfully.")
            except patch.PatchError as e:
                QMessageBox.warning(self, "Rename Error", f"Error renaming patch: {str(e)}")

    def delete_patch(self):
        """Delete the selected patch."""
        selected_items = self.patch_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select a patch to delete.")
            return

        selected_item = selected_items[0]
        patch_path = selected_item.data(Qt.UserRole)
        patch_name = selected_item.text()

        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the patch '{patch_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            try:
                patch.delete_patch(patch_path)

                # Reset current patch name if this was the loaded patch
                if self.current_patch_path == patch_path:
                    self.current_patch_name = "Untitled"
                    self.current_patch_path = None
                    self.current_patch_label.setText("Untitled")

                self.refresh_patch_list()
                QMessageBox.information(self, "Patch Deleted", f"Patch '{patch_name}' deleted successfully.")
            except patch.PatchError as e:
                QMessageBox.warning(self, "Delete Error", f"Error deleting patch: {str(e)}")

    def refresh_patch_list(self):
        """Refresh the list of available patches."""
        self.patch_list.clear()

        try:
            patches = patch.list_patches()

            for p in patches:
                item = QListWidgetItem(p["name"])
                item.setData(Qt.UserRole, p["path"])
                # Use created date as tooltip
                if "created" in p:
                    try:
                        # Try to format the ISO timestamp nicely
                        created = p["created"].replace("T", " ").replace("Z", " UTC")
                        item.setToolTip(f"Created: {created}")
                    except:
                        item.setToolTip(f"Created: {p['created']}")
                self.patch_list.addItem(item)

                # Select current patch if it exists
                if self.current_patch_path and p["path"] == self.current_patch_path:
                    self.patch_list.setCurrentItem(item)
        except Exception as e:
            print(f"Error refreshing patch list: {e}")

    def filter_patches(self, search_text):
        """Filter the patches list based on search text."""
        search_text = search_text.lower()

        for i in range(self.patch_list.count()):
            item = self.patch_list.item(i)
            if not search_text or search_text in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)


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

    # Start audio using synth entry points
    if not synth.start_audio():
        QMessageBox.critical(gui, "Audio Error", "Failed to start audio. Please check your audio system.")
        sys.exit(1)

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
        # Clean up using synth entry points
        synth.stop_audio()


if __name__ == "__main__":
    start_gui()
