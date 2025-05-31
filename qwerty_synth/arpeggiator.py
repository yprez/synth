"""Arpeggiator module for QWERTY Synth."""

import time
import threading
from typing import List, Dict, Set
from PyQt5.QtCore import Qt, QTimer, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QComboBox, QDial, QSlider
)

from qwerty_synth import config
from qwerty_synth.controller import play_midi_note_direct


class Arpeggiator(QWidget):
    """Arpeggiator with configurable patterns and timing for QWERTY Synth."""

    # Arpeggio patterns
    PATTERNS = {
        'Up': 'up',
        'Down': 'down',
        'Up/Down': 'up_down',
        'Down/Up': 'down_up',
        'Random': 'random',
        'Chord': 'chord',
        'Octaves': 'octaves',
        'Order Played': 'order'
    }

    # Color constants for the dark theme
    COLORS = {
        'inactive': '#353535',      # Dark gray for inactive
        'active': '#2980b9',        # Blue for active
        'current': '#4fc3f7',       # Bright cyan-blue for current playing note
    }

    # Toggle button style
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

    def __init__(self, parent=None):
        """Initialize the arpeggiator."""
        super().__init__(parent)

        # Arpeggiator state - initialize from config
        self.enabled = config.arpeggiator_enabled
        self.pattern = config.arpeggiator_pattern
        self.rate = config.arpeggiator_rate
        self.gate = config.arpeggiator_gate
        self.octave_range = config.arpeggiator_octave_range
        self.sync_to_bpm = config.arpeggiator_sync_to_bpm
        self.sustain_base = config.arpeggiator_sustain_base

        # Internal state
        self.held_notes: Set[int] = set()  # MIDI notes currently held
        self.note_order: List[int] = []  # Order notes were pressed (for 'order' pattern)
        self.current_sequence: List[int] = []  # Current arpeggio sequence
        self.sequence_position = 0
        self.is_running = False

        # Test mode flag - when True, bypasses Qt threading for unit tests
        self._test_mode = False

        # Timing
        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self.advance_arpeggio)
        self.step_timer.setSingleShot(False)

        # Thread lock for note management
        self.notes_lock = threading.Lock()

        # Currently playing note (for visualization)
        self.current_note = None
        self.last_played_time = 0

        self.setup_ui()
        self.update_timing()

    def setup_ui(self):
        """Set up the arpeggiator UI components."""
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # Left side controls
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        main_layout.addWidget(controls_container)

        # First row: Enable, Pattern, Sync
        controls_row1 = QHBoxLayout()
        controls_layout.addLayout(controls_row1)

        # Enable button
        self.enable_button = QPushButton("Enable Arpeggiator")
        self.enable_button.setCheckable(True)
        self.enable_button.setChecked(self.enabled)
        self.enable_button.clicked.connect(self.toggle_enabled)
        self.enable_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        controls_row1.addWidget(self.enable_button)

        # Sync to BPM button
        self.sync_button = QPushButton("Sync to BPM")
        self.sync_button.setCheckable(True)
        self.sync_button.setChecked(self.sync_to_bpm)
        self.sync_button.clicked.connect(self.toggle_sync)
        self.sync_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        controls_row1.addWidget(self.sync_button)

        # Sustain base button
        self.sustain_button = QPushButton("Sustain Base")
        self.sustain_button.setCheckable(True)
        self.sustain_button.setChecked(self.sustain_base)
        self.sustain_button.clicked.connect(self.toggle_sustain_base)
        self.sustain_button.setStyleSheet(self.TOGGLE_BUTTON_STYLE)
        controls_row1.addWidget(self.sustain_button)

        controls_row1.addStretch(1)

        # Second row: Pattern selector and Octave range
        controls_row2 = QHBoxLayout()
        controls_layout.addLayout(controls_row2)

        # Pattern selector
        controls_row2.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(self.PATTERNS.keys()))
        # Set initial pattern from config
        pattern_name = next((k for k, v in self.PATTERNS.items() if v == self.pattern), "Up")
        self.pattern_combo.setCurrentText(pattern_name)
        self.pattern_combo.currentTextChanged.connect(self.update_pattern)
        controls_row2.addWidget(self.pattern_combo)

        # Octave range
        controls_row2.addWidget(QLabel("Octave Range:"))
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(1, 4)
        self.octave_spinbox.setValue(self.octave_range)
        self.octave_spinbox.valueChanged.connect(self.update_octave_range)
        controls_row2.addWidget(self.octave_spinbox)

        controls_row2.addStretch(1)

        # Third row: Rate and Gate controls
        controls_row3 = QGridLayout()
        controls_layout.addLayout(controls_row3)

        # Rate control
        rate_label = QLabel("Rate")
        rate_label.setAlignment(Qt.AlignCenter)
        controls_row3.addWidget(rate_label, 0, 0)

        self.rate_dial = QDial()
        self.rate_dial.setRange(40, 200)  # 40-200 BPM
        self.rate_dial.setValue(self.rate)
        self.rate_dial.valueChanged.connect(self.update_rate)
        self.rate_dial.setNotchesVisible(True)
        controls_row3.addWidget(self.rate_dial, 1, 0)

        self.rate_label = QLabel(f"{self.rate} BPM")
        self.rate_label.setAlignment(Qt.AlignCenter)
        controls_row3.addWidget(self.rate_label, 2, 0)

        # Gate control
        gate_label = QLabel("Gate")
        gate_label.setAlignment(Qt.AlignCenter)
        controls_row3.addWidget(gate_label, 0, 1)

        self.gate_dial = QDial()
        self.gate_dial.setRange(10, 100)  # 0.1 to 1.0 (x100)
        self.gate_dial.setValue(int(self.gate * 100))
        self.gate_dial.valueChanged.connect(self.update_gate)
        self.gate_dial.setNotchesVisible(True)
        controls_row3.addWidget(self.gate_dial, 1, 1)

        self.gate_label = QLabel(f"{self.gate:.1f}")
        self.gate_label.setAlignment(Qt.AlignCenter)
        controls_row3.addWidget(self.gate_label, 2, 1)

        # Add spacer to push controls to the top
        controls_layout.addStretch(1)

        # Right side - Note display (simple text for now)
        display_container = QWidget()
        display_layout = QVBoxLayout(display_container)
        main_layout.addWidget(display_container, stretch=1)

        # Current notes display
        display_layout.addWidget(QLabel("Held Notes:"))
        self.held_notes_label = QLabel("None")
        self.held_notes_label.setWordWrap(True)
        display_layout.addWidget(self.held_notes_label)

        # Current sequence display
        display_layout.addWidget(QLabel("Arpeggio Sequence:"))
        self.sequence_label = QLabel("None")
        self.sequence_label.setWordWrap(True)
        display_layout.addWidget(self.sequence_label)

        # Current playing note
        display_layout.addWidget(QLabel("Playing:"))
        self.current_note_label = QLabel("None")
        display_layout.addWidget(self.current_note_label)

        display_layout.addStretch(1)

    def toggle_enabled(self, checked):
        """Enable or disable the arpeggiator."""
        self.enabled = checked
        config.arpeggiator_enabled = checked
        if self.enabled:
            self.enable_button.setText("Disable Arpeggiator")
            # Start arpeggiator if we have held notes
            if self.held_notes:
                self.start_arpeggio()
        else:
            self.enable_button.setText("Enable Arpeggiator")
            self.stop_arpeggio()

    def toggle_sync(self, checked):
        """Toggle BPM sync."""
        self.sync_to_bpm = checked
        config.arpeggiator_sync_to_bpm = checked
        self.update_timing()

    def toggle_sustain_base(self, checked):
        """Toggle sustaining base notes while arpeggiating."""
        self.sustain_base = checked
        config.arpeggiator_sustain_base = checked

    def update_pattern(self, pattern_name):
        """Update the arpeggio pattern."""
        self.pattern = self.PATTERNS[pattern_name]
        config.arpeggiator_pattern = self.pattern
        self.generate_sequence()

    def update_rate(self, value):
        """Update the arpeggio rate."""
        self.rate = value
        config.arpeggiator_rate = value
        self.rate_label.setText(f"{value} BPM")
        self.update_timing()

    def update_gate(self, value):
        """Update the gate time."""
        self.gate = value / 100.0
        config.arpeggiator_gate = self.gate
        self.gate_label.setText(f"{self.gate:.1f}")

    def update_octave_range(self, value):
        """Update the octave range."""
        self.octave_range = value
        config.arpeggiator_octave_range = value
        self.generate_sequence()

    def update_timing(self):
        """Update timer interval based on rate and sync settings."""
        if self.sync_to_bpm:
            # Use global BPM from config
            bpm = config.bpm
        else:
            # Use arpeggiator's own rate
            bpm = self.rate

        # Calculate interval in milliseconds (16th notes)
        interval_ms = int(60000 / bpm / 4)

        # Always set the timer interval, whether running or not
        self.step_timer.setInterval(interval_ms)

    def add_note(self, midi_note):
        """Add a note to the held notes."""
        with self.notes_lock:
            if midi_note not in self.held_notes:
                self.held_notes.add(midi_note)
                self.note_order.append(midi_note)
                self.generate_sequence()
                self.update_display()

                # Start arpeggio if enabled and not already running
                if self.enabled and not self.is_running:
                    self.start_arpeggio()

    def remove_note(self, midi_note):
        """Remove a note from the held notes."""
        with self.notes_lock:
            if midi_note in self.held_notes:
                self.held_notes.remove(midi_note)
                if midi_note in self.note_order:
                    self.note_order.remove(midi_note)

                self.generate_sequence()
                self.update_display()

                # Stop arpeggio if no notes left
                if not self.held_notes:
                    self.stop_arpeggio()

    def generate_sequence(self):
        """Generate the arpeggio sequence based on current pattern and held notes."""
        if not self.held_notes:
            self.current_sequence = []
            return

        # Get base notes sorted
        base_notes = sorted(list(self.held_notes))

        # Generate sequence based on pattern
        if self.pattern == 'up':
            sequence = []
            for octave in range(self.octave_range):
                sequence.extend([note + (octave * 12) for note in base_notes])
            self.current_sequence = sequence

        elif self.pattern == 'down':
            sequence = []
            for octave in reversed(range(self.octave_range)):
                sequence.extend([note + (octave * 12) for note in reversed(base_notes)])
            self.current_sequence = sequence

        elif self.pattern == 'up_down':
            sequence = []
            for octave in range(self.octave_range):
                sequence.extend([note + (octave * 12) for note in base_notes])
            # Add descending part (excluding the highest note to avoid repetition)
            if len(sequence) > 1:
                sequence.extend(reversed(sequence[:-1]))
            self.current_sequence = sequence

        elif self.pattern == 'down_up':
            # Build proper descending sequence (highest to lowest across octaves)
            down_sequence = []
            for octave in reversed(range(self.octave_range)):
                down_sequence.extend([note + (octave * 12) for note in reversed(base_notes)])

            # Build ascending sequence (lowest to highest across octaves)
            up_sequence = []
            for octave in range(self.octave_range):
                up_sequence.extend([note + (octave * 12) for note in base_notes])

            # Combine: down sequence + up sequence (excluding first note to avoid repetition)
            sequence = down_sequence[:]
            if len(up_sequence) > 1:
                sequence.extend(up_sequence[1:])

            self.current_sequence = sequence

        elif self.pattern == 'random':
            sequence = []
            for octave in range(self.octave_range):
                sequence.extend([note + (octave * 12) for note in base_notes])
            # For random, we'll randomize in the advance_arpeggio method
            self.current_sequence = sequence

        elif self.pattern == 'chord':
            # Play all notes at once
            sequence = []
            for octave in range(self.octave_range):
                sequence.extend([note + (octave * 12) for note in base_notes])
            self.current_sequence = [sequence]  # Single "chord" element

        elif self.pattern == 'octaves':
            # Play each note across all octaves before moving to next note
            sequence = []
            for note in base_notes:
                for octave in range(self.octave_range):
                    sequence.append(note + (octave * 12))
            self.current_sequence = sequence

        elif self.pattern == 'order':
            # Play notes in the order they were pressed
            sequence = []
            for octave in range(self.octave_range):
                sequence.extend([note + (octave * 12) for note in self.note_order])
            self.current_sequence = sequence

        # Reset sequence position
        self.sequence_position = 0

    def start_arpeggio(self):
        """Start the arpeggio playback (thread-safe)."""
        if not self.current_sequence or self.is_running:
            return

        # In test mode, call directly; otherwise use Qt threading
        if self._test_mode:
            self._start_arpeggio_impl()
        else:
            # Use QMetaObject.invokeMethod to ensure timer operations happen in the correct thread
            QMetaObject.invokeMethod(self, "_start_arpeggio_impl", Qt.QueuedConnection)

    @pyqtSlot()
    def _start_arpeggio_impl(self):
        """Internal implementation of start_arpeggio that runs in the main thread."""
        if not self.current_sequence or self.is_running:
            return

        self.is_running = True

        # Skip timer operations in test mode
        if self._test_mode:
            return

        # Calculate timer interval
        if self.sync_to_bpm:
            bpm = config.bpm
        else:
            bpm = self.rate

        interval_ms = int(60000 / bpm / 4)  # 16th notes
        self.step_timer.setInterval(interval_ms)
        self.step_timer.start()

    def stop_arpeggio(self):
        """Stop the arpeggio playback (thread-safe)."""
        # In test mode, call directly; otherwise use Qt threading
        if self._test_mode:
            self._stop_arpeggio_impl()
        else:
            # Use QMetaObject.invokeMethod to ensure timer operations happen in the correct thread
            QMetaObject.invokeMethod(self, "_stop_arpeggio_impl", Qt.QueuedConnection)

    @pyqtSlot()
    def _stop_arpeggio_impl(self):
        """Internal implementation of stop_arpeggio that runs in the main thread."""
        self.is_running = False
        self.current_note = None

        # Skip timer operations in test mode
        if not self._test_mode:
            self.step_timer.stop()

        # In test mode, skip UI updates; otherwise use Qt threading
        if self._test_mode:
            # In test mode, just clear the current note without updating UI
            pass
        else:
            # Use QMetaObject.invokeMethod for UI updates too
            QMetaObject.invokeMethod(self.current_note_label, "setText", Qt.QueuedConnection, Q_ARG(str, "None"))

    def advance_arpeggio(self):
        """Advance to the next note in the arpeggio."""
        if not self.current_sequence or not self.enabled:
            self.stop_arpeggio()
            return

        # Calculate note duration based on gate
        if self.sync_to_bpm:
            bpm = config.bpm
        else:
            bpm = self.rate

        step_duration = 60.0 / bpm / 4  # Duration of one 16th note
        note_duration = step_duration * self.gate

        # Get current note(s)
        if self.pattern == 'chord':
            # Play all notes in the chord
            notes_to_play = self.current_sequence[0]  # Chord is stored as a list
            for note in notes_to_play:
                play_midi_note_direct(note, note_duration, 0.8)
            self.current_note = f"Chord: {', '.join([self.midi_to_note_name(n) for n in notes_to_play])}"
        elif self.pattern == 'random':
            # Pick a random note from the sequence
            import random
            note = random.choice(self.current_sequence)
            play_midi_note_direct(note, note_duration, 0.8)
            self.current_note = self.midi_to_note_name(note)
        else:
            # Regular sequential patterns
            note = self.current_sequence[self.sequence_position]
            play_midi_note_direct(note, note_duration, 0.8)
            self.current_note = self.midi_to_note_name(note)

            # Advance position
            self.sequence_position = (self.sequence_position + 1) % len(self.current_sequence)

        # Update display using appropriate method for test mode
        if self._test_mode:
            # In test mode, just store the current note without updating UI
            pass
        else:
            QMetaObject.invokeMethod(self.current_note_label, "setText", Qt.QueuedConnection, Q_ARG(str, self.current_note))
        self.last_played_time = time.time()

    def midi_to_note_name(self, midi_note):
        """Convert MIDI note number to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note = note_names[midi_note % 12]
        return f"{note}{octave}"

    def update_display(self):
        """Update the display labels."""
        # Skip UI updates in test mode to avoid Qt threading issues
        if self._test_mode:
            return

        if self.held_notes:
            note_names = [self.midi_to_note_name(note) for note in sorted(self.held_notes)]
            self.held_notes_label.setText(", ".join(note_names))
        else:
            self.held_notes_label.setText("None")

        if self.current_sequence:
            if self.pattern == 'chord':
                seq_names = [self.midi_to_note_name(note) for note in self.current_sequence[0]]
                self.sequence_label.setText(f"Chord: {', '.join(seq_names)}")
            else:
                seq_names = [self.midi_to_note_name(note) for note in self.current_sequence]
                # Limit display to first 8 notes for readability
                if len(seq_names) > 8:
                    display_names = seq_names[:8] + ["..."]
                else:
                    display_names = seq_names
                self.sequence_label.setText(", ".join(display_names))
        else:
            self.sequence_label.setText("None")

    def clear_notes(self):
        """Clear all held notes."""
        with self.notes_lock:
            self.held_notes.clear()
            self.note_order.clear()
            self.current_sequence.clear()
            self.stop_arpeggio()
            self.update_display()

    def sync_bpm_changed(self, bpm):
        """Called when global BPM changes (for sync mode)."""
        if self.sync_to_bpm:
            self.update_timing()

    def stop(self):
        """Stop the arpeggiator and clean up."""
        self.stop_arpeggio()
        self.clear_notes()


# Global arpeggiator instance that will be created by the GUI
arpeggiator_instance = None
