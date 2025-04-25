"""Step sequencer module for QWERTY Synth."""

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QComboBox, QFrame
)

from qwerty_synth.controller import play_midi_note


class StepSequencer(QWidget):
    """16-step sequencer with configurable note rows for QWERTY Synth."""

    # Interval patterns for different scales (semitones)
    SCALES = {
        'Major': [0, 2, 4, 5, 7, 9, 11, 12],  # C, D, E, F, G, A, B, C
        'Minor': [0, 2, 3, 5, 7, 8, 10, 12],  # C, D, Eb, F, G, Ab, Bb, C
        'Dorian': [0, 2, 3, 5, 7, 9, 10, 12],  # C, D, Eb, F, G, A, Bb, C
        'Phrygian': [0, 1, 3, 5, 7, 8, 10, 12],  # C, Db, Eb, F, G, Ab, Bb, C
        'Lydian': [0, 2, 4, 6, 7, 9, 11, 12],  # C, D, E, F#, G, A, B, C
        'Mixolydian': [0, 2, 4, 5, 7, 9, 10, 12],  # C, D, E, F, G, A, Bb, C
        'Locrian': [0, 1, 3, 5, 6, 8, 10, 12],  # C, Db, Eb, F, Gb, Ab, Bb, C
        'Pentatonic Major': [0, 2, 4, 7, 9, 12, 14, 16],  # C, D, E, G, A, C, D, E
        'Pentatonic Minor': [0, 3, 5, 7, 10, 12, 15, 17],  # C, Eb, F, G, Bb, C, Eb, F
        'Blues': [0, 3, 5, 6, 7, 10, 12, 15],  # C, Eb, F, F#, G, Bb, C, Eb
        'Chromatic': [0, 1, 2, 3, 4, 5, 6, 7],  # C, C#, D, D#, E, F, F#, G
    }

    # Extended intervals for scales with more rows
    EXTENDED_SCALES = {
        'Major': [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24],  # Up to 2 octaves
        'Minor': [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24],
        'Dorian': [0, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19, 21, 22, 24],
        'Phrygian': [0, 1, 3, 5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24],
        'Lydian': [0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 18, 19, 21, 23, 24],
        'Mixolydian': [0, 2, 4, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 22, 24],
        'Locrian': [0, 1, 3, 5, 6, 8, 10, 12, 13, 15, 17, 18, 20, 22, 24],
        'Pentatonic Major': [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33],
        'Pentatonic Minor': [0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24, 27, 29, 31, 34],
        'Blues': [0, 3, 5, 6, 7, 10, 12, 15, 17, 18, 19, 22, 24, 27, 29],
        'Chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    }

    # Root note options with their MIDI base values (for C4 = middle C)
    ROOT_NOTES = {
        "C": 60,
        "C#/Db": 61,
        "D": 62,
        "D#/Eb": 63,
        "E": 64,
        "F": 65,
        "F#/Gb": 66,
        "G": 67,
        "G#/Ab": 68,
        "A": 69,
        "A#/Bb": 70,
        "B": 71
    }

    # Note names in chromatic scale (both enharmonic versions where relevant)
    CHROMATIC_NOTES = [
        ["C"],
        ["C#", "Db"],
        ["D"],
        ["D#", "Eb"],
        ["E"],
        ["F"],
        ["F#", "Gb"],
        ["G"],
        ["G#", "Ab"],
        ["A"],
        ["A#", "Bb"],
        ["B"]
    ]

    # Preferred accidentals for each scale (sharps vs flats)
    # True for sharps, False for flats
    SCALE_SHARPS_PREFERENCE = {
        'Major': True,     # Use sharps (F#)
        'Minor': False,    # Use flats (Bb)
        'Dorian': False,   # Use flats (Bb)
        'Phrygian': False, # Use flats (Db)
        'Lydian': True,    # Use sharps (F#)
        'Mixolydian': False, # Use flats (Bb)
        'Locrian': False,  # Use flats (Bb)
        'Pentatonic Major': True,  # Use sharps
        'Pentatonic Minor': False, # Use flats
        'Blues': False,    # Use flats (Bb)
        'Chromatic': True  # Use sharps by default
    }

    # Sharp keys (use sharps for accidentals)
    SHARP_KEYS = ["C", "G", "D", "A", "E", "B", "F#/Gb", "C#/Db"]

    # Flat keys (use flats for accidentals)
    FLAT_KEYS = ["F", "Bb/A#", "Eb/D#", "Ab/G#", "Db/C#", "Gb/F#", "Cb"]

    def __init__(self, parent=None):
        """Initialize the step sequencer."""
        super().__init__(parent)

        # Initialize sequencer state
        self.sequencer_steps = [[False for _ in range(16)] for _ in range(8)]
        self.current_step = -1  # No step active until started
        self.sequencer_running = False
        self.bpm = 60  # Don't change this! I like it this way.
        self.step_buttons = []  # Will hold references to all step buttons
        self.note_labels = []   # Will hold references to note labels

        # Scale and note configuration
        self.current_scale = 'Major'  # Default scale
        self.root_note_name = 'C'     # Default root note name
        self.root_note = self.ROOT_NOTES[self.root_note_name]  # MIDI note for root
        self.octave_offset = 0  # Default octave (C4 to C5 with offset 0)
        self.num_rows = 8      # Default number of rows

        # Grid layout container that will be recreated when rows change
        self.grid_container = None
        self.grid_widget = None
        self.grid_layout = None

        # Generate base notes based on the chosen scale
        self.base_notes = self._generate_scale_notes()

        # Current sequencer notes (will be updated when octave/scale changes)
        self.sequencer_notes = self.base_notes.copy()

        # Duration for each step in seconds
        self.step_duration = 0.1

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the sequencer UI components."""
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # Top controls in two rows for better organization
        top_controls = QVBoxLayout()
        main_layout.addLayout(top_controls)

        # First row of controls: BPM, Scale, Root Note
        controls_row1 = QHBoxLayout()
        top_controls.addLayout(controls_row1)

        # BPM control
        controls_row1.addWidget(QLabel("BPM:"))
        self.bpm_spinbox = QSpinBox()
        self.bpm_spinbox.setRange(40, 300)
        self.bpm_spinbox.setValue(self.bpm)
        self.bpm_spinbox.valueChanged.connect(self.update_sequencer_bpm)
        controls_row1.addWidget(self.bpm_spinbox)

        # Root note selector
        controls_row1.addWidget(QLabel("Root:"))
        self.root_note_combo = QComboBox()
        self.root_note_combo.addItems(list(self.ROOT_NOTES.keys()))
        self.root_note_combo.setCurrentText(self.root_note_name)
        self.root_note_combo.currentTextChanged.connect(self.update_root_note)
        controls_row1.addWidget(self.root_note_combo)

        # Scale selector
        controls_row1.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(list(self.SCALES.keys()))
        self.scale_combo.setCurrentText(self.current_scale)
        self.scale_combo.currentTextChanged.connect(self.update_scale)
        controls_row1.addWidget(self.scale_combo)

        # Octave selector
        controls_row1.addWidget(QLabel("Octave:"))
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(-3, 4)  # -3 to +4 octaves (C1 to C8)
        self.octave_spinbox.setValue(self.octave_offset)
        # Display sign for all values
        self.octave_spinbox.setDisplayIntegerBase(10)
        # Set initial prefix
        if self.octave_offset > 0:
            self.octave_spinbox.setPrefix("+")
        self.octave_spinbox.valueChanged.connect(self.update_octave)
        # Connect to the valueChanged signal to update the display prefix
        self.octave_spinbox.valueChanged.connect(self._update_octave_display)
        controls_row1.addWidget(self.octave_spinbox)

        # Number of rows selector
        controls_row1.addWidget(QLabel("Rows:"))
        self.rows_spinbox = QSpinBox()
        self.rows_spinbox.setRange(4, 16)  # Allow 4 to 16 rows
        self.rows_spinbox.setValue(self.num_rows)
        self.rows_spinbox.valueChanged.connect(self.update_num_rows)
        controls_row1.addWidget(self.rows_spinbox)

        controls_row1.addStretch(1)

        # Second row of controls: Start/Stop, Clear, Random
        controls_row2 = QHBoxLayout()
        top_controls.addLayout(controls_row2)

        # Start/Stop button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.clicked.connect(self.toggle_sequencer)
        controls_row2.addWidget(self.start_stop_button)

        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_sequencer)
        controls_row2.addWidget(clear_button)

        # Random fill button
        random_button = QPushButton("Random")
        random_button.clicked.connect(self.random_fill_sequencer)
        controls_row2.addWidget(random_button)

        # Add spacer to push controls to the left
        controls_row2.addStretch(1)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Create a container for the grid that can be rebuilt
        self.grid_container = QVBoxLayout()
        main_layout.addLayout(self.grid_container)

        # Create initial grid
        self.create_step_grid()

        # Create sequencer timer but don't start it
        self.sequencer_timer = QTimer(self)
        self.sequencer_timer.timeout.connect(self.advance_sequence)

    def create_step_grid(self):
        """Create the step sequencer grid with the current number of rows."""
        # Clear existing grid if it exists
        if self.grid_widget:
            # Store the state of buttons before rebuilding
            current_state = self.get_grid_state()

            # Remove the old grid widget
            self.grid_container.removeWidget(self.grid_widget)
            self.grid_widget.deleteLater()

            # Clear button references
            self.step_buttons = []
            self.note_labels = []

        # Create new grid widget and layout
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(2)  # Tighter spacing for grid
        self.grid_container.addWidget(self.grid_widget)

        # Generate notes for current scale and settings
        self.base_notes = self._generate_scale_notes()
        self.sequencer_notes = self.base_notes.copy()

        # Ensure sequencer_steps has the right dimensions
        self.resize_step_array()

        # Add note labels on the left
        self.note_labels = []
        for row in range(self.num_rows):
            # Get the MIDI note number for this row
            row_index = self.num_rows - 1 - row  # Highest note at top
            midi_note = self.sequencer_notes[row_index]

            # Get the note label with appropriate name and octave
            label = QLabel(self.get_note_name_for_midi(midi_note))
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setMinimumWidth(50)  # Ensure enough space for sharps/flats
            self.grid_layout.addWidget(label, row, 0)
            self.note_labels.append(label)

        # Create the grid of buttons
        self.step_buttons = []
        for row in range(self.num_rows):
            row_buttons = []
            for col in range(16):
                button = QPushButton()
                button.setCheckable(True)
                button.setFixedSize(30, 30)
                # Store row and column in the button properties
                button.setProperty("row", self.num_rows - 1 - row)  # Reverse order (highest note at top)
                button.setProperty("col", col)
                button.toggled.connect(self.toggle_step)

                # Set color based on position (alternating groups of 4 for visual rhythm)
                if (col // 4) % 2 == 0:
                    button.setStyleSheet("QPushButton { background-color: #e0e0e0; }")
                else:
                    button.setStyleSheet("QPushButton { background-color: #f0f0f0; }")

                # Check if this step was active before
                if row < len(self.sequencer_steps) and col < len(self.sequencer_steps[0]):
                    if self.sequencer_steps[self.num_rows - 1 - row][col]:
                        button.setChecked(True)
                        button.setStyleSheet("QPushButton { background-color: #80b0ff; }")

                self.grid_layout.addWidget(button, row, col + 1)  # +1 for the note labels
                row_buttons.append(button)
            self.step_buttons.append(row_buttons)

    def get_grid_state(self):
        """Get the current state of the grid buttons."""
        state = []
        for row in range(len(self.step_buttons)):
            row_state = []
            for col in range(len(self.step_buttons[row])):
                button = self.step_buttons[row][col]
                row_state.append(button.isChecked())
            state.append(row_state)
        return state

    def resize_step_array(self):
        """Resize the step array to match the current number of rows."""
        # Create a new array with the right dimensions
        new_steps = [[False for _ in range(16)] for _ in range(self.num_rows)]

        # Copy existing states where possible
        for row in range(min(len(self.sequencer_steps), self.num_rows)):
            for col in range(16):
                new_steps[row][col] = self.sequencer_steps[row][col]

        self.sequencer_steps = new_steps

    def _generate_scale_notes(self):
        """Generate MIDI note values for the current scale and octave."""
        # Use the extended scales if we need more than 8 notes
        if self.num_rows <= 8:
            scale_intervals = self.SCALES[self.current_scale][:self.num_rows]
        else:
            scale_intervals = self.EXTENDED_SCALES[self.current_scale][:self.num_rows]

        return [self.root_note + interval + (self.octave_offset * 12) for interval in scale_intervals]

    def get_note_name_for_midi(self, midi_note):
        """Get proper note name with octave for any MIDI note based on current scale and root."""
        # Calculate octave using standard convention where middle C (MIDI 60) is C4
        octave = (midi_note // 12) - 1  # Standard octave numbering: C4 = MIDI 60

        # Get the note position in the chromatic scale (0-11)
        note_position = midi_note % 12

        # Determine if we should use sharps or flats based on root and scale
        use_sharps = True  # Default

        # Consider the root note (some keys typically use flats, others use sharps)
        if self.root_note_name in self.FLAT_KEYS:
            use_sharps = False
        elif self.root_note_name in self.SHARP_KEYS:
            use_sharps = True

        # Override based on scale preference
        scale_prefers_sharps = self.SCALE_SHARPS_PREFERENCE.get(self.current_scale, True)
        if scale_prefers_sharps:
            use_sharps = True
        else:
            use_sharps = False

        # Special cases for accidentals in certain scales:
        # For example, F Major uses Bb not A#
        root_pos = self.root_note % 12

        # Get the possible note names
        note_names = self.CHROMATIC_NOTES[note_position]

        # Choose the appropriate name based on our sharps/flats decision
        if len(note_names) > 1:  # It's an accidental with two possible names
            note_name = note_names[0] if use_sharps else note_names[1]  # 0 is sharp, 1 is flat
        else:
            note_name = note_names[0]  # Only one name available

        # Special case handling for certain scales/keys
        # This handles specific musical conventions when traditional
        # sharp/flat preference doesn't match scale degree requirements

        # Major scale special cases
        if self.current_scale == 'Major':
            # F Major uses Bb not A#
            if root_pos == 5 and note_position == 10:  # F root, and we're looking at A#/Bb
                note_name = "Bb"
            # C Major doesn't have any accidentals

        # Minor scale special cases (especially harmonic minor)
        elif self.current_scale == 'Minor':
            # A Minor uses standard notes with no accidentals
            if root_pos == 9 and note_position in [1, 3, 6, 8, 10]:  # A root
                # These would be the black keys
                note_name = note_names[1] if use_sharps else note_names[0]  # Invert our choice

        # Return full note name with octave
        return f"{note_name}{octave}"

    def update_sequencer_bpm(self, bpm):
        """Update the sequencer BPM and timing."""
        self.bpm = bpm
        # Calculate step interval in milliseconds
        # 60000 ms / BPM = milliseconds per beat
        # We want 16th notes so divide by 4
        step_interval = int(60000 / bpm / 4)
        # Calculate step duration
        self.step_duration = step_interval / 1000

        if self.sequencer_running:
            # Update timer if running
            self.sequencer_timer.setInterval(step_interval)

    def update_root_note(self, root_name):
        """Update the root note and regenerate notes."""
        self.root_note_name = root_name
        self.root_note = self.ROOT_NOTES[root_name]

        # Regenerate notes for the new root
        self.base_notes = self._generate_scale_notes()
        self.sequencer_notes = self.base_notes.copy()

        # Update note labels to show correct names
        self.update_note_labels()

    def update_scale(self, scale_name):
        """Update the current scale and regenerate notes."""
        self.current_scale = scale_name

        # Regenerate notes for the new scale
        self.base_notes = self._generate_scale_notes()
        self.sequencer_notes = self.base_notes.copy()

        # Update note labels to show correct names
        self.update_note_labels()

    def update_note_labels(self):
        """Update all note labels based on current scale and root."""
        for i, label in enumerate(self.note_labels):
            row_index = self.num_rows - 1 - i  # Convert from reversed display order
            midi_note = self.sequencer_notes[row_index]
            label.setText(self.get_note_name_for_midi(midi_note))

    def update_num_rows(self, num_rows):
        """Update the number of rows and rebuild the grid."""
        if num_rows == self.num_rows:
            return  # No change

        self.num_rows = num_rows

        # Rebuild the entire grid
        self.create_step_grid()

    def toggle_sequencer(self):
        """Start or stop the sequencer."""
        if self.sequencer_running:
            # Stop the sequencer
            self.sequencer_timer.stop()
            self.sequencer_running = False
            self.start_stop_button.setText("Start")

            # Reset highlight on all buttons
            for row in range(self.num_rows):
                for col in range(16):
                    button = self.step_buttons[row][col]
                    if button.isChecked():
                        button.setStyleSheet("QPushButton { background-color: #80b0ff; }")
                    else:
                        # Restore original color
                        if (col // 4) % 2 == 0:
                            button.setStyleSheet("QPushButton { background-color: #e0e0e0; }")
                        else:
                            button.setStyleSheet("QPushButton { background-color: #f0f0f0; }")
        else:
            # Start the sequencer
            # Calculate step interval based on BPM
            step_interval = int(60000 / self.bpm / 4)  # 16th notes
            self.sequencer_timer.setInterval(step_interval)
            self.current_step = 15  # Will advance to 0 on first step
            self.sequencer_timer.start()
            self.sequencer_running = True
            self.start_stop_button.setText("Stop")

    def advance_sequence(self):
        """Advance the sequencer to the next step and play notes."""
        # Clear highlight from the current step
        if self.current_step >= 0:
            for row in range(self.num_rows):
                button = self.step_buttons[row][self.current_step]
                if button.isChecked():
                    button.setStyleSheet("QPushButton { background-color: #80b0ff; }")
                else:
                    # Restore original color
                    if (self.current_step // 4) % 2 == 0:
                        button.setStyleSheet("QPushButton { background-color: #e0e0e0; }")
                    else:
                        button.setStyleSheet("QPushButton { background-color: #f0f0f0; }")

        # Move to next step
        self.current_step = (self.current_step + 1) % 16

        # Highlight the current step
        for row in range(self.num_rows):
            button = self.step_buttons[row][self.current_step]
            if button.isChecked():
                button.setStyleSheet("QPushButton { background-color: #4080ff; font-weight: bold; }")
            else:
                button.setStyleSheet("QPushButton { background-color: #a0a0a0; }")

        # Play notes for the current step
        for row in range(self.num_rows):
            if self.sequencer_steps[row][self.current_step]:
                midi_note = self.sequencer_notes[row]
                play_midi_note(midi_note, self.step_duration, 0.8)

    def clear_sequencer(self):
        """Clear all steps in the sequencer."""
        # Reset internal state - clear all steps
        self.sequencer_steps = [[False for _ in range(16)] for _ in range(max(self.num_rows, len(self.sequencer_steps)))]

        # Reset visual state - update all visible buttons
        for row in range(self.num_rows):
            for col in range(16):
                button = self.step_buttons[row][col]
                # Uncheck the button
                button.setChecked(False)

                # Reset to default color based on column position
                if (col // 4) % 2 == 0:
                    button.setStyleSheet("QPushButton { background-color: #e0e0e0; }")
                else:
                    button.setStyleSheet("QPushButton { background-color: #f0f0f0; }")

    def random_fill_sequencer(self):
        """Fill the sequencer with random steps."""
        # Clear existing pattern first
        self.clear_sequencer()

        # Randomly set about 15% of steps
        for row in range(self.num_rows):
            for col in range(16):
                if np.random.random() < 0.15:
                    if row < len(self.sequencer_steps):
                        self.sequencer_steps[row][col] = True
                        button = self.step_buttons[row][col]
                        button.setChecked(True)
                        button.setStyleSheet("QPushButton { background-color: #80b0ff; }")

    def stop(self):
        """Stop the sequencer and clean up."""
        if self.sequencer_running:
            self.sequencer_timer.stop()
            self.sequencer_running = False

    def update_octave(self, value):
        """Update the octave offset and recalculate note values."""
        self.octave_offset = value

        # Regenerate note values
        self.base_notes = self._generate_scale_notes()
        self.sequencer_notes = self.base_notes.copy()

        # Update note labels
        self.update_note_labels()

    def toggle_step(self):
        """Handle toggling a step button in the sequencer grid."""
        button = self.sender()
        if button:
            row = button.property("row")
            col = button.property("col")
            state = button.isChecked()

            # Ensure we're within the valid range
            if row < len(self.sequencer_steps):
                self.sequencer_steps[row][col] = state

                # Update button appearance based on state
                if state:
                    button.setStyleSheet("QPushButton { background-color: #80b0ff; }")
                else:
                    # Restore original color based on column position
                    if (col // 4) % 2 == 0:
                        button.setStyleSheet("QPushButton { background-color: #e0e0e0; }")
                    else:
                        button.setStyleSheet("QPushButton { background-color: #f0f0f0; }")

    def _update_octave_display(self, value):
        """Update the display format of the octave spinbox."""
        if value > 0:
            self.octave_spinbox.setPrefix("+")
        else:
            self.octave_spinbox.setPrefix("")
