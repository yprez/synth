"""Comprehensive tests for the input module."""

from unittest.mock import Mock, patch
from pynput import keyboard

from qwerty_synth import input as input_module
from qwerty_synth import config


class TestKeyMidiMapping:
    """Test cases for key to MIDI note mapping."""

    def test_key_midi_map_completeness(self):
        """Test that all expected keys are mapped to MIDI notes."""
        expected_keys = ['a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'k', 'o', 'l', 'p', ';', "'"]

        for key in expected_keys:
            assert key in input_module.key_midi_map
            assert isinstance(input_module.key_midi_map[key], int)
            assert 60 <= input_module.key_midi_map[key] <= 77  # C4 to F5

    def test_key_midi_map_values(self):
        """Test specific MIDI note mappings."""
        assert input_module.key_midi_map['a'] == 60  # C4
        assert input_module.key_midi_map['h'] == 69  # A4 (440Hz)
        assert input_module.key_midi_map['k'] == 72  # C5

    def test_chromatic_sequence(self):
        """Test that mapped keys form a chromatic sequence."""
        # Test a subset of the chromatic sequence
        assert input_module.key_midi_map['a'] == 60   # C4
        assert input_module.key_midi_map['w'] == 61   # C#4
        assert input_module.key_midi_map['s'] == 62   # D4
        assert input_module.key_midi_map['e'] == 63   # D#4


class TestOnPress:
    """Test cases for key press handling."""

    def setup_method(self):
        """Reset state before each test."""
        config.active_notes = {}
        config.mono_pressed_keys = []
        config.mono_mode = False
        config.octave_offset = 0
        config.octave_min = -2
        config.octave_max = 3
        config.waveform_type = 'sine'
        input_module.gui_instance = None

    @patch('qwerty_synth.controller.midi_to_freq')
    @patch('qwerty_synth.synth.Oscillator')
    def test_on_press_polyphonic_mode(self, mock_oscillator, mock_midi_to_freq):
        """Test key press in polyphonic mode."""
        config.mono_mode = False
        mock_midi_to_freq.return_value = 440.0
        mock_osc = Mock()
        mock_osc.key = 'a'
        mock_oscillator.return_value = mock_osc

        # Create mock key
        mock_key = Mock()
        mock_key.char = 'a'

        input_module.on_press(mock_key)

        # Should create oscillator and add to active notes
        mock_midi_to_freq.assert_called_once_with(60)  # a = 60 + 0 offset
        mock_oscillator.assert_called_once_with(440.0, 'sine')
        assert 'a' in config.active_notes
        assert config.active_notes['a'] == mock_osc

    @patch('qwerty_synth.controller.midi_to_freq')
    @patch('qwerty_synth.synth.Oscillator')
    def test_on_press_mono_mode_new_note(self, mock_oscillator, mock_midi_to_freq):
        """Test key press in mono mode with no existing note."""
        config.mono_mode = True
        mock_midi_to_freq.return_value = 440.0
        mock_osc = Mock()
        mock_osc.key = 'mono'
        mock_osc.released = False
        mock_oscillator.return_value = mock_osc

        mock_key = Mock()
        mock_key.char = 'a'

        input_module.on_press(mock_key)

        # Should create mono oscillator
        mock_oscillator.assert_called_once_with(440.0, 'sine')
        assert 'mono' in config.active_notes
        assert 'a' in config.mono_pressed_keys

    @patch('qwerty_synth.controller.midi_to_freq')
    def test_on_press_mono_mode_update_frequency(self, mock_midi_to_freq):
        """Test key press in mono mode with existing note."""
        config.mono_mode = True
        mock_midi_to_freq.return_value = 550.0

        # Create existing mono oscillator
        existing_osc = Mock()
        existing_osc.released = False
        config.active_notes['mono'] = existing_osc

        mock_key = Mock()
        mock_key.char = 's'  # D4

        input_module.on_press(mock_key)

        # Should update target frequency
        assert existing_osc.target_freq == 550.0
        assert 's' in config.mono_pressed_keys

    def test_on_press_octave_down(self):
        """Test octave down key press."""
        config.octave_offset = 12  # Start at +1 octave

        mock_key = Mock()
        mock_key.char = 'z'

        with patch('builtins.print') as mock_print:
            input_module.on_press(mock_key)

        assert config.octave_offset == 0  # Should decrease by 12
        mock_print.assert_called_once_with('Octave down: +0')

    def test_on_press_octave_up(self):
        """Test octave up key press."""
        config.octave_offset = 0  # Start at middle octave

        mock_key = Mock()
        mock_key.char = 'x'

        with patch('builtins.print') as mock_print:
            input_module.on_press(mock_key)

        assert config.octave_offset == 12  # Should increase by 12
        mock_print.assert_called_once_with('Octave up: +1')

    def test_on_press_octave_limits(self):
        """Test octave change limits."""
        # Test lower limit
        config.octave_offset = 12 * config.octave_min  # At minimum
        mock_key = Mock()
        mock_key.char = 'z'

        input_module.on_press(mock_key)
        assert config.octave_offset == 12 * config.octave_min  # Should not change

        # Test upper limit
        config.octave_offset = 12 * config.octave_max  # At maximum
        mock_key.char = 'x'

        input_module.on_press(mock_key)
        assert config.octave_offset == 12 * config.octave_max  # Should not change

    def test_on_press_with_gui_instance(self):
        """Test key press with GUI instance present."""
        # Create mock GUI instance
        mock_gui = Mock()
        mock_gui.running = True
        mock_gui.octave_label = Mock()
        mock_gui.octave_dial = Mock()
        input_module.gui_instance = mock_gui

        config.octave_offset = 0
        mock_key = Mock()
        mock_key.char = 'x'

        input_module.on_press(mock_key)

        # Should update config but not directly update GUI elements
        assert config.octave_offset == 12
        # GUI updates are now handled by the GUI's update_plots method

    def test_on_press_with_octave_offset(self):
        """Test key press with octave offset applied."""
        config.octave_offset = 12  # +1 octave

        mock_key = Mock()
        mock_key.char = 'a'  # C4 normally

        with patch('qwerty_synth.controller.midi_to_freq') as mock_midi_to_freq:
            with patch('qwerty_synth.synth.Oscillator'):
                mock_midi_to_freq.return_value = 523.25  # C5

                input_module.on_press(mock_key)

                # Should call with offset MIDI note
                mock_midi_to_freq.assert_called_once_with(72)  # 60 + 12

    @patch('sounddevice.stop')
    def test_on_press_escape_key(self, mock_sd_stop):
        """Test escape key press without GUI."""
        mock_key = keyboard.Key.esc

        with patch('builtins.print') as mock_print:
            result = input_module.on_press(mock_key)

        mock_print.assert_called_once_with('Exiting...')
        mock_sd_stop.assert_called_once()
        assert result is False  # Should return False to stop listener

    @patch('sounddevice.stop')
    @patch('PyQt5.QtCore.QTimer')
    def test_on_press_escape_with_gui(self, mock_qtimer, mock_sd_stop):
        """Test escape key press with GUI instance."""
        mock_gui = Mock()
        input_module.gui_instance = mock_gui
        mock_key = keyboard.Key.esc

        with patch('builtins.print'):
            result = input_module.on_press(mock_key)

        # Should use QTimer.singleShot to schedule close on main thread
        mock_qtimer.singleShot.assert_called_once_with(0, mock_gui.close)
        assert result is False

    def test_on_press_unknown_key(self):
        """Test pressing an unmapped key."""
        mock_key = Mock()
        mock_key.char = 'q'  # Not in key_midi_map

        # Should not raise exception and not modify state
        input_module.on_press(mock_key)
        assert len(config.active_notes) == 0
        assert len(config.mono_pressed_keys) == 0

    def test_on_press_special_key_without_char(self):
        """Test pressing a special key without char attribute."""
        mock_key = Mock()
        del mock_key.char  # Remove char attribute

        # Should handle AttributeError gracefully
        input_module.on_press(mock_key)
        assert len(config.active_notes) == 0


class TestOnRelease:
    """Test cases for key release handling."""

    def setup_method(self):
        """Reset state before each test."""
        config.active_notes = {}
        config.mono_pressed_keys = []
        config.mono_mode = False

    def test_on_release_polyphonic_mode(self):
        """Test key release in polyphonic mode."""
        config.mono_mode = False

        # Create mock oscillator
        mock_osc = Mock()
        mock_osc.released = False
        config.active_notes['a'] = mock_osc

        mock_key = Mock()
        mock_key.char = 'a'

        input_module.on_release(mock_key)

        # Should release the oscillator
        assert mock_osc.released is True
        assert mock_osc.env_time == 0.0
        assert mock_osc.lfo_env_time == 0.0

    def test_on_release_mono_mode_last_key(self):
        """Test key release in mono mode when it's the last key."""
        config.mono_mode = True
        config.mono_pressed_keys = ['a']

        # Create mock mono oscillator
        mock_osc = Mock()
        mock_osc.released = False
        config.active_notes['mono'] = mock_osc

        mock_key = Mock()
        mock_key.char = 'a'

        input_module.on_release(mock_key)

        # Should release mono oscillator and remove key from list
        assert mock_osc.released is True
        assert mock_osc.env_time == 0.0
        assert mock_osc.lfo_env_time == 0.0
        assert 'a' not in config.mono_pressed_keys

    @patch('qwerty_synth.controller.midi_to_freq')
    def test_on_release_mono_mode_switch_note(self, mock_midi_to_freq):
        """Test key release in mono mode with other keys still pressed."""
        config.mono_mode = True
        config.mono_pressed_keys = ['a', 's']  # Two keys pressed
        mock_midi_to_freq.return_value = 293.66  # D4

        # Create mock mono oscillator
        mock_osc = Mock()
        mock_osc.released = False
        config.active_notes['mono'] = mock_osc

        mock_key = Mock()
        mock_key.char = 'a'  # Release first key

        input_module.on_release(mock_key)

        # Should switch to the last pressed key (s)
        assert mock_osc.target_freq == 293.66
        assert mock_osc.key == 's'
        assert mock_osc.lfo_env_time == 0.0
        assert 'a' not in config.mono_pressed_keys
        assert 's' in config.mono_pressed_keys

    def test_on_release_key_not_in_active_notes(self):
        """Test releasing a key that's not in active notes."""
        mock_key = Mock()
        mock_key.char = 'a'

        # Should handle gracefully
        input_module.on_release(mock_key)
        assert len(config.active_notes) == 0

    def test_on_release_key_not_in_mono_pressed_keys(self):
        """Test releasing a key that's not in mono_pressed_keys."""
        config.mono_mode = True
        config.mono_pressed_keys = ['s']  # Different key

        mock_key = Mock()
        mock_key.char = 'a'

        input_module.on_release(mock_key)
        assert config.mono_pressed_keys == ['s']  # Should remain unchanged

    def test_on_release_special_key_without_char(self):
        """Test releasing a special key without char attribute."""
        mock_key = Mock()
        del mock_key.char  # Remove char attribute

        # Should handle AttributeError gracefully
        input_module.on_release(mock_key)
        assert len(config.active_notes) == 0

    def test_on_release_mono_mode_no_mono_oscillator(self):
        """Test mono mode release when no mono oscillator exists."""
        config.mono_mode = True
        config.mono_pressed_keys = ['a']

        mock_key = Mock()
        mock_key.char = 'a'

        # Should handle gracefully
        input_module.on_release(mock_key)
        assert 'a' not in config.mono_pressed_keys


class TestKeyboardListener:
    """Test cases for keyboard listener functionality."""

    @patch('qwerty_synth.input.keyboard.Listener')
    def test_run_keyboard_listener(self, mock_listener_class):
        """Test running the keyboard listener."""
        mock_listener = Mock()
        mock_listener_class.return_value.__enter__ = Mock(return_value=mock_listener)
        mock_listener_class.return_value.__exit__ = Mock(return_value=None)

        input_module.run_keyboard_listener()

        # Should create listener with correct callbacks
        mock_listener_class.assert_called_once_with(
            on_press=input_module.on_press,
            on_release=input_module.on_release
        )
        mock_listener.join.assert_called_once()

    @patch('qwerty_synth.input.run_keyboard_listener')
    @patch('threading.Thread')
    def test_start_keyboard_input(self, mock_thread, mock_run_listener):
        """Test starting keyboard input in a thread."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        result = input_module.start_keyboard_input()

        # Should create and start daemon thread
        mock_thread.assert_called_once_with(
            target=input_module.run_keyboard_listener,
            daemon=True
        )
        mock_thread_instance.start.assert_called_once()
        assert result == mock_thread_instance


class TestGuiIntegration:
    """Test cases for GUI integration."""

    def setup_method(self):
        """Reset state before each test."""
        input_module.gui_instance = None
        config.octave_offset = 0

    def test_gui_instance_assignment(self):
        """Test that gui_instance can be assigned."""
        mock_gui = Mock()
        input_module.gui_instance = mock_gui
        assert input_module.gui_instance == mock_gui

    def test_octave_change_without_gui(self):
        """Test octave change when no GUI instance is set."""
        input_module.gui_instance = None
        config.octave_offset = 0

        mock_key = Mock()
        mock_key.char = 'x'

        # Should not raise exception
        with patch('builtins.print'):
            input_module.on_press(mock_key)

        assert config.octave_offset == 12

    def test_octave_change_with_non_running_gui(self):
        """Test octave change when GUI instance is not running."""
        mock_gui = Mock()
        mock_gui.running = False
        input_module.gui_instance = mock_gui

        config.octave_offset = 0
        mock_key = Mock()
        mock_key.char = 'x'

        with patch('builtins.print'):
            input_module.on_press(mock_key)

        # Should update config regardless of GUI state
        assert config.octave_offset == 12
        # GUI updates are handled by the GUI's own update cycle, not by input module


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def setup_method(self):
        """Reset state before each test."""
        config.active_notes = {}
        config.mono_pressed_keys = []
        config.mono_mode = False
        config.octave_offset = 0

    def test_case_insensitive_key_handling(self):
        """Test that uppercase keys are handled correctly."""
        mock_key = Mock()
        mock_key.char = 'A'  # Uppercase

        with patch('qwerty_synth.controller.midi_to_freq') as mock_midi_to_freq:
            with patch('qwerty_synth.synth.Oscillator'):
                mock_midi_to_freq.return_value = 440.0

                input_module.on_press(mock_key)

                # Should convert to lowercase and process
                assert 'a' in config.active_notes

    def test_concurrent_key_presses(self):
        """Test handling multiple concurrent key presses."""
        config.mono_mode = False

        with patch('qwerty_synth.controller.midi_to_freq') as mock_midi_to_freq:
            with patch('qwerty_synth.synth.Oscillator') as mock_oscillator:
                mock_midi_to_freq.return_value = 440.0
                mock_oscillator.side_effect = [Mock(key='a'), Mock(key='s'), Mock(key='d')]

                # Press multiple keys
                for char in ['a', 's', 'd']:
                    mock_key = Mock()
                    mock_key.char = char
                    input_module.on_press(mock_key)

                assert len(config.active_notes) == 3
                assert all(key in config.active_notes for key in ['a', 's', 'd'])

    def test_mono_mode_key_order_tracking(self):
        """Test that mono mode correctly tracks key press order."""
        config.mono_mode = True

        with patch('qwerty_synth.controller.midi_to_freq'):
            with patch('qwerty_synth.synth.Oscillator'):
                # Press keys in sequence
                for char in ['a', 's', 'd']:
                    mock_key = Mock()
                    mock_key.char = char
                    input_module.on_press(mock_key)

                # Should track all pressed keys in order
                assert config.mono_pressed_keys == ['a', 's', 'd']

    def test_extreme_octave_values(self):
        """Test handling of extreme octave offset values."""
        # Test with values at the boundaries
        config.octave_offset = 12 * config.octave_min
        mock_key = Mock()
        mock_key.char = 'z'

        input_module.on_press(mock_key)
        assert config.octave_offset == 12 * config.octave_min  # Should not go below minimum

        config.octave_offset = 12 * config.octave_max
        mock_key.char = 'x'

        input_module.on_press(mock_key)
        assert config.octave_offset == 12 * config.octave_max  # Should not go above maximum
