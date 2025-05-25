"""Tests for patch management functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path

from qwerty_synth import config
from qwerty_synth.patch import (
    collect_params, apply_params, save_patch, load_patch,
    list_patches, rename_patch, delete_patch, PatchError
)


class TestPatchManagement:
    """Test cases for patch management functionality."""

    def setup_method(self):
        """Set up test state before each test."""
        # Store original config state
        self.original_config = {
            'waveform_type': config.waveform_type,
            'octave_offset': config.octave_offset,
            'volume': config.volume,
            'adsr': dict(config.adsr),
            'filter_enabled': config.filter_enabled,
            'filter_type': config.filter_type,
            'filter_topology': config.filter_topology,
            'filter_slope': config.filter_slope,
            'filter_cutoff': config.filter_cutoff,
            'filter_resonance': config.filter_resonance,
            'filter_env_amount': config.filter_env_amount,
            'filter_adsr': dict(config.filter_adsr),
            'drive_on': config.drive_on,
            'drive_gain': config.drive_gain,
            'drive_type': config.drive_type,
            'drive_tone': config.drive_tone,
            'drive_mix': config.drive_mix,
            'drive_asymmetry': config.drive_asymmetry,
            'mono_mode': config.mono_mode,
            'glide_time': config.glide_time,
            'lfo_enabled': config.lfo_enabled,
            'lfo_rate': config.lfo_rate,
            'lfo_depth': config.lfo_depth,
            'lfo_target': config.lfo_target,
            'lfo_attack_time': config.lfo_attack_time,
            'lfo_delay_time': config.lfo_delay_time,
            'delay_enabled': config.delay_enabled,
            'delay_time_ms': config.delay_time_ms,
            'delay_feedback': config.delay_feedback,
            'delay_mix': config.delay_mix,
            'delay_sync_enabled': config.delay_sync_enabled,
            'delay_pingpong': config.delay_pingpong,
            'delay_division': config.delay_division,
            'chorus_enabled': config.chorus_enabled,
            'chorus_rate': config.chorus_rate,
            'chorus_depth': config.chorus_depth,
            'chorus_mix': config.chorus_mix,
            'chorus_voices': config.chorus_voices,
        }

    def teardown_method(self):
        """Restore original config state after each test."""
        # Restore config state
        config.waveform_type = self.original_config['waveform_type']
        config.octave_offset = self.original_config['octave_offset']
        config.volume = self.original_config['volume']
        config.adsr.update(self.original_config['adsr'])
        config.filter_enabled = self.original_config['filter_enabled']
        config.filter_type = self.original_config['filter_type']
        config.filter_topology = self.original_config['filter_topology']
        config.filter_slope = self.original_config['filter_slope']
        config.filter_cutoff = self.original_config['filter_cutoff']
        config.filter_resonance = self.original_config['filter_resonance']
        config.filter_env_amount = self.original_config['filter_env_amount']
        config.filter_adsr.update(self.original_config['filter_adsr'])
        config.drive_on = self.original_config['drive_on']
        config.drive_gain = self.original_config['drive_gain']
        config.drive_type = self.original_config['drive_type']
        config.drive_tone = self.original_config['drive_tone']
        config.drive_mix = self.original_config['drive_mix']
        config.drive_asymmetry = self.original_config['drive_asymmetry']
        config.mono_mode = self.original_config['mono_mode']
        config.glide_time = self.original_config['glide_time']
        config.lfo_enabled = self.original_config['lfo_enabled']
        config.lfo_rate = self.original_config['lfo_rate']
        config.lfo_depth = self.original_config['lfo_depth']
        config.lfo_target = self.original_config['lfo_target']
        config.lfo_attack_time = self.original_config['lfo_attack_time']
        config.lfo_delay_time = self.original_config['lfo_delay_time']
        config.delay_enabled = self.original_config['delay_enabled']
        config.delay_time_ms = self.original_config['delay_time_ms']
        config.delay_feedback = self.original_config['delay_feedback']
        config.delay_mix = self.original_config['delay_mix']
        config.delay_sync_enabled = self.original_config['delay_sync_enabled']
        config.delay_pingpong = self.original_config['delay_pingpong']
        config.delay_division = self.original_config['delay_division']
        config.chorus_enabled = self.original_config['chorus_enabled']
        config.chorus_rate = self.original_config['chorus_rate']
        config.chorus_depth = self.original_config['chorus_depth']
        config.chorus_mix = self.original_config['chorus_mix']
        config.chorus_voices = self.original_config['chorus_voices']

    def test_collect_params(self):
        """Test collection of parameters from config."""
        # Modify some config values
        config.waveform_type = 'square'
        config.volume = 0.8
        config.filter_enabled = True
        config.adsr['attack'] = 0.05

        params = collect_params()

        # Check that parameters were collected correctly
        assert params['waveform_type'] == 'square'
        assert params['volume'] == 0.8
        assert params['filter_enabled'] is True
        assert params['adsr']['attack'] == 0.05

        # Check that all expected keys are present
        expected_keys = [
            'waveform_type', 'octave_offset', 'volume', 'adsr',
            'filter_enabled', 'filter_type', 'filter_topology', 'filter_slope',
            'filter_cutoff', 'filter_resonance', 'filter_env_amount', 'filter_adsr',
            'drive_on', 'drive_gain', 'drive_type', 'drive_tone', 'drive_mix', 'drive_asymmetry',
            'mono_mode', 'glide_time',
            'lfo_enabled', 'lfo_rate', 'lfo_depth', 'lfo_target', 'lfo_attack_time', 'lfo_delay_time',
            'delay_enabled', 'delay_time_ms', 'delay_feedback', 'delay_mix',
            'delay_sync_enabled', 'delay_pingpong', 'delay_division',
            'chorus_enabled', 'chorus_rate', 'chorus_depth', 'chorus_mix', 'chorus_voices'
        ]

        for key in expected_keys:
            assert key in params

    def test_apply_params_basic_settings(self):
        """Test applying basic synth parameters."""
        params = {
            'waveform_type': 'triangle',
            'octave_offset': 24,
            'volume': 0.6
        }

        apply_params(params)

        assert config.waveform_type == 'triangle'
        assert config.octave_offset == 24
        assert config.volume == 0.6

    def test_apply_params_adsr(self):
        """Test applying ADSR parameters."""
        params = {
            'adsr': {
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.5,
                'release': 0.3
            }
        }

        apply_params(params)

        assert config.adsr['attack'] == 0.1
        assert config.adsr['decay'] == 0.2
        assert config.adsr['sustain'] == 0.5
        assert config.adsr['release'] == 0.3

    def test_apply_params_filter_settings(self):
        """Test applying filter parameters."""
        params = {
            'filter_enabled': True,
            'filter_type': 'highpass',
            'filter_topology': 'biquad',
            'filter_slope': 24,
            'filter_cutoff': 2000.0,
            'filter_resonance': 0.7,
            'filter_env_amount': 3000.0,
            'filter_adsr': {
                'attack': 0.15,
                'decay': 0.25,
                'sustain': 0.6,
                'release': 0.4
            }
        }

        apply_params(params)

        assert config.filter_enabled is True
        assert config.filter_type == 'highpass'
        assert config.filter_topology == 'biquad'
        assert config.filter_slope == 24
        assert config.filter_cutoff == 2000.0
        assert config.filter_resonance == 0.7
        assert config.filter_env_amount == 3000.0
        assert config.filter_adsr['attack'] == 0.15
        assert config.filter_adsr['decay'] == 0.25
        assert config.filter_adsr['sustain'] == 0.6
        assert config.filter_adsr['release'] == 0.4

    def test_apply_params_drive_settings(self):
        """Test applying drive parameters."""
        params = {
            'drive_on': True,
            'drive_gain': 2.5,
            'drive_type': 'fuzz',
            'drive_tone': 0.3,
            'drive_mix': 0.8,
            'drive_asymmetry': 0.4
        }

        apply_params(params)

        assert config.drive_on is True
        assert config.drive_gain == 2.5
        assert config.drive_type == 'fuzz'
        assert config.drive_tone == 0.3
        assert config.drive_mix == 0.8
        assert config.drive_asymmetry == 0.4

    def test_apply_params_lfo_settings(self):
        """Test applying LFO parameters."""
        params = {
            'lfo_enabled': True,
            'lfo_rate': 3.0,
            'lfo_depth': 0.5,
            'lfo_target': 'cutoff',
            'lfo_attack_time': 0.3,
            'lfo_delay_time': 0.15
        }

        apply_params(params)

        assert config.lfo_enabled is True
        assert config.lfo_rate == 3.0
        assert config.lfo_depth == 0.5
        assert config.lfo_target == 'cutoff'
        assert config.lfo_attack_time == 0.3
        assert config.lfo_delay_time == 0.15

    def test_apply_params_delay_settings(self):
        """Test applying delay parameters."""
        params = {
            'delay_enabled': True,
            'delay_time_ms': 250.0,
            'delay_feedback': 0.4,
            'delay_mix': 0.3,
            'delay_sync_enabled': False,
            'delay_pingpong': True,
            'delay_division': '1/8'
        }

        apply_params(params)

        assert config.delay_enabled is True
        assert config.delay_time_ms == 250.0
        assert config.delay_feedback == 0.4
        assert config.delay_mix == 0.3
        assert config.delay_sync_enabled is False
        assert config.delay_pingpong is True
        assert config.delay_division == '1/8'

    def test_apply_params_chorus_settings(self):
        """Test applying chorus parameters."""
        params = {
            'chorus_enabled': True,
            'chorus_rate': 1.5,
            'chorus_depth': 0.015,
            'chorus_mix': 0.4,
            'chorus_voices': 3
        }

        apply_params(params)

        assert config.chorus_enabled is True
        assert config.chorus_rate == 1.5
        assert config.chorus_depth == 0.015
        assert config.chorus_mix == 0.4
        assert config.chorus_voices == 3

    def test_apply_params_partial(self):
        """Test applying only some parameters (partial patch)."""
        original_waveform = config.waveform_type
        original_volume = config.volume

        params = {
            'waveform_type': 'sawtooth'
            # Deliberately omitting volume
        }

        apply_params(params)

        assert config.waveform_type == 'sawtooth'
        assert config.volume == original_volume  # Should remain unchanged

    def test_save_and_load_patch(self):
        """Test saving and loading a complete patch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)

            # Modify some config values
            config.waveform_type = 'square'
            config.volume = 0.75
            config.filter_enabled = True
            config.filter_cutoff = 1500.0
            config.adsr['attack'] = 0.08

            # Save patch
            patch_path = save_patch("test_patch", patch_dir)

            # Verify file was created
            assert patch_path.exists()
            assert patch_path.name == "test_patch.yml"

            # Change config values
            config.waveform_type = 'sine'
            config.volume = 0.5
            config.filter_enabled = False
            config.filter_cutoff = 10000.0
            config.adsr['attack'] = 0.01

            # Load patch
            loaded_params = load_patch(patch_path)
            apply_params(loaded_params)

            # Verify values were restored
            assert config.waveform_type == 'square'
            assert config.volume == 0.75
            assert config.filter_enabled is True
            assert config.filter_cutoff == 1500.0
            assert config.adsr['attack'] == 0.08

    def test_save_patch_metadata(self):
        """Test that patch files contain proper metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)

            patch_path = save_patch("metadata_test", patch_dir)

            # Read raw YAML content
            with open(patch_path, 'r') as f:
                content = yaml.safe_load(f)

            # Check metadata (actual structure used by implementation)
            assert 'name' in content
            assert 'created' in content
            assert 'schema_version' in content
            assert content['name'] == "metadata_test"
            assert content['schema_version'] == 1

            # Check parameters section
            assert 'params' in content

    def test_list_patches(self):
        """Test listing patches in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)

            # Create some patches
            save_patch("patch1", patch_dir)
            save_patch("patch2", patch_dir)
            save_patch("patch3", patch_dir)

            # List patches
            patches = list_patches(patch_dir)

            assert len(patches) == 3
            patch_names = [p['name'] for p in patches]
            assert "patch1" in patch_names
            assert "patch2" in patch_names
            assert "patch3" in patch_names

            # Check patch structure (actual structure used by implementation)
            for patch in patches:
                assert 'name' in patch
                assert 'path' in patch
                assert 'created' in patch
                assert 'filename' in patch

    def test_list_patches_empty_directory(self):
        """Test listing patches in an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)
            patches = list_patches(patch_dir)
            assert patches == []

    def test_list_patches_nonexistent_directory(self):
        """Test listing patches in a nonexistent directory."""
        # Use a path we can actually create/access in temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"
            patches = list_patches(nonexistent_dir)
            assert patches == []

    def test_rename_patch(self):
        """Test renaming a patch file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)

            # Create a patch
            original_path = save_patch("original_name", patch_dir)
            assert original_path.exists()

            # Rename it
            new_path = rename_patch(original_path, "new_name")

            # Check results
            assert not original_path.exists()
            assert new_path.exists()
            assert new_path.name == "new_name.yml"

            # Verify the metadata was updated
            with open(new_path, 'r') as f:
                content = yaml.safe_load(f)
            assert content['name'] == "new_name"

    def test_rename_patch_nonexistent(self):
        """Test renaming a nonexistent patch file."""
        # Use a temporary directory path for a more realistic test
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent.yml"

            with pytest.raises(PatchError, match="Failed to rename patch"):
                rename_patch(nonexistent_path, "new_name")

    def test_rename_patch_existing_target(self):
        """Test renaming to a name that already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)

            # Create two patches
            patch1_path = save_patch("patch1", patch_dir)
            save_patch("patch2", patch_dir)

            # Try to rename patch1 to patch2 (current implementation allows overwrite)
            # The current implementation doesn't check for existing targets
            new_path = rename_patch(patch1_path, "patch2")
            assert new_path.exists()
            assert new_path.name == "patch2.yml"

    def test_delete_patch(self):
        """Test deleting a patch file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir)

            # Create a patch
            patch_path = save_patch("to_delete", patch_dir)
            assert patch_path.exists()

            # Delete it
            delete_patch(patch_path)

            # Verify it's gone
            assert not patch_path.exists()

    def test_delete_patch_nonexistent(self):
        """Test deleting a nonexistent patch file."""
        # Use a temporary directory path for a more realistic test
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent.yml"

            with pytest.raises(PatchError, match="Failed to delete patch"):
                delete_patch(nonexistent_path)

    def test_load_patch_invalid_file(self):
        """Test loading an invalid patch file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = Path(temp_dir) / "invalid.yaml"

            # Create an invalid YAML file
            with open(invalid_file, 'w') as f:
                f.write("invalid: yaml: content: [unclosed")

            with pytest.raises(PatchError, match="Failed to load patch"):
                load_patch(invalid_file)

    def test_load_patch_nonexistent(self):
        """Test loading a nonexistent patch file."""
        # Use a temporary directory path for a more realistic test
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent.yml"

            with pytest.raises(PatchError, match="Failed to load patch"):
                load_patch(nonexistent_path)

    def test_save_patch_creates_directory(self):
        """Test that save_patch creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_dir = Path(temp_dir) / "nested" / "patches"
            assert not patch_dir.exists()

            patch_path = save_patch("test_patch", patch_dir)

            assert patch_dir.exists()
            assert patch_path.exists()

    def test_type_conversion_in_apply_params(self):
        """Test that apply_params properly converts string numbers to floats."""
        params = {
            'volume': "0.75",  # String instead of float
            'filter_cutoff': "2000",  # String instead of float
            'adsr': {
                'attack': "0.1",  # String instead of float
                'sustain': "0.5"
            }
        }

        apply_params(params)

        assert config.volume == 0.75
        assert config.filter_cutoff == 2000.0
        assert config.adsr['attack'] == 0.1
        assert config.adsr['sustain'] == 0.5
