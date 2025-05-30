"""Patch management for QWERTY Synth.

This module provides functionality to save, load, list, and delete patches
which are snapshots of all synth parameters.
"""

import os
import yaml
import datetime
from pathlib import Path
from typing import Dict, List, Any

from qwerty_synth import config

# Default patches directory is in user's config directory
DEFAULT_DIR = Path.home() / ".config" / "qwerty_synth" / "patches"

# Override with environment variable if set
if "QWERTY_PATCH_DIR" in os.environ:
    DEFAULT_DIR = Path(os.environ["QWERTY_PATCH_DIR"])

# Schema version for future-proofing
SCHEMA_VERSION = 1


class PatchError(Exception):
    """Exception raised for patch-related errors."""
    pass


def collect_params() -> Dict[str, Any]:
    """Collect all mutable parameters from config into a dictionary.

    Returns:
        Dict containing all mutable synth parameters
    """
    params = {
        # Basic settings
        "waveform_type": config.waveform_type,
        "octave_offset": config.octave_offset,
        "volume": config.volume,

        # ADSR envelope
        "adsr": dict(config.adsr),

        # Filter settings
        "filter_enabled": config.filter_enabled,
        "filter_type": config.filter_type,
        "filter_topology": config.filter_topology,
        "filter_slope": config.filter_slope,
        "filter_cutoff": config.filter_cutoff,
        "filter_resonance": config.filter_resonance,
        "filter_env_amount": config.filter_env_amount,
        "filter_adsr": dict(config.filter_adsr),

        # Drive settings
        "drive_on": config.drive_on,
        "drive_gain": config.drive_gain,
        "drive_type": config.drive_type,
        "drive_tone": config.drive_tone,
        "drive_mix": config.drive_mix,
        "drive_asymmetry": config.drive_asymmetry,

        # Mono mode and portamento
        "mono_mode": config.mono_mode,
        "glide_time": config.glide_time,

        # LFO settings
        "lfo_enabled": config.lfo_enabled,
        "lfo_rate": config.lfo_rate,
        "lfo_depth": config.lfo_depth,
        "lfo_target": config.lfo_target,
        "lfo_attack_time": config.lfo_attack_time,
        "lfo_delay_time": config.lfo_delay_time,

        # Delay settings
        "delay_enabled": config.delay_enabled,
        "delay_time_ms": config.delay_time_ms,
        "delay_feedback": config.delay_feedback,
        "delay_mix": config.delay_mix,
        "delay_sync_enabled": config.delay_sync_enabled,
        "delay_pingpong": config.delay_pingpong,
        "delay_division": config.delay_division,

        # Chorus settings
        "chorus_enabled": config.chorus_enabled,
        "chorus_rate": config.chorus_rate,
        "chorus_depth": config.chorus_depth,
        "chorus_mix": config.chorus_mix,
        "chorus_voices": config.chorus_voices,

        # Arpeggiator settings
        "arpeggiator_enabled": config.arpeggiator_enabled,
        "arpeggiator_pattern": config.arpeggiator_pattern,
        "arpeggiator_rate": config.arpeggiator_rate,
        "arpeggiator_gate": config.arpeggiator_gate,
        "arpeggiator_octave_range": config.arpeggiator_octave_range,
        "arpeggiator_sync_to_bpm": config.arpeggiator_sync_to_bpm,
        "arpeggiator_sustain_base": config.arpeggiator_sustain_base,
    }

    return params


def apply_params(params: Dict[str, Any]) -> None:
    """Apply parameters from a dictionary to the config.

    Args:
        params: Dictionary containing synth parameters

    Raises:
        PatchError: If required parameters are missing or invalid
    """
    try:
        # Basic settings
        if "waveform_type" in params:
            config.waveform_type = params["waveform_type"]
        if "octave_offset" in params:
            config.octave_offset = params["octave_offset"]
        if "volume" in params:
            config.volume = float(params["volume"])

        # ADSR envelope
        if "adsr" in params:
            adsr_params = params["adsr"]
            for key in ["attack", "decay", "sustain", "release"]:
                if key in adsr_params:
                    config.adsr[key] = float(adsr_params[key])

        # Filter settings
        if "filter_enabled" in params:
            config.filter_enabled = bool(params["filter_enabled"])
        if "filter_type" in params:
            config.filter_type = params["filter_type"]
        if "filter_topology" in params:
            config.filter_topology = params["filter_topology"]
        if "filter_slope" in params:
            config.filter_slope = params["filter_slope"]
        if "filter_cutoff" in params:
            config.filter_cutoff = float(params["filter_cutoff"])
        if "filter_resonance" in params:
            config.filter_resonance = float(params["filter_resonance"])
        if "filter_env_amount" in params:
            config.filter_env_amount = float(params["filter_env_amount"])

        if "filter_adsr" in params:
            filter_adsr_params = params["filter_adsr"]
            for key in ["attack", "decay", "sustain", "release"]:
                if key in filter_adsr_params:
                    config.filter_adsr[key] = float(filter_adsr_params[key])

        # Drive settings
        if "drive_on" in params:
            config.drive_on = bool(params["drive_on"])
        if "drive_gain" in params:
            config.drive_gain = float(params["drive_gain"])
        if "drive_type" in params:
            config.drive_type = params["drive_type"]
        if "drive_tone" in params:
            config.drive_tone = float(params["drive_tone"])
        if "drive_mix" in params:
            config.drive_mix = float(params["drive_mix"])
        if "drive_asymmetry" in params:
            config.drive_asymmetry = float(params["drive_asymmetry"])

        # Mono mode and portamento
        if "mono_mode" in params:
            config.mono_mode = bool(params["mono_mode"])
        if "glide_time" in params:
            config.glide_time = float(params["glide_time"])

        # LFO settings
        if "lfo_enabled" in params:
            config.lfo_enabled = bool(params["lfo_enabled"])
        if "lfo_rate" in params:
            config.lfo_rate = float(params["lfo_rate"])
        if "lfo_depth" in params:
            config.lfo_depth = float(params["lfo_depth"])
        if "lfo_target" in params:
            config.lfo_target = params["lfo_target"]
        if "lfo_attack_time" in params:
            config.lfo_attack_time = float(params["lfo_attack_time"])
        if "lfo_delay_time" in params:
            config.lfo_delay_time = float(params["lfo_delay_time"])

        # Delay settings
        if "delay_enabled" in params:
            config.delay_enabled = bool(params["delay_enabled"])
        if "delay_time_ms" in params:
            config.delay_time_ms = float(params["delay_time_ms"])
        if "delay_feedback" in params:
            config.delay_feedback = float(params["delay_feedback"])
        if "delay_mix" in params:
            config.delay_mix = float(params["delay_mix"])
        if "delay_sync_enabled" in params:
            config.delay_sync_enabled = bool(params["delay_sync_enabled"])
        if "delay_pingpong" in params:
            config.delay_pingpong = bool(params["delay_pingpong"])
        if "delay_division" in params:
            config.delay_division = params["delay_division"]

        # Chorus settings
        if "chorus_enabled" in params:
            config.chorus_enabled = bool(params["chorus_enabled"])
        if "chorus_rate" in params:
            config.chorus_rate = float(params["chorus_rate"])
        if "chorus_depth" in params:
            config.chorus_depth = float(params["chorus_depth"])
        if "chorus_mix" in params:
            config.chorus_mix = float(params["chorus_mix"])
        if "chorus_voices" in params:
            config.chorus_voices = int(params["chorus_voices"])

        # Arpeggiator settings
        if "arpeggiator_enabled" in params:
            config.arpeggiator_enabled = bool(params["arpeggiator_enabled"])
        if "arpeggiator_pattern" in params:
            config.arpeggiator_pattern = params["arpeggiator_pattern"]
        if "arpeggiator_rate" in params:
            config.arpeggiator_rate = float(params["arpeggiator_rate"])
        if "arpeggiator_gate" in params:
            config.arpeggiator_gate = float(params["arpeggiator_gate"])
        if "arpeggiator_octave_range" in params:
            config.arpeggiator_octave_range = params["arpeggiator_octave_range"]
        if "arpeggiator_sync_to_bpm" in params:
            config.arpeggiator_sync_to_bpm = bool(params["arpeggiator_sync_to_bpm"])
        if "arpeggiator_sustain_base" in params:
            config.arpeggiator_sustain_base = bool(params["arpeggiator_sustain_base"])

    except (ValueError, TypeError, KeyError) as e:
        raise PatchError(f"Error applying parameters: {str(e)}")


def save_patch(name: str, dir_: Path = DEFAULT_DIR) -> Path:
    """Save current synth parameters as a patch.

    Args:
        name: Name of the patch
        dir_: Directory to save the patch in (defaults to DEFAULT_DIR)

    Returns:
        Path to the saved patch file

    Raises:
        PatchError: If patch cannot be saved
    """
    try:
        # Ensure directory exists
        dir_ = Path(dir_)
        dir_.mkdir(parents=True, exist_ok=True)

        # Create timestamp in ISO 8601 format with Z suffix for UTC
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        # Prepare patch data structure
        patch_data = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created": timestamp,
            "params": collect_params()
        }

        # Generate safe filename from name
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        safe_name = safe_name.replace(" ", "_")
        filename = f"{safe_name}.yml"
        path = dir_ / filename

        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(patch_data, f, default_flow_style=False, allow_unicode=True)

        return path

    except (OSError, yaml.YAMLError) as e:
        raise PatchError(f"Failed to save patch: {str(e)}")


def load_patch(path: Path) -> Dict[str, Any]:
    """Load a patch from file and apply it.

    Args:
        path: Path to the patch file

    Returns:
        Dictionary containing the loaded patch data

    Raises:
        PatchError: If patch cannot be loaded or applied
    """
    try:
        # Ensure path is a Path object
        path = Path(path)

        # Load YAML file
        with open(path, 'r', encoding='utf-8') as f:
            patch_data = yaml.safe_load(f)

        # Check schema version
        schema_version = patch_data.get("schema_version", 1)
        if schema_version > SCHEMA_VERSION:
            raise PatchError(f"Patch uses newer schema version: {schema_version}")

        # Apply parameters
        if "params" not in patch_data:
            raise PatchError("Invalid patch file: missing 'params' key")

        apply_params(patch_data["params"])

        return patch_data

    except (OSError, yaml.YAMLError) as e:
        raise PatchError(f"Failed to load patch: {str(e)}")


def list_patches(dir_: Path = DEFAULT_DIR) -> List[Dict[str, Any]]:
    """List available patches.

    Args:
        dir_: Directory to list patches from (defaults to DEFAULT_DIR)

    Returns:
        List of dictionaries with patch metadata
    """
    patches = []

    # Ensure dir_ is a Path object
    dir_ = Path(dir_)

    # Create directory if it doesn't exist
    if not dir_.exists():
        dir_.mkdir(parents=True, exist_ok=True)
        return patches

    # Get all .yml files
    for path in dir_.glob("*.yml"):
        try:
            # Read just the metadata, not the full patch
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            patches.append({
                "name": data.get("name", path.stem),
                "created": data.get("created", "Unknown"),
                "path": str(path),
                "filename": path.name
            })
        except (OSError, yaml.YAMLError):
            # Skip files that can't be read
            continue

    # Sort by name
    patches.sort(key=lambda x: x["name"].lower())

    return patches


def rename_patch(path: Path, new_name: str) -> Path:
    """Rename a patch.

    Args:
        path: Path to the patch file
        new_name: New name for the patch

    Returns:
        Path to the renamed patch file

    Raises:
        PatchError: If patch cannot be renamed
    """
    try:
        # Ensure path is a Path object
        path = Path(path)

        # Load existing patch
        with open(path, 'r', encoding='utf-8') as f:
            patch_data = yaml.safe_load(f)

        # Update name
        patch_data["name"] = new_name

        # Write back to original file
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(patch_data, f, default_flow_style=False, allow_unicode=True)

        # Generate new filename
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in new_name)
        safe_name = safe_name.replace(" ", "_")
        new_filename = f"{safe_name}.yml"
        new_path = path.parent / new_filename

        # Rename file if new filename is different
        if new_path != path:
            path.rename(new_path)

        return new_path

    except (OSError, yaml.YAMLError) as e:
        raise PatchError(f"Failed to rename patch: {str(e)}")


def delete_patch(path: Path) -> None:
    """Delete a patch.

    Args:
        path: Path to the patch file

    Raises:
        PatchError: If patch cannot be deleted
    """
    try:
        # Ensure path is a Path object
        path = Path(path)

        # Delete the file
        path.unlink()
    except OSError as e:
        raise PatchError(f"Failed to delete patch: {str(e)}")
