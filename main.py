"""Backward-compatible entry point for running the QWERTY Synth GUI."""

from qwerty_synth.entry import main as _launch_gui


def main():
    """Launch the QWERTY Synth application."""
    _launch_gui()


if __name__ == "__main__":
    main()
