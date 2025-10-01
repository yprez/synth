"""MIDI controller input handling for QWERTY Synth."""

import logging
import threading
import time

import mido

from qwerty_synth.keyboard_midi import MidiEvent, MidiEventDispatcher

LOGGER = logging.getLogger(__name__)


def list_midi_ports() -> list[str]:
    """Return a list of available MIDI input port names."""
    try:
        return mido.get_input_names()
    except Exception as exc:
        LOGGER.exception('Failed to enumerate MIDI ports: %s', exc)
        return []


class MidiPortTranslator:
    """Translate external MIDI controller input into internal MIDI events."""

    def __init__(
        self,
        dispatcher: MidiEventDispatcher,
        port_name: str | None = None,
    ) -> None:
        """
        Initialize the MIDI port translator.

        Args:
            dispatcher: Callback invoked for each translated MIDI event
            port_name: Name of MIDI input port to open (None = auto-select first)
        """
        if not callable(dispatcher):
            raise TypeError('dispatcher must be callable')

        self._dispatcher = dispatcher
        self._port_name = port_name
        self._port: mido.ports.BaseInput | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> bool:
        """
        Open the MIDI port and start listening in a background thread.

        Returns:
            True if port opened successfully, False otherwise
        """
        if self._thread is not None and self._thread.is_alive():
            LOGGER.warning('MIDI port translator already running')
            return True

        try:
            available_ports = list_midi_ports()
            if not available_ports:
                LOGGER.warning('No MIDI input ports available')
                return False

            port_name = self._port_name
            if port_name is None:
                port_name = available_ports[0]
                LOGGER.info('Auto-selected MIDI port: %s', port_name)
            elif port_name not in available_ports:
                LOGGER.error('MIDI port %s not found. Available: %s', port_name, available_ports)
                return False

            self._port = mido.open_input(port_name)
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            LOGGER.info('MIDI port translator started on %s', port_name)
            return True

        except (OSError, IOError) as exc:
            LOGGER.error('Failed to open MIDI port %s: %s', self._port_name, exc)
            return False

    def stop(self) -> None:
        """Stop listening and close the MIDI port."""
        self._stop_event.set()

        # Close port first to unblock the iterator in the thread
        if self._port is not None:
            try:
                self._port.close()
            except Exception as exc:
                LOGGER.warning('Error closing MIDI port: %s', exc)
            finally:
                self._port = None

        # Now join the thread (should exit quickly since port is closed)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                LOGGER.warning('MIDI listener thread did not stop cleanly')
            self._thread = None

        LOGGER.info('MIDI port translator stopped')

    def _listen_loop(self) -> None:
        """Background thread loop that reads MIDI messages and dispatches events."""
        if self._port is None:
            return

        try:
            for msg in self._port:
                if self._stop_event.is_set():
                    break

                event = self._translate_message(msg)
                if event is not None:
                    self._dispatch(event)

        except (OSError, IOError):
            # Port was closed, exit cleanly
            if not self._stop_event.is_set():
                LOGGER.warning('MIDI port closed unexpectedly')
        except Exception as exc:
            LOGGER.exception('Error in MIDI listen loop: %s', exc)

    def _translate_message(self, msg: mido.Message) -> MidiEvent | None:
        """
        Convert a mido Message into a MidiEvent.

        Args:
            msg: mido Message object

        Returns:
            MidiEvent or None if message type is not supported
        """
        if msg.type == 'note_on':
            if msg.velocity == 0:
                return MidiEvent(
                    event_type='note_off',
                    note=msg.note,
                    velocity=0,
                    channel=msg.channel,
                    timestamp=time.time(),
                )
            return MidiEvent(
                event_type='note_on',
                note=msg.note,
                velocity=msg.velocity,
                channel=msg.channel,
                timestamp=time.time(),
            )
        elif msg.type == 'note_off':
            return MidiEvent(
                event_type='note_off',
                note=msg.note,
                velocity=0,
                channel=msg.channel,
                timestamp=time.time(),
            )

        return None

    def _dispatch(self, event: MidiEvent) -> None:
        """Invoke the dispatcher callback with error handling."""
        try:
            self._dispatcher(event)
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            LOGGER.exception('Exception while dispatching MIDI event %s: %s', event, exc)


__all__ = ['MidiPortTranslator', 'list_midi_ports']
