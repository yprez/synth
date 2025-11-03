"""Sample-accurate event scheduler for QWERTY Synth.

This module provides a sample-accurate event scheduling system that eliminates
timing drift by scheduling events based on audio sample counts rather than
wall clock time. This ensures perfect synchronization between MIDI playback,
step sequencer, arpeggiator, and audio synthesis.
"""

import heapq
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from qwerty_synth import config


@dataclass(order=True)
class ScheduledEvent:
    """Event scheduled at a specific sample time."""

    sample_time: int
    event_type: str = field(compare=False)  # 'note_on', 'note_off', 'callback'
    midi_note: Optional[int] = field(default=None, compare=False)
    velocity: float = field(default=1.0, compare=False)
    duration: float = field(default=0.0, compare=False)  # Duration in seconds (for note_on events)
    callback: Optional[Callable] = field(default=None, compare=False)
    callback_args: tuple = field(default=(), compare=False)
    source: str = field(default='unknown', compare=False)  # Source of event: 'sequencer', 'midi', 'arpeggiator', etc.


class EventScheduler:
    """Sample-accurate event scheduler for timing-critical operations."""

    def __init__(self):
        """Initialize the event scheduler."""
        self.current_sample = 0
        self.event_queue = []  # Priority queue (heap) of ScheduledEvent objects
        self.lock = threading.Lock()
        self.sample_rate = config.sample_rate

    def reset(self):
        """Reset the scheduler (clear all events and sample counter)."""
        with self.lock:
            self.current_sample = 0
            self.event_queue.clear()

    def get_current_sample(self) -> int:
        """Get the current sample position."""
        with self.lock:
            return self.current_sample

    def get_current_time_seconds(self) -> float:
        """Get the current time in seconds."""
        with self.lock:
            return self.current_sample / self.sample_rate

    def samples_to_seconds(self, samples: int) -> float:
        """Convert sample count to seconds."""
        return samples / self.sample_rate

    def seconds_to_samples(self, seconds: float) -> int:
        """Convert seconds to sample count."""
        return int(seconds * self.sample_rate)

    def schedule_note_on(self, midi_note: int, velocity: float, delay_seconds: float,
                         duration_seconds: float = 0, source: str = 'unknown') -> int:
        """Schedule a note_on event.

        Args:
            midi_note: MIDI note number
            velocity: Note velocity (0.0-1.0)
            delay_seconds: Delay before note starts (in seconds from now)
            duration_seconds: Note duration in seconds (0 = infinite)
            source: Source of the event (e.g., 'sequencer', 'midi', 'arpeggiator')

        Returns:
            Sample time when the note will be triggered
        """
        with self.lock:
            sample_time = self.current_sample + self.seconds_to_samples(delay_seconds)
            event = ScheduledEvent(
                sample_time=sample_time,
                event_type='note_on',
                midi_note=midi_note,
                velocity=velocity,
                duration=duration_seconds,
                source=source
            )
            heapq.heappush(self.event_queue, event)
            return sample_time

    def schedule_note_off(self, midi_note: int, delay_seconds: float, source: str = 'unknown') -> int:
        """Schedule a note_off event.

        Args:
            midi_note: MIDI note number
            delay_seconds: Delay before note stops (in seconds from now)
            source: Source of the event (e.g., 'sequencer', 'midi', 'arpeggiator')

        Returns:
            Sample time when the note will be released
        """
        with self.lock:
            sample_time = self.current_sample + self.seconds_to_samples(delay_seconds)
            event = ScheduledEvent(
                sample_time=sample_time,
                event_type='note_off',
                midi_note=midi_note,
                source=source
            )
            heapq.heappush(self.event_queue, event)
            return sample_time

    def schedule_callback(self, callback: Callable, delay_seconds: float, *args, source: str = 'unknown') -> int:
        """Schedule a generic callback function.

        Args:
            callback: Function to call
            delay_seconds: Delay before calling (in seconds from now)
            *args: Arguments to pass to the callback
            source: Source of the event (e.g., 'sequencer', 'midi', 'arpeggiator')

        Returns:
            Sample time when the callback will be invoked
        """
        with self.lock:
            sample_time = self.current_sample + self.seconds_to_samples(delay_seconds)
            event = ScheduledEvent(
                sample_time=sample_time,
                event_type='callback',
                callback=callback,
                callback_args=args,
                source=source
            )
            heapq.heappush(self.event_queue, event)
            return sample_time

    def clear_all_events(self):
        """Clear all scheduled events."""
        with self.lock:
            self.event_queue.clear()

    def clear_events_by_source(self, source: str):
        """Clear all events from a specific source.

        Args:
            source: Source identifier to filter by (e.g., 'sequencer')
        """
        with self.lock:
            # Filter out events from the specified source
            self.event_queue = [event for event in self.event_queue if event.source != source]
            # Re-heapify the queue
            heapq.heapify(self.event_queue)

    def process_events(self, num_frames: int) -> list:
        """Process all events in the current audio buffer.

        Args:
            num_frames: Number of audio frames in the current buffer

        Returns:
            List of (frame_offset, event) tuples for events in this buffer
        """
        events_to_process = []

        with self.lock:
            buffer_start = self.current_sample
            buffer_end = self.current_sample + num_frames

            # Process all events that should happen in this buffer
            while self.event_queue and self.event_queue[0].sample_time < buffer_end:
                event = heapq.heappop(self.event_queue)

                # Calculate frame offset within this buffer
                frame_offset = max(0, event.sample_time - buffer_start)

                events_to_process.append((frame_offset, event))

                # If it's a note_on with duration, schedule the note_off
                if event.event_type == 'note_on' and event.duration > 0:
                    note_off_time = event.sample_time + self.seconds_to_samples(event.duration)
                    note_off_event = ScheduledEvent(
                        sample_time=note_off_time,
                        event_type='note_off',
                        midi_note=event.midi_note
                    )
                    heapq.heappush(self.event_queue, note_off_event)

            # Advance the sample counter
            self.current_sample += num_frames

        return events_to_process

    def get_pending_event_count(self) -> int:
        """Get the number of pending events."""
        with self.lock:
            return len(self.event_queue)


# Global scheduler instance
global_scheduler = EventScheduler()
