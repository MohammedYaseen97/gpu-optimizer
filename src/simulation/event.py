"""
Event class - Represents events in discrete event simulation.

Events are used to drive the simulation forward. Each event has a timestamp
and type, and can carry data about the event.
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation."""
    JOB_ARRIVAL = "job_arrival"
    JOB_COMPLETION = "job_completion"
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"


@dataclass
class Event:
    """
    Represents a single event in the discrete event simulation.
    
    Attributes you should understand:
    ----------
    event_type : EventType
        Type of this event (JOB_ARRIVAL, JOB_COMPLETION, etc.)
    timestamp : float
        Simulation time when this event occurs
    data : Dict[str, Any]
        Additional data associated with the event
        (e.g., {"job": job_object} for JOB_ARRIVAL events)
    
    Methods you should implement:
    ----------
    The dataclass decorator provides __init__, __repr__, __eq__ automatically.
    But you need to implement comparison methods for priority queue.
    
    __lt__(other) -> bool
        Compare events by timestamp (for priority queue sorting)
    """
    
    event_type: EventType
    timestamp: float
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize data dict if None."""
        if self.data is None:
            self.data = {}
    
    def __lt__(self, other: 'Event') -> bool:
        """
        Less-than comparison for priority queue.
        """
        if self.timestamp < other.timestamp:
            return True
        if self.timestamp > other.timestamp:
            return False

        # Tie-breaker on event type to have a deterministic ordering when
        # timestamps are equal.
        priority_order = [
            EventType.SIMULATION_START,
            EventType.JOB_ARRIVAL,
            EventType.JOB_COMPLETION,
            EventType.SIMULATION_END,
        ]
        return priority_order.index(self.event_type) < priority_order.index(other.event_type)
    
    def __repr__(self) -> str:
        """
        String representation.
        """
        job_id = None
        job = self.data.get("job")
        if job is not None and hasattr(job, "get_state"):
            job_id = job.get_state().get("job_id")
        return f"Event(type={self.event_type}, time={self.timestamp}, job={job_id})"

