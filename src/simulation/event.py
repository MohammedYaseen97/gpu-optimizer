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
        
        TODO:
        - Compare by timestamp (earlier timestamp = smaller)
        - If timestamps are equal, you might want to break ties by event_type
          (e.g., JOB_ARRIVAL before JOB_COMPLETION)
        """
        if self.timestamp < other.timestamp:
            return True
        if self.timestamp > other.timestamp:
            return False
        
        priority_order = ["simulation_start", "job_arrival", "job_completion", "simulation_end"]
        return priority_order.index(self.event_type) < priority_order.index(other.event_type)
    
    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"Event(type={self.event_type.upper()}, time={self.timestamp}, job={self.data['job'].get_state()['job_id'] if self.data.get("job") else "None"})"

