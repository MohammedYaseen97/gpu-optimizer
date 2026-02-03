"""
Job class - Represents a GPU job in the scheduling system.

A job represents a computational task that requires one or more GPUs.
Each job has attributes like resource requirements, duration, priority, etc.
"""

from enum import Enum
from typing import Dict, Optional


class JobStatus(Enum):
    """Status of a job in the system."""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """
    Represents a GPU job that needs to be scheduled.
    """
    
    def __init__(
        self,
        job_id: str,
        submission_time: float,
        required_gpus: int,
        estimated_duration: float,
        priority: int = 0,
        user_id: str = "default",
        **kwargs
    ):
        """
        Initialize a new job.
        """
        self.job_id = job_id
        self.submission_time = submission_time
        self.required_gpus = required_gpus
        self.estimated_duration = estimated_duration
        self.actual_duration = None
        self.priority = priority
        self.user_id = user_id
        self.status = JobStatus.SUBMITTED
        self.start_time = None
        # Simulator-time timestamp when this job is expected to complete.
        # Populated when the scheduler starts the job.
        self.expected_end_time: Optional[float] = None
    
    def __repr__(self) -> str:
        """
        Return string representation.
        """
        return f"Job(id={self.job_id}, gpus={self.required_gpus}, duration={self.estimated_duration}, priority={self.priority}, user={self.user_id}, status={self.status})"
    
    def get_state(self) -> Dict:
        """
        Get current state of the job as a dictionary.
        """
        return {
            "job_id": self.job_id,
            "required_gpus": self.required_gpus,
            "estimated_duration": self.estimated_duration,
            "priority": self.priority,
            "user_id": self.user_id,
            "status": self.status
        }
        
    
    def start_job(self, start_time: float) -> None:
        """
        Mark job as running.
        """
        self.status = JobStatus.RUNNING
        self.start_time = start_time
    
    def complete_job(self, completion_time: float) -> None:
        """
        Mark job as completed.
        """
        self.status = JobStatus.COMPLETED
        self.actual_duration = completion_time - self.start_time

