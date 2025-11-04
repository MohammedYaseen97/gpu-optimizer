"""
GPU class - Represents a single GPU in the cluster.

Each GPU can be assigned one job at a time. The GPU tracks its current state
and the job it's running.
"""

from typing import Optional
from .job import Job
import time

class GPU:
    """
    Represents a single GPU in the cluster.
    """
    
    def __init__(self, gpu_id: str):
        """
        Initialize a GPU.
        """
        self.gpu_id = gpu_id
        self.current_job = None
        self.status = "idle"
    
    def assign_job(self, job: Job) -> bool:
        """
        Assign a job to this GPU.
        """
        if self.status == "idle":
            self.current_job = job
            self.status = "busy"
            self.current_job.start_job(time.time())
            return True
    
    def release_job(self) -> Optional[Job]:
        """
        Release the current job from this GPU.
        """
        if self.current_job:
            released_job = self.current_job
        
        self.current_job = None
        self.status = "idle"
        return released_job
    
    def is_idle(self) -> bool:
        """
        Check if GPU is idle.
        """
        return self.status == "idle"
    
    def get_state(self) -> dict:
        """
        Get current state of the GPU.
        """
        return {
            "gpu_id": self.gpu_id,
            "status": self.status,
            "current_job": self.current_job.get_state()["job_id"] if self.current_job else None
        }
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        """
        return f"GPU(id={self.gpu_id}, status={self.status}, job={self.current_job.get_state()['job_id'] if self.current_job else None})"

