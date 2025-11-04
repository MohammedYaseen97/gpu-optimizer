"""
Cluster class - Represents a collection of GPUs.

The cluster manages multiple GPUs and provides methods to query their state
and assign jobs to them.
"""

from typing import List, Optional
from .gpu import GPU
from .job import Job


class Cluster:
    """
    Represents a cluster of GPUs.
    
    Attributes you should implement:
    ----------
    gpus : List[GPU]
        List of GPU objects in the cluster
    total_gpus : int
        Total number of GPUs in the cluster
    
    Methods you should implement:
    ----------
    __init__(num_gpus: int)
        Create a cluster with num_gpus GPUs
    
    get_idle_gpus() -> List[GPU]
        Return list of all idle GPU objects
    
    get_idle_count() -> int
        Return count of idle GPUs
    
    get_busy_count() -> int
        Return count of busy GPUs
    
    assign_job(job: Job) -> bool
        Try to assign a single job to GPUs. Return True if successful.
        Note: If a job requires multiple GPUs, you need to assign it to multiple GPUs.
        Use job.required_gpus to know how many GPUs are needed.
    
    release_job(job: Job) -> None
        Release all GPUs assigned to a given job
    
    get_state() -> dict
        Return dictionary with cluster state
    """
    
    def __init__(self, num_gpus: int):
        """
        Initialize a cluster with num_gpus GPUs.
        """
        self.gpus = [GPU(gpu_id="gpu_{i}") for i in range(num_gpus)]
        self.total_gpus = num_gpus
    
    def get_idle_gpus(self) -> List[GPU]:
        """
        Get all idle GPUs.
        """
        return [gpu for gpu in self.gpus if gpu.is_idle()]
    
    def get_idle_count(self) -> int:
        """
        Get count of idle GPUs.
        """
        return len(self.get_idle_gpus())
    
    def get_busy_count(self) -> int:
        """
        Get count of busy GPUs.
        """
        return self.total_gpus - self.get_idle_count()
    
    def assign_job(self, job: Job) -> bool:
        """
        Assign a single job to GPUs.
        """
        if job.required_gpus > self.get_idle_count():
            return False
        
        idle_gpus = self.get_idle_gpus()
        
        for i in range(job.required_gpus):
            idle_gpus[i].assign_job(job)
        
        return True
    
    def release_job(self, job: Job) -> None:
        """
        Release all GPUs assigned to a given job.
        """
        assigned_gpus = [gpu for gpu in self.gpus if gpu.get_state().current_job == job.job_id]
        
        assert len(assigned_gpus) == job.required_gpus
        
        for gpu in assigned_gpus:
            gpu.release_job()
    
    def get_state(self) -> dict:
        """
        Get current state of the cluster.
        
        TODO:
        - Return dict with: total_gpus, idle_count, busy_count, 
          list of GPU states
        """
        return {
            "total_gpus": self.total_gpus,
            "idle_count": self.get_idle_count(),
            "busy_count": self.get_busy_count(),
            "gpu_states": [gpu.get_state() for gpu in self.gpus]
        }
    
    def __repr__(self) -> str:
        """
        String representation.
        
        TODO:
        - Return something like "Cluster(gpus=8, idle=3, busy=5)"
        """
        return f"Cluster(gpus={self.total_gpus}, idle={self.get_idle_count()}, busy={self.get_busy_count()})"

