"""
Discrete Event Simulator for GPU scheduling.

This simulator drives the environment forward using an event-driven approach.
Events are processed in chronological order, advancing simulation time.

KEY CONCEPT - Two Different Queues:

1. EVENT_QUEUE: Simulation mechanism (internal to simulator)
   - Contains EVENTS (things that happen at specific times)
   - Examples: "At time 10.0, job arrives", "At time 50.0, job completes"
   - Processed in chronological order to advance simulation time
   - This is how discrete event simulation works

2. JOB_QUEUE: The actual waiting list (what agent sees)
   - Contains JOBS that have arrived but haven't been scheduled yet
   - This is the "queue" from MDP spec - agent sees a window of this
   - When job arrives (JOB_ARRIVAL event fires) → job added here
   - When agent schedules job → job removed from here, assigned to GPUs
   - When job completes → job already removed (was removed when scheduled)

FLOW:
1. JOB_ARRIVAL event scheduled → goes in event_queue
2. Event fires → job added to job_queue (waiting list)
3. Agent sees window of job_queue (this is the state in MDP)
4. Agent schedules job → job removed from job_queue, assigned to GPUs
5. JOB_COMPLETION event scheduled (at current_time + duration) → goes in event_queue
6. Completion event fires → job removed from GPUs, metrics updated
"""

import heapq
from typing import List, Optional, Callable, Dict
from .event import Event, EventType
from ..environment.job import Job
from ..environment.cluster import Cluster


class Simulator:
    """
    Discrete event simulator for GPU job scheduling.
    
    The simulator manages:
    - Event queue: Events scheduled for future times (simulation mechanism)
    - Job queue: Jobs waiting to be scheduled (what agent observes)
    - Cluster: GPUs and running jobs
    - Metrics: Performance statistics
    """
    
    def __init__(
        self,
        cluster: Cluster,
        job_arrival_generator: Optional[Callable] = None,
        max_time: float = 10000.0
    ):
        """
        Initialize the simulator.
        
        Parameters:
        ----------
        cluster : Cluster
            The GPU cluster being simulated
        job_arrival_generator : Callable, optional
            Function that generates job arrival events
            Should return a list of (timestamp, Job) tuples or schedule events directly
        max_time : float
            Maximum simulation time
        """
        self.current_time = 0
        self.event_queue = []
        self.cluster = cluster
        self.job_queue = []
        self.metrics = {
            "jobs_completed": 0,
            "total_jobs": 0,
            "total_wait_time": 0.0,
            "gpu_busy_time": 0.0
        }
        self.max_time = max_time,
        self.job_arrival_generator = job_arrival_generator
 
    
    def schedule_event(self, event: Event) -> None:
        """
        Add an event to the event queue.
        """
        return heapq.heappush(self.event_queue, event)
    
    
    def get_next_event(self) -> Optional[Event]:
        """
        Get and remove the next event from the event queue (earliest timestamp).
        
        Returns:
        -------
        Event or None if queue is empty
        """
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)
    
    
    def handle_job_arrival(self, event: Event) -> None:
        """
        Handle a job arrival event.
        
        When this event fires:
        - Extract the job from event.data
        - Add job to job_queue (the waiting list - this is what agent sees!)
        - Update metrics: total_jobs += 1
        
        NOTE: We do NOT schedule the job here - that's the agent's decision!
        The job just goes into the waiting queue.
        """
        job = event.data["job"]
        self.job_queue.append(job)
        self.metrics["total_jobs"] += 1
    
    
    def schedule_job(self, job: Job) -> bool:
        """
        Schedule a job (called by agent/environment when agent takes action).
        
        This is called when the agent decides to schedule a job.
        It:
        - Removes job from job_queue (waiting list)
        - Assigns job to cluster (GPUs)
        - Schedules a JOB_COMPLETION event for when job will finish
        
        Parameters:
        ----------
        job : Job
            Job to schedule (must be in job_queue)
        
        Returns:
        -------
        bool
            True if successfully scheduled, False if not enough GPUs or job not in queue
        """
        if not job in self.job_queue:
            return False
        
        assigned = self.cluster.assign_job(job)
        if not assigned:
            return False
        
        self.job_queue.remove(job)
        job.start_job(self.current_time)
        completion_time = self.current_time + job.estimated_duration
        job_completion_event = Event(event_type=EventType.JOB_COMPLETION, timestamp=completion_time, data={"job": job})
        self.schedule_event(job_completion_event)
        return True
    
    
    def handle_job_completion(self, event: Event) -> None:
        """
        Handle a job completion event.
        
        When this event fires:
        - Extract the job from event.data
        - Release GPUs assigned to this job
        - Mark job as completed
        - Update metrics (jobs_completed, wait_time, etc.)
        
        NOTE: The job was already removed from job_queue when it was scheduled.
        This just releases GPUs and updates completion metrics.
        """
        job = event.data["job"]
        self.cluster.release_jobs()
        
        job.complete_job(self.current_time)
        
        self.metrics.jobs_completed += 1
        if job.start_time:
            self.metrics.total_wait_time += (job.start_time - job.submission_time)
        
        return
    
    
    def advance_time(self, new_time: float) -> None:
        """
        Advance simulation time.
        """
        assert new_time >= self.current_time, "New time must be greater than or equal to current time"
        self.current_time = new_time
        return
    
    
    def step(self) -> bool:
        """
        Process one event (one simulation step).
        
        This advances time and processes the next event.
        
        Returns:
        -------
        bool
            True if simulation should continue, False if done
        
        NOTE: This does NOT schedule jobs - that's done by agent via schedule_job()
        """
        event = self.get_next_event()
        if not event:
            return False
        
        self.advance_time(event.timestamp)
        match event.event_type:
            case EventType.JOB_ARRIVAL:
                self.handle_job_arrival(event)
            case EventType.JOB_COMPLETION:
                self.handle_job_completion(event)
            case EventType.SIMULATION_END:
                return False
        
        if self.current_time >= self.max_time:
            return False
        return True
    
    
    def run(self, until_time: Optional[float] = None) -> None:
        """
        Run simulation until specified time or until no more events.
        """
        while True:
            if not until_time or self.current_time < until_time:
                continue_steps = self.step()
            if not continue_steps:
                break
    
    
    def get_metrics(self) -> Dict:
        """
        Get current simulation metrics.
        
        Returns metrics needed for MDP reward calculation:
        - queue_length: Current number of jobs in job_queue
        - jobs_completed: Total jobs completed
        - total_jobs: Total jobs that arrived
        - total_wait_time: Sum of wait times
        - avg_wait_time: Average wait time
        - gpu_utilization: Fraction of time GPUs were busy
        """
        return {
            "queue_length": len(self.job_queue),
            "jobs_completed": self.metrics.jobs_completed,
            "total_jobs": self.metrics.total_jobs,
            "total_wait_time": self.metrics.total_wait_time,
            "avg_wait_time": self.metrics.total_wait_time if self.metrics.jobs_completed else 0.0,
            "gpu_utilization": (self.metrics.gpu_busy_time / (self.cluster.total_gpus * self.current_time)) if self.current_time else 0.0
        }
    
    def reset(self) -> None:
        """
        Reset simulator to initial state.
        
        TODO:
        - Reset current_time = 0.0
        - Clear event_queue (set to empty list)
        - Clear job_queue (set to empty list)
        - Reset all metrics to 0
        - Note: Cluster reset might be needed (check if Cluster has reset method)
        """
        self.current_time = 0.0
        self.event_queue = []
        self.job_queue = []
        self.metrics = {
            "jobs_completed": 0,
            "total_jobs": 0,
            "total_wait_time": 0.0,
            "gpu_busy_time": 0.0
        }
        self.cluster.reset()
    
    def get_job_queue(self) -> List[Job]:
        """
        Get the current job queue (waiting list).
        
        This is what the agent observes (window of this queue).
        
        Returns:
        -------
        List[Job]
            Current job queue (jobs waiting to be scheduled)
        """
        return self.job_queue
