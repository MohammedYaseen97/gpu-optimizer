"""
Test script: Run environment with random policy.

This validates that your environment works end-to-end.
Run this after implementing Simulator and SchedulerEnv.
"""

import numpy as np
from src.environment.scheduler_env import SchedulerEnv


def test_random_policy(num_steps: int = 100):
    """
    Test environment with random actions.
    
    This verifies:
    - Environment can be created
    - reset() works
    - step() works with random actions
    - No crashes or errors
    - Basic metrics are collected
    
    Parameters:
    ----------
    num_steps : int
        Number of steps to run
    """
    print("Creating environment...")
    env = SchedulerEnv(num_gpus=8, max_queue_size=50)
    
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial action mask: {info.get('action_mask', 'Not provided')}")
    
    print(f"\nRunning {num_steps} steps with random policy...")
    total_reward = 0.0
    
    for step in range(num_steps):
        # Get valid actions from mask
        action_mask = info.get('action_mask', None)
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0  # No-op if no valid actions
        else:
            # If no mask, just pick random action
            action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: action={action}, reward={reward:.3f}, "
                  f"terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    print("Test completed successfully!")


if __name__ == "__main__":
    test_random_policy(num_steps=100)
