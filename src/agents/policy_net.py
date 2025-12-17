"""
Policy network skeleton (PyTorch).

This will map flattened state observations to action probabilities
over the discrete action space of `SchedulerEnv`.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Simple MLP policy:
    - Input: state vector (float32, shape = [state_dim])
    - Output: logits over actions (shape = [action_dim])

    You will:
    - Instantiate this with (state_dim, action_dim)
    - Use it inside your PPO agent in Week 3
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Simple 2-layer MLP; you can tweak hidden_dim later if needed.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input state batch, shape (batch_size, state_dim)

        Returns
        -------
        logits : torch.Tensor
            Unnormalized action scores, shape (batch_size, action_dim).
            (You can apply softmax in the agent when constructing a distribution.)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits


