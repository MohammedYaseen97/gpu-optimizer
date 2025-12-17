"""
Value network skeleton (PyTorch).

This will map flattened state observations to a scalar value estimate V(s).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    Simple MLP value function:
    - Input: state vector (float32, shape = [state_dim])
    - Output: scalar value V(s)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Mirror PolicyNetwork structure but output a single value.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input state batch, shape (batch_size, state_dim)

        Returns
        -------
        values : torch.Tensor
            Value estimates, shape (batch_size, 1)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.out(x)


