"""
Policy network skeleton (PyTorch).

This will map flattened state observations to action probabilities
over the discrete action space of `SchedulerEnv`.
"""

import torch
import torch.nn as nn
import math


class PolicyNetwork(nn.Module):
    """
    Cross-attention policy (jobs attend to GPUs).

    Observation layout (from `SchedulerEnv`):
    - jobs:  max_queue_size * 4  (duration, priority, req_gpus, waiting_time)
    - gpus:  num_gpus * 1        (remaining busy time per GPU)
    - global: 2                  (idle_ratio, busy_ratio)

    Output:
    - logits of shape (B, 1 + max_queue_size)
      action 0 = no-op, action i>0 = pick job (i-1) from the window
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # action_dim = 1 (no-op) + max_queue_size
        self.max_queue_size = int(action_dim - 1)
        if self.max_queue_size <= 0:
            raise ValueError(f"action_dim must be >= 2, got {action_dim}")

        self.job_feat_dim = 4
        self.cluster_ratio_dim = 2  # (idle_ratio, busy_ratio)
        self.job_flat_dim = self.max_queue_size * self.job_feat_dim
        if state_dim < self.job_flat_dim + self.cluster_ratio_dim:
            raise ValueError(
                f"state_dim too small for queue window: state_dim={state_dim}, "
                f"needs >= {self.job_flat_dim + self.cluster_ratio_dim}"
            )
        self.num_gpus = int(state_dim - self.job_flat_dim - self.cluster_ratio_dim)
        if self.num_gpus <= 0:
            raise ValueError(
                f"Could not infer num_gpus from state_dim={state_dim} "
                f"(job_flat_dim={self.job_flat_dim}, cluster_ratio_dim=2)"
            )

        d = int(hidden_dim)  # token dimension
        self.d_model = d

        # Token embeddings (small MLPs)
        self.job_embed = nn.Sequential(
            nn.Linear(self.job_feat_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.gpu_embed = nn.Sequential(
            nn.Linear(1, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.global_embed = nn.Sequential(
            nn.Linear(self.cluster_ratio_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Single-head cross-attention (explicit Q/K/V)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.attn_ln = nn.LayerNorm(d)
        self.fuse_ln = nn.LayerNorm(d)

        # Inject global context into each job token before scoring.
        self.global_to_job = nn.Linear(d, d)
        self.job_head = nn.Linear(d, 1)  # per-job logit
        # No-op logit is produced through the SAME scoring head as jobs.
        # This avoids the "separate noop MLP dominates everything" failure mode.
        # We still allow a small bias toward scheduling by initializing this negative.
        self.noop_bias = nn.Parameter(torch.tensor(-1.0))

    def _split_obs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split flattened obs into (jobs, gpus, globals)."""
        B = x.shape[0]
        jobs = x[:, : self.job_flat_dim].view(B, self.max_queue_size, self.job_feat_dim)  # (B, N, 4)
        gpu_start = self.job_flat_dim
        gpus = x[:, gpu_start : gpu_start + self.num_gpus].view(B, self.num_gpus, 1)  # (B, G, 1)
        globals_ = x[:, -self.cluster_ratio_dim :]  # (B, 2)
        return jobs, gpus, globals_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, state_dim)
        jobs, gpus, globals_ = self._split_obs(x)

        # Embed tokens
        job_emb = self.job_embed(jobs)      # (B, N, d)
        gpu_emb = self.gpu_embed(gpus)      # (B, G, d)
        global_emb = self.global_embed(globals_)  # (B, d)

        # Cross-attention: jobs attend to GPUs
        Q = self.Wq(job_emb)  # (B, N, d)
        K = self.Wk(gpu_emb)  # (B, G, d)
        V = self.Wv(gpu_emb)  # (B, G, d)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        ctx = torch.matmul(attn_weights, V)  # (B, N, d)
        ctx = self.Wo(ctx)

        # Residual + layernorm
        job_ctx = self.attn_ln(job_emb + ctx)  # (B, N, d)

        # Fuse global context into job tokens
        job_ctx = self.fuse_ln(job_ctx + self.global_to_job(global_emb).unsqueeze(1))

        # Job logits: (B, N)
        job_logits = self.job_head(job_ctx).squeeze(-1)  # (B, N)

        # No-op logit: build a single "noop token" and score it with `job_head`.
        # The noop token can attend to GPUs (like jobs do) and gets global context.
        job_pool = job_ctx.mean(dim=1)  # (B, d)
        gpu_pool = gpu_emb.mean(dim=1)  # (B, d)
        noop_base = (job_pool + gpu_pool + global_emb) / 3.0  # (B, d)
        noop_base = noop_base.unsqueeze(1)  # (B, 1, d)

        Q0 = self.Wq(noop_base)  # (B, 1, d)
        # Reuse K,V computed for GPUs
        attn_scores0 = torch.matmul(Q0, K.transpose(1, 2)) / math.sqrt(self.d_model)  # (B, 1, G)
        attn_weights0 = torch.softmax(attn_scores0, dim=-1)
        ctx0 = torch.matmul(attn_weights0, V)  # (B, 1, d)
        ctx0 = self.Wo(ctx0)

        noop_ctx = self.attn_ln(noop_base + ctx0)  # (B, 1, d)
        noop_ctx = self.fuse_ln(noop_ctx + self.global_to_job(global_emb).unsqueeze(1))  # (B, 1, d)
        noop_logit = self.job_head(noop_ctx).squeeze(-1) + self.noop_bias  # (B, 1)

        logits = torch.cat([noop_logit, job_logits], dim=1)  # (B, 1+N)
        return logits


