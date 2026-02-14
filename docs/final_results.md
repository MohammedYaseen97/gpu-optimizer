# Final results

## Protocol (how numbers were produced)

- **Checkpoint selection**: during training, we periodically evaluate on a fixed **VAL** seed set and keep the checkpoint with best `eval_mean_return`. Final numbers below are computed after reloading that checkpoint.
- **Deterministic evaluation**: greedy / argmax actions (stable curves).
- **VAL vs REPORT**:
  - **VAL** = fixed seed set used during training to select checkpoints.
  - **REPORT** = disjoint fixed seed set used only for the final number (reduces selection bias).
- **Common setup**: `seed=1`, `jobs_per_episode=400`, `max_queue_size=50`, `eval_num_blocks=3`, `eval_episodes=20` (60 eval episodes total).
- **Lookahead modalities**: `off`, `per_gpu`, `sorted`, `cdf` (see table).

## Baselines (heuristics)

```text
FIFO          | mean=0.369  min=0.340  max=0.388 | invalid=0.000  noop=0.526
SJF           | mean=0.529  min=0.511  max=0.554 | invalid=0.000  noop=0.497
RAND_FEAS_NO0 | mean=0.525  min=0.450  max=0.626 | invalid=0.000  noop=0.513
```

## PPO results (configs and outcomes)

Notes:
- **Invalid actions** are treated as **no-op** in the environment and we also use action masking, so `invalid_frac` is effectively 0.000; we focus on reward and `noop_frac`.
- **REPORT seed base** used here is `12000001` (fixed across configs).

We evaluate **all combinations**:
- **Policy**: `mlp` vs `attn`
- **Lookahead**: `off` vs `per_gpu` vs `sorted` vs `cdf`

| Policy | Lookahead mode | VAL avg reward | REPORT avg reward | Best checkpoint |
|---|---|---:|---:|---|
| MLP | `off` | 0.520 | 0.525 | `runs/checkpoints/ppo_baseline_stable_archmlp_lookahead_off_400jobs_w50_seed1_20260214_144604_best.pt` |
| MLP | `per_gpu` | 0.237 | 0.286 | `runs/checkpoints/ppo_baseline_stable_archmlp_lookahead_on_400jobs_w50_seed1_20260214_155242_best.pt` |
| MLP | `sorted` | **TBD** | **TBD** | **TBD** |
| MLP | `cdf` | **TBD** | **TBD** | **TBD** |
| Attention | `off` | 0.614 | 0.577 | `runs/checkpoints/ppo_baseline_stable_archattn_lookahead_off_400jobs_w50_seed1_20260214_180047_best.pt` |
| Attention | `per_gpu` | **0.608** | **0.589** | `runs/checkpoints/ppo_baseline_stable_archattn_lookahead_on_400jobs_w50_seed1_20260214_222855_best.pt` |
| Attention | `sorted` | **TBD** | **TBD** | **TBD** |
| Attention | `cdf` | **TBD** | **TBD** | **TBD** |

## Conclusion

- **Best completed config so far**: **Attention + `per_gpu` lookahead** achieves **REPORT = 0.589**, beating the best baseline (SJF mean 0.529).
- **Value of attention** (vs MLP) is large in this setup.
- **Value of per-GPU lookahead** for attention is **modest** (REPORT: 0.589 vs 0.577).
