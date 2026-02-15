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
| MLP | `sorted` | 0.560 | **0.613** | `runs/checkpoints/ppo_baseline_stable_archmlp_lookahead_sorted_400jobs_w50_seed1_20260215_002350_best.pt` |
| MLP | `cdf` | 0.407 | 0.342 | `runs/checkpoints/ppo_baseline_stable_archmlp_lookahead_cdf_400jobs_w50_seed1_20260215_082918_best.pt` |
| Attention | `off` | 0.614 | 0.577 | `runs/checkpoints/ppo_baseline_stable_archattn_lookahead_off_400jobs_w50_seed1_20260214_180047_best.pt` |
| Attention | `per_gpu` | **0.608** | **0.589** | `runs/checkpoints/ppo_baseline_stable_archattn_lookahead_on_400jobs_w50_seed1_20260214_222855_best.pt` |
| Attention | `sorted` | 0.578 | 0.555 | `runs/checkpoints/ppo_baseline_stable_archattn_lookahead_sorted_400jobs_w50_seed1_20260215_101004_best.pt` |
| Attention | `cdf` | 0.590 | **0.615** | `runs/checkpoints/ppo_baseline_stable_archattn_lookahead_cdf_400jobs_w50_seed1_20260215_120428_best.pt` |

## Post-hoc analysis (discussion)

### Why `sorted` helped MLP but not attention

- **MLP is not permutation-invariant** over the GPU feature vector. With `per_gpu`, the same physical state can appear under many GPU index permutations, which the MLP must waste capacity to relearn. `sorted` removes this nuisance variation and makes the representation permutation-invariant while still preserving detailed timing structure (the GPU-time *order statistics*).
- **The attention policy is already permutation-invariant** over GPUs in this implementation: GPU tokens share weights (`gpu_embed`) and there are no positional/ID embeddings, so reordering GPU entries does not provide the same structural benefit. In that case `sorted` mostly changes optimization details / training trajectory rather than adding useful information.

### Why `cdf` helped attention but hurt MLP

- `cdf` is a **lossy compression** of GPU remaining times: it keeps a monotone “future capacity curve” (fraction of GPUs free by time thresholds) but discards fine-grained gap structure that can matter for exact job-start timing.
- For a plain MLP, this compression can be too coarse (especially with small `num_gpus`, where the CDF moves in increments of \(1/G\)), so it may underfit the scheduling decisions.
- For the attention policy, the CDF can be a **cleaner global availability signal** to fuse with job tokens, reducing the need to reason over many near-tied GPU tokens. That can make PPO updates more stable and improve generalization, even though the signal is lossy.

## Conclusion

- **Best completed config so far**: **Attention + `cdf` lookahead** achieves **REPORT = 0.615**, beating the best baseline (SJF mean 0.529).
- **Value of attention** (vs MLP) is large in this setup.
- **Value of per-GPU lookahead** for attention is **modest** (REPORT: 0.589 vs 0.577).
