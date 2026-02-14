"""
Sanity-check the SchedulerEnv lookahead representations.

We verify:
- Observation shape is unchanged across modes.
- The lookahead slice equals the expected transform of the per-GPU vector:
  - off    -> zeros
  - per_gpu-> raw remaining busy times
  - sorted -> sort(per_gpu)
  - cdf    -> cdf over uniform thresholds tau[j]=(j+1)/G
"""

from __future__ import annotations

import numpy as np

from src.environment.scheduler_env import SchedulerEnv


def _lookahead_slice(env: SchedulerEnv, obs: np.ndarray) -> np.ndarray:
    rem_start = int(env.max_queue_size) * 4
    rem_end = rem_start + int(env.num_gpus)
    return obs[rem_start:rem_end]


def _first_feasible_non_noop(action_mask: np.ndarray) -> int:
    # action_mask shape = (max_queue_size+1,), action 0 is no-op
    idxs = np.flatnonzero(action_mask.astype(bool))
    for a in idxs.tolist():
        if int(a) != 0:
            return int(a)
    return 0


def main() -> None:
    seed = 123
    num_gpus = 4
    max_queue_size = 10

    # Keep it tiny + deterministic (lots of initial jobs so we can schedule immediately).
    env_kwargs = dict(
        num_gpus=num_gpus,
        max_queue_size=max_queue_size,
        jobs_per_episode=40,
        arrival_mode="all_at_zero",
        max_duration=1000.0,
        debug=False,
    )

    env_off = SchedulerEnv(lookahead_mode="off", **env_kwargs)
    env_per = SchedulerEnv(lookahead_mode="per_gpu", **env_kwargs)
    env_sorted = SchedulerEnv(lookahead_mode="sorted", **env_kwargs)
    env_cdf = SchedulerEnv(lookahead_mode="cdf", **env_kwargs)

    obs_off, info_off = env_off.reset(seed=seed)
    obs_per, info_per = env_per.reset(seed=seed)
    obs_sorted, info_sorted = env_sorted.reset(seed=seed)
    obs_cdf, info_cdf = env_cdf.reset(seed=seed)

    # 1) Shape unchanged
    assert obs_off.shape == obs_per.shape == obs_sorted.shape == obs_cdf.shape, (
        obs_off.shape,
        obs_per.shape,
        obs_sorted.shape,
        obs_cdf.shape,
    )

    # 2) Per-step checks under identical actions
    steps = 8
    for t in range(steps):
        raw = _lookahead_slice(env_per, obs_per)
        got_off = _lookahead_slice(env_off, obs_off)
        got_sorted = _lookahead_slice(env_sorted, obs_sorted)
        got_cdf = _lookahead_slice(env_cdf, obs_cdf)

        # off -> zeros
        assert np.allclose(got_off, 0.0, atol=0.0, rtol=0.0), (t, got_off)

        # sorted -> sort(raw)
        exp_sorted = np.sort(raw.astype(np.float32, copy=False))
        assert np.allclose(got_sorted, exp_sorted, atol=1e-6, rtol=0.0), (
            t,
            got_sorted,
            exp_sorted,
        )

        # cdf -> mean(raw <= tau[j]) for tau[j]=(j+1)/G
        G = int(env_per.num_gpus)
        taus = (np.arange(1, G + 1, dtype=np.float32) / float(G)).astype(np.float32)
        exp_cdf = np.array([(raw <= tau).mean() for tau in taus], dtype=np.float32)
        assert np.allclose(got_cdf, exp_cdf, atol=1e-6, rtol=0.0), (
            t,
            got_cdf,
            exp_cdf,
        )

        # Pick actions based on the per_gpu env mask and replay everywhere.
        a = _first_feasible_non_noop(info_per["action_mask"])

        obs_off, _, term_off, trunc_off, info_off = env_off.step(a)
        obs_per, _, term_per, trunc_per, info_per = env_per.step(a)
        obs_sorted, _, term_sorted, trunc_sorted, info_sorted = env_sorted.step(a)
        obs_cdf, _, term_cdf, trunc_cdf, info_cdf = env_cdf.step(a)

        assert (term_off, trunc_off) == (term_per, trunc_per) == (term_sorted, trunc_sorted) == (
            term_cdf,
            trunc_cdf,
        ), (t, term_off, trunc_off, term_per, trunc_per, term_sorted, trunc_sorted, term_cdf, trunc_cdf)

        if term_per or trunc_per:
            break

    print(
        "OK: lookahead modes behave as expected "
        f"(tested steps={t+1}, obs_shape={obs_per.shape}, num_gpus={num_gpus}, max_queue_size={max_queue_size})."
    )


if __name__ == "__main__":
    main()

