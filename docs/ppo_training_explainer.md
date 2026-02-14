# PPO training curves (scheduler) — an intuitive debugging playbook

This note is written to match the code in:
- `src/agents/ppo_agent.py` (current PPO implementation)
- `scripts/train_ppo.py` (experiment runner)

It explains how to interpret the *signals you actually see* while training PPO:

- eval curves (`eval_mean_return`, per-block returns, worst-block)
- rollout stats (`mean_ep_return`, `noop_frac`, `invalid_frac`)
- optimization stats (`pg_loss`, `v_loss`, `entropy`, `approx_kl_*`)

And it gives a practical “if you see X, suspect Y” checklist so you can diagnose issues the way RL practitioners do (rather than with supervised-learning instincts).

---

## 1) The one mental model to keep

In supervised learning, you minimize a fixed loss on a fixed dataset:

- data is i.i.d. and stationary (usually)
- the loss you optimize is the same thing you care about (often)

In PPO (RL), you *do not* have a fixed dataset. Your “dataset” is the rollouts your current policy generates, and that changes every time the policy changes.

So you should think of PPO training as two interleaved loops:

1. **Collect rollouts** using the current policy $\pi_{\theta_{\mathrm{old}}}$ (this creates training data).
2. **Update the policy** to $\pi_\theta$ on that data (but only “a bit”, using clipping / KL guardrails).

The important consequence:

- The “loss” you see is a *surrogate objective on the current rollout batch*.
- It is not guaranteed to decrease smoothly, and it is not guaranteed that “lower loss ⇒ better scheduler”.

---

## 2) What we plot vs what we optimize

### **Eval curve**

In our runs we log **deterministic evaluation** (argmax over masked logits) on **fixed seed blocks**.

That produces:
- `eval_mean_return`: mean over blocks
- `eval_min_block_return`: worst block (robustness)
- per-block returns: `eval_block0_return`, ...

Interpretation:
- **Mean** answers: “on average on these workloads, how good is the policy?”
- **Min-block** answers: “does it fall apart on some workload family?”

If mean rises but min-block falls, you’re improving on some seeds while regressing on others.

### **Rollout (training) return is not eval return**

Our logs contain **two different return-ish concepts**:

- `mean_ep_return` / `mean_ep_length`: computed from episodes that happen to end *inside the rollout buffer* while collecting on-policy data.
  - For long episodes, many rollouts contain **0–3 complete episodes**, so `mean_ep_return` is **high variance** and not a stable learning curve.
- `eval_*`: computed by running full episodes on **fixed seeds**. This is the curve to trust.

Rule of thumb:
- Treat `mean_ep_return` as a “is something catastrophically broken?” check.
- Treat `eval_mean_return` and `eval_min_block_return` as the real training curve.

### **What PPO optimizes (per update)**

PPO doesn’t directly optimize your eval metric; it optimizes a **clipped policy-gradient surrogate**, plus a value loss and entropy bonus:

$$
\mathcal{L}(\theta)
\approx
\mathbb{E}\big[\min(r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta),1\pm\epsilon)\,\hat{A}_t)\big]
 \ -\ c_{\mathrm{ent}}\,H(\pi_\theta(\cdot\mid s_t))
\ +\ c_v\,\text{ValueLoss}
$$

Where:
- $r_t(\theta) = \exp(\log\pi_\theta(a_t\mid s_t) - \log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t))$
- $\hat{A}_t$ are your advantages (GAE)
- $\epsilon$ is `clip_coef`

This is why “loss down” doesn’t mean “eval up”.

---

## 3) What each logged scalar means (in our code)

Below I’ll define each signal and then immediately translate it into **diagnostic meaning**.

---

## 3.5) Why PPO has so much clipping / clamping (and why we added extra clamps)

PPO is basically “policy gradient + **trust region-ish** constraints”.

The core idea:
- Policy gradient *wants* to take a big step if it thinks it found improvement.
- In RL, big steps often destroy behavior because your rollout distribution shifts.
- So PPO adds **clamps** that keep updates “local”.

There are multiple “clamps” in our stack; each protects a different failure mode.

### (A) **Policy ratio clip** (`clip_coef`) — the main PPO clamp

Define the probability ratio on rollout actions:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
          =\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t))
$$

The unclipped objective would push $r_t\hat{A}_t$ as much as possible. PPO instead uses:

$$
\min\Big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\Big)
$$

Intuition:
- If the new policy increases the probability of “good” actions too much (ratio $> 1+\epsilon$),
  PPO stops giving you additional benefit (the gradient gets cut off).
- If the new policy decreases the probability of “good” actions too much (ratio $< 1-\epsilon$),
  PPO also stops you from pushing further.

What it prevents:
- **Overshoot**: “one update turns a good policy into garbage”.

What it does *not* guarantee:
- “many small updates can’t drift” — they can (slow drift).

### (B) **Value function clip** (value prediction clamp)

We also clamp the critic update:

- unclipped: $(V_\theta(s_t) - R_t)^2$
- clipped: we limit how far $V_\theta$ can move from the old prediction $V_{\mathrm{old}}$ in one update

Intuition:
- If the critic moves too fast, your advantages $\hat{A}_t$ change scale/sign unpredictably.
- That feeds back into the policy update and destabilizes training.

This clamp is “PPO’s trust region idea applied to the critic”.

### (C) **Gradient norm clip** (`max_grad_norm`) — numerical/optimization safety

After backprop we clip:

$$
\|\nabla_\theta \mathcal{L}\|_2 \le \mathrm{max\_grad\_norm}
$$

Intuition:
- protects against occasional huge gradients from rare trajectories / bad minibatches
- keeps Adam from taking a massive parameter step due to one spike

This is not “PPO theory”; it’s practical deep learning hygiene.

### (D) **Entropy “jump” clamp** (`entropy_jump_clip`) — keep exploration from whiplashing

Standard PPO uses an entropy bonus:

$$
 -c_{\mathrm{ent}}\,H(\pi_\theta(\cdot\mid s_t))
$$

But entropy can change sharply between old/new policies on a minibatch.
We clamp the *change* per minibatch:

$$
H_{\mathrm{new}} \leftarrow H_{\mathrm{old}} + \text{clip}(H_{\mathrm{new}}-H_{\mathrm{old}},\ -\delta,\ +\delta)
$$

Intuition:
- In scheduling, sudden entropy drops often correlate with “commit to no-op / collapse”.
- Sudden entropy rises can also destabilize the value function because behavior changes too quickly.
- This clamp makes exploration level change **smoothly**, update-to-update.

This is an *extra* stabilizer (not required by PPO), similar in spirit to other “trust region” constraints.

### (E) **Target-KL early stop** (`target_kl`) — an explicit trust-region brake

We also monitor an approximate KL on rollout data:

$$
\mathrm{approx\_kl} \approx \mathbb{E}\big[(r_t - 1) - \log r_t\big]
$$

If it exceeds a threshold, we stop doing more epochs on the same batch.

Intuition:
- PPO does multiple epochs over the same data; KL tends to grow with each epoch.
- If KL gets big, you’re no longer doing “local improvement” on the data you collected.
- Early stop prevents the late epochs from being destructive.

Note:
- Some PPO implementations “penalize KL” instead of early stopping.
- In practice, early stopping is a simple and effective guardrail.

### (F) **Action-mask logit clamp** (invalid actions → very negative logit)

We do:
- valid actions: keep logits
- invalid actions: set logit to approximately $-10^9$

Intuition:
- This is a *hard constraint*.
- It removes “wasted gradient” on impossible actions and drives `invalid_frac` to ~0.

It’s also a clamp: it forces probability mass away from invalid actions.

---

### **`pg_loss`**

This is the *policy* part of PPO (ratio-clipped objective).

In our implementation, we minimize:
- `pg_loss = mean(max(pg_loss1, pg_loss2))`

Because the objective is written as a *loss* (negative of what we want), the sign can feel backwards:
- “better” policy improvement often makes `pg_loss` **more negative**, but not always.

What matters:
- if `pg_loss` suddenly becomes extreme (very negative/positive) across updates, it often signals unstable advantage scale or too-aggressive updates.

Practical reading:
- **Small-ish, noisy** `pg_loss`: normal.
- **Sudden sustained large magnitude**: policy is taking big steps *in the surrogate objective*, which often corresponds to KL spikes / instability.
- **pg_loss near 0 for a long time** with decent entropy: often means advantages are tiny / inconsistent (critic/GAE issues or reward too sparse/flat).

### **`v_loss`**

Value function regression (critic). We use a **clipped value loss** similar in spirit to policy clipping.

Interpretation:
- low `v_loss` means the critic predicts returns on the current rollout distribution well
- but even a good critic doesn’t guarantee good policy

For long-horizon scheduling, value learning can be hard; noisy value estimates → noisy advantages → unstable policy updates.

Practical reading:
- **v_loss very high and not decreasing**: critic can’t fit returns → advantages are noisy → policy learning will be noisy.
- **v_loss very low** doesn’t guarantee success; it can mean the critic learned a “boring” baseline while the policy is still bad.
- **v_loss spikes** that line up with eval drops: often means the **policy changed enough to change what states it visits**, so the critic is briefly wrong on the new rollout distribution (advantages get noisier until the critic catches up).

### **`entropy`**

We log the mean action entropy on minibatches.

Interpretation:
- high entropy = policy is uncertain / exploring
- low entropy = policy is sharp / deterministic

Entropy can be “too high” (policy stays random) or “too low” (prematurely collapses to no-op or a bad heuristic).

In our code we additionally clamp *entropy change per update* (`entropy_jump_clip`) so the exploration level doesn’t swing wildly update-to-update.

Practical reading (scheduler-specific):
- If **entropy collapses** while `noop_frac` rises → classic “no-op collapse / do-nothing policy”.
- If **entropy stays very high** and eval never improves → policy stays too random (learning rate too low, advantages too noisy, or reward too weak).

Important nuance:
- In masked action spaces, entropy isn’t “over all actions”; it’s entropy over **valid actions** after masking. If there are very few valid actions, entropy will naturally be lower.

### **`approx_kl_last / mean / max`**

This is an estimate of how much the new policy changed from the old one on the rollout batch.

We use a common estimator (used in many PPO baselines):

$$
\mathrm{approx\_kl} \approx \mathbb{E}\big[(r_t - 1) - \log r_t\big]
$$

Interpretation:
- small KL: update is “small”, policy stayed close
- large KL: update is “big”, policy moved far

We track:
- **last**: last minibatch value (debugging)
- **mean**: average over minibatches (typical update size)
- **max**: worst minibatch (guardrail)

And if `target_kl` is set, we stop PPO epochs early when `approx_kl_last` exceeds it.

Important nuance:
- this KL is computed on the **rollout’s state-action distribution**, not “all possible states”.
- it’s still a useful stability signal, but it’s not a perfect global distance.

Practical reading:
- **KL spikes** are the best “PPO is updating too aggressively” alarm you have.
- `approx_kl_max` is the most conservative: a single bad minibatch can push you into a bad region.
- If you see eval collapse and KL is **high** around the same time → overshoot / too aggressive.
- If you see eval collapse and KL is **tiny** → not “one big update”; suspect slow drift, advantage noise, or reward nonstationarity.

### **`invalid_frac`**

This is the fraction of rollout steps where the env reports `info["invalid_action"]=True`.

With **hard action masking in logits**, you should expect:
- `invalid_frac ≈ 0.0` almost always.

If `invalid_frac` becomes non-trivial:
- **Mask mismatch bug**: the mask used for logits doesn’t match what `env.step()` checks as feasible.
- **Numerical issue**: logits for all valid actions became very negative and the distribution is effectively broken (rare if masking is correct).
- **Observation/mask timing bug**: using stale `info["action_mask"]` (off by one step) — one of the easiest mistakes.

### **`noop_frac`**

Fraction of rollout steps where the chosen action is `0` (no-op).

Interpretation depends heavily on workload:
- Some no-ops are legitimate (nothing feasible to start right now).
- But persistent high no-op when there are feasible jobs is “policy is avoiding commitment”.

Practical reading:
- **High noop_frac early** can be normal if the policy’s initial logits prefer no-op.
- If **noop_frac trends upward** and eval trends downward → no-op collapse (policy learned that “doing nothing is safer” under the reward).
- If `noop_frac` differs a lot between training and eval:
  - training rollouts include exploration (sampling), and can be spikier
  - eval is greedy and averaged over full episodes (less spiky)

---

## 4) Why “best eval early then collapse later” happens

This is the part that feels unlike supervised learning.

Common failure modes in PPO on long-horizon problems:

- **Policy drift**: even if each update is small, many updates can slowly move you away from a good region.
- **Overshoot**: an update changes the policy enough that it stops visiting “good” trajectories, so subsequent rollouts become worse and training follows them.
- **Critic/advantage noise**: if the critic is wrong, advantages point the policy in the wrong direction.
- **Reward landscape weirdness**: shaped rewards can produce local optima that are “good enough” early; later training “optimizes” the surrogate and accidentally breaks the behavior.

This is why we added:
- deterministic multi-block eval (to separate “real” drift from eval noise)
- LR schedules + smaller updates
- KL logging / target-KL early stop

---

## 5) The debugging workflow (how to actually reason from curves)

This is the process that replaces “traditional ML instincts”.

### Step A — confirm you’re looking at a reliable curve

- Trust: `eval_mean_return`, `eval_min_block_return`, and per-block returns.
- Use: `mean_ep_return` only as a sanity check.

If eval is noisy, fix measurement before tuning:
- use deterministic eval (argmax)
- use fixed seeds
- use multi-block eval (mean + worst-block)

### Step B — classify the failure mode by the **shape** of eval + robustness

1) **Never improves (flat line)**:
- policy can’t find signal, or updates too small

2) **Improves then collapses**:
- overshoot or drift, or critic/advantage instability

3) **Mean improves but worst-block collapses**:
- specialization / brittleness (not robust)

4) **Sawtooth / high-frequency oscillations**:
- too aggressive updates or too small rollout batch

### Step C — use optimization stats to confirm the “why”

Use this table:

- **Eval drop + KL spike** ⇒ overshoot / too aggressive.
  - knobs: lower LR, fewer epochs, smaller clip, stronger target_kl, larger batch.

- **Eval drop + KL tiny** ⇒ slow drift or advantage noise.
  - knobs: improve critic (capacity, LR), longer rollouts, reward simplification, better normalization, reduce reward terms that introduce nonstationarity.

- **Eval flat + entropy high + KL tiny** ⇒ learning rate too low or advantages too weak.
  - knobs: raise LR slightly, increase epochs, improve reward density, reduce noise.

- **Eval flat + entropy low + noop_frac high** ⇒ no-op collapse / premature determinism.
  - knobs: increase entropy early, penalize noop carefully, architecture bias fixes (we already reduced noop-head domination).

### Step D — use “action-space stats” to catch bugs fast

- `invalid_frac > 0` with hard masking: treat as a **bug until proven otherwise**.
- sudden change in `noop_frac` with stable entropy: often indicates env dynamics changed (workload seeds/config) or reward gradient changed.

---

## 6) Common patterns and what they mean (scheduler-specific)

### Pattern: “High eval at first checkpoint, then worse later”

Common causes:
- first checkpoint still close to “random feasible” which does okay
- PPO then optimizes shaped reward in a way that breaks throughput behavior
- critic/advantage noise pushes policy away from the useful heuristic

How to validate:
- compare PPO to `RAND_FEAS_NO0` baseline
- see if later policy increases `noop_frac` or changes entropy drastically
- check if worst-block return falls first (robustness collapse)

### Pattern: “No-op collapse”

Signature:
- `noop_frac` rises over updates
- entropy falls (policy becomes confident in no-op)
- eval return drops

Common root causes:
- no-op is “safe” under reward (penalty too small, or deadline miss penalty dominates)
- network has an architectural bias toward no-op (we mitigated this by integrating noop scoring)
- advantages are too noisy so policy gravitates to low-variance actions

### Pattern: “Eval improves, KL stays low, then still collapses”

This is the confusing one.

Interpretation:
- It’s not “one big jump”.
- It’s “many small steps” accumulating into a bad region, or the policy slowly changing what states it visits so the learned heuristic stops generalizing to the eval seeds (a gradual policy-induced data shift).

What to try:
- exponential LR decay (you already did)
- fewer epochs
- slightly smaller clip
- more robust objective (optimize worst-block / risk-sensitive variants) — future work

### Pattern: “Value loss dominates”

Signature:
- `v_loss` much larger than policy loss scale and doesn’t settle
- eval is erratic

Interpretation:
- critic is the bottleneck; advantages are garbage

What to try:
- larger value net, lower value LR, more rollout steps, simpler reward

---

## 7) A quick “symptom → suspect → check” cheat sheet

- **`invalid_frac` non-zero** → suspect bug → check mask timing and feasibility definition.
- **`noop_frac` high + entropy low** → no-op collapse → check reward/noop penalty + architecture.
- **`approx_kl_max` spikes then eval drops** → overshoot → lower LR/epochs, tighten target_kl.
- **`approx_kl_max` tiny but eval drifts down** → slow drift → stronger LR decay, fewer epochs, simplify reward, improve critic.
- **`mean improves, min-block drops`** → brittleness → compare per-block returns, consider training diversity / objective.

---

## 8) How to read the curves in practice (for this repo)

When you look at a run CSV:

1. Look at **`eval_min_block_return`** first.
   - If it drops hard, you have a robustness collapse.
2. Compare **`eval_mean_return`** vs baselines.
   - That answers “is PPO beating heuristics on these workloads?”
3. Then check **`entropy`** and **`noop_frac`**.
   - If `noop_frac` shoots up and entropy drops, that’s a no-op collapse.
4. Then check **`approx_kl_max`** and whether **early stop** is firing.
   - If collapse happens with rising KL: too-aggressive updates.
   - If collapse happens with tiny KL: likely advantage/value/reward issues or slow drift.

---

## 9) Answering your intuition questions directly

### “Why doesn’t it look like a regression loss curve?”

Because the optimization target is not “match labels”, it’s “improve expected return under the current policy-induced data distribution”, using a constrained surrogate.

The “training loss” is not a stable proxy for eval return.

### “If the best point is early, is training useless after that?”

Not necessarily. It can mean:
- the policy found a good heuristic quickly, and then updates drift
- OR your eval set is narrow (we fixed this with multi-block seeds)

In practice, people often select the best checkpoint by eval return even if training continues.

---

## 10) Where the math is in our code (anchors)

- **Ratio clipping**: inside `PPOAgent.train()` where `ratio = exp(new_logprobs - old_logprobs)` and `torch.clamp(ratio, 1±clip_coef)` is used.
- **Value clipping**: where `v_pred_clipped = old_values + clamp(new_values-old_values, ±clip_coef)` is used.
- **Approx KL**: where `approx_kl = ((ratio - 1) - logratio).mean()` is computed.
- **Action masking**: `_mask_logits()` sets invalid logits to `-1e9`.
- **Multi-block eval**: evaluation loop that computes per-block returns and min/max.

---

## 11) One last mindset shift: “debugging PPO” is debugging a control system

You’re not just fitting a model; you’re tuning a feedback loop:

- policy produces data
- data trains policy
- policy changes data distribution

So the right debugging questions are:
- “Did my measurement change?” (seeds, determinism, block sizing)
- “Did my data distribution change?” (policy became different; reset semantics)
- “Did my update step size change?” (LR/epochs/clip/KL)
- “Did my learning signal change?” (reward shaping, critic quality)
