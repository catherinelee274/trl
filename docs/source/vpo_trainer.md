# VPO Trainer

[![model badge](https://img.shields.io/badge/All_models-VPO-blue)](https://huggingface.co/models?other=vpo,trl)

## Overview

TRL supports the Vector Policy Optimization (VPO) Trainer for training language models that produce *diverse, competent sets* of answers, as described in the paper [Vector Policy Optimization: Training for Diversity Improves Test-Time Search](https://huggingface.co/papers/2605.22817) by Mahdi Bahlous-Boldi, Raviteja Puri, Idan Shenfeld, Aviral Kumar, Mehul Damani, Sebastian Risi, Omar Khattab, Joey Hong, and Lakshya A. Agrawal.

The abstract from the paper is the following:

> When a language model is deployed inside a test-time search loop—best-of-N, pass@k, or evolutionary procedures such as AlphaEvolve—the post-training objective should produce a diverse, competent set of candidates rather than collapse to a single mode. We introduce Vector Policy Optimization (VPO), which trains the policy to emit a set of answers scored by a vector reward and combined through stochastic scalarization. VPO is a drop-in modification of Group Relative Policy Optimization that consistently improves test-time search performance and reward-space diversity over scalar baselines.

VPO is **GRPO with a set-level, stochastically-scalarized reward**. Where [GRPO](grpo_trainer) scores a single completion with a scalar reward, VPO:

1. **Multi-answer rollouts** — the policy emits a *set* of `num_answers` candidate answers in one autoregressive chain.
2. **Vector reward** — each answer is scored by a *vector* reward \\( r(x, y) \in \mathbb{R}^d \\), one component per reward function (the `d` objectives passed as `reward_funcs`).
3. **Stochastic scalarization** — for each prompt group, `num_scalarizations` weight vectors \\( w \sim \mathrm{Dir}(\alpha) \\) are sampled (shared across the group) and the set is scored with

$$
\hat{R}(S) = \frac{1}{K} \sum_{k=1}^{K} \max_{s \in S}\, w_k^\top r(x, s).
$$

4. **GRPO** — \\( \hat{R}(S) \\) is consumed as the per-rollout scalar reward (group-relative normalization and clipped policy-gradient loss are inherited unchanged from [`GRPOTrainer`]).

Optimizing the set reward rewards covering the objectives with *different* answers rather than collapsing to one dominant mode, which improves test-time search such as best-of-N and pass@k.

This trainer extends [`GRPOTrainer`]; the [GRPO documentation](grpo_trainer) covers the shared mechanics (generation, vLLM, PEFT, logging, scaling) that also apply here.

## Quick start

VPO needs at least two reward objectives (the components of the vector reward) that *together* cannot be maximized by a single answer—otherwise there is no incentive to produce a diverse set. The self-contained example below trains [Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on a toy "multi-target" task: each prompt provides two integer targets, an answer is one integer, and each objective rewards closeness to one target. A single integer can be near only one target, so a *set* is needed to cover both.

```python
# train_vpo.py
import random

from datasets import Dataset
from trl import VPOConfig, VPOTrainer


# Build a toy multi-target dataset: each prompt carries two integer targets and asks the
# model to emit `num_answers` integers, one per line.
def make_dataset(n=1024):
    rows = []
    for _ in range(n):
        t0, t1 = random.randint(0, 100), random.randint(0, 100)
        rows.append({
            "prompt": [
                {"role": "system", "content": "Provide 3 candidate integers, one per line."},
                {"role": "user", "content": f"Give numbers near {t0} and near {t1}."},
            ],
            "target_0": t0,
            "target_1": t1,
        })
    return Dataset.from_list(rows)


dataset = make_dataset()


def _to_int(completion):
    text = completion if isinstance(completion, str) else completion[-1]["content"]
    try:
        return int(text.strip())
    except ValueError:
        return None


# Objective 0: closeness to the first target.
def closeness_to_target_0(completions, target_0, **kwargs):
    rewards = []
    for c, t in zip(completions, target_0):
        v = _to_int(c)
        rewards.append(-abs(v - t) if v is not None else -1000.0)
    return rewards


# Objective 1: closeness to the second target.
def closeness_to_target_1(completions, target_1, **kwargs):
    rewards = []
    for c, t in zip(completions, target_1):
        v = _to_int(c)
        rewards.append(-abs(v - t) if v is not None else -1000.0)
    return rewards


trainer = VPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[closeness_to_target_0, closeness_to_target_1],  # d = 2 objectives
    args=VPOConfig(
        output_dir="Qwen2.5-0.5B-VPO",
        num_answers=3,  # m: set size
        num_scalarizations=16,  # K: Dirichlet samples per group
        scalarization_alpha=1.0,  # uniform over the simplex
    ),
    train_dataset=dataset,
)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_vpo.py
```

## Expected dataset type

VPO uses the same [prompt-only](dataset_formats#prompt-only) dataset type as [GRPO](grpo_trainer). The dataset must contain a `prompt` column; any additional columns (e.g. per-objective targets such as `target_0`, `target_1`) are forwarded to the objective reward functions as keyword arguments.

Because VPO scores a *set*, the prompt (or the system message) should instruct the model to emit exactly `num_answers` candidate answers in the format expected by `answer_separator` (or `answer_pattern`). For example, with the default `answer_separator="\n"`, a system message like *"Provide 3 candidate answers, one per line."* makes the completion parseable into the set.

## How VPO parses the answer set

Each completion is parsed into exactly `num_answers` answers:

- By default the completion text is split on [`VPOConfig.answer_separator`] (default `"\n"`), and empty segments are discarded.
- Set [`VPOConfig.answer_pattern`] to a regex to extract structured answers instead, e.g. `answer_pattern=r"<answer>(.*?)</answer>"`. The last capturing group is used when present, otherwise the whole match.
- If fewer than `num_answers` answers are parsed, the last one is repeated (duplicates do not change the `max` over the set); if more are parsed, only the first `num_answers` are kept.

Every parsed answer is scored by every objective reward function, exactly like a standard TRL reward function but applied to a single answer.

## Logged metrics

In addition to the [metrics logged by GRPO](grpo_trainer#logged-metrics), VPO logs:

- `set_reward`: the mean set reward \\( \hat{R}(S) \\) across rollouts (this is also the value GRPO normalizes into the advantage, reported as `rewards/vpo_set_reward/mean`).
- `objectives/{name}/mean`: the mean of each objective's reward component, averaged over all answers in all sets.
- `answer_diversity`: the spread (standard deviation) of which answer index is selected as the per-scalarization argmax. Higher values indicate that different scalarizations are best served by different answers in the set—the signal VPO is trained to increase.

## Notes

- **Objectives must be callables.** Each entry in `reward_funcs` must be a callable reward function with the standard TRL signature scoring a single answer. Reward *models* (string ids or `nn.Module`) are not supported as VPO objectives.
- **`reward_weights` is ignored.** Objective trade-offs are handled by the sampled Dirichlet scalarization weights, not by fixed reward weights. A value passed to `reward_weights` is ignored with a warning.
- **`num_generations` is the GRPO group size `G`** and must be `>= 2`; the same `num_scalarizations` weight vectors are shared across a group so that its sets are scored comparably.

## VPOTrainer

[[autodoc]] VPOTrainer
    - train
    - save_model
    - push_to_hub

## VPOConfig

[[autodoc]] VPOConfig
