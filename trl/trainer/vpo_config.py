# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class VPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`VPOTrainer`].

    This class extends [`GRPOConfig`] with parameters specific to Vector Policy Optimization (VPO) training. For a full
    list of training arguments, please refer to the [`GRPOConfig`] and [`~transformers.TrainingArguments`]
    documentation. Note that default values in this class may differ from those in the parent classes.

    VPO is GRPO with a **set-level, stochastically-scalarized reward**. Each rollout emits a *set* of `num_answers`
    candidate answers in a single autoregressive chain. Every answer is scored by a *vector* reward `r(x, y) ∈ ℝ^d`
    (one component per reward function). For each prompt group, `num_scalarizations` weight vectors are sampled from a
    Dirichlet distribution and the *set* is scored with:

    ```
    R̂(S) = (1 / K) · Σ_k  max_{s ∈ S}  w_kᵀ r(x, s)
    ```

    This set reward `R̂(S)` is then consumed by GRPO exactly like a scalar reward (group-relative normalization +
    clipped policy-gradient loss). Optimizing it rewards *diverse, competent* sets rather than a single dominant mode,
    which improves test-time search (best-of-N, pass@k).

    Parameters:
        > Parameters specific to VPO

        num_answers (`int`, *optional*, defaults to `3`):
            Number of candidate answers the policy is expected to emit per rollout (the set size `m`). The completion
            is parsed into this many answers using `answer_separator` / `answer_pattern`. If fewer answers are parsed,
            the last one is repeated to reach `num_answers`; if more are parsed, only the first `num_answers` are kept.
        num_scalarizations (`int`, *optional*, defaults to `16`):
            Number of Dirichlet weight vectors `K` sampled per prompt group to estimate the set reward `R̂(S)`. The
            same `K` weight vectors are shared across all generations of a group so that the sets are scored
            comparably. Larger values reduce the variance of the set-reward estimate at the cost of more compute.
        scalarization_alpha (`float`, *optional*, defaults to `1.0`):
            Concentration parameter of the symmetric Dirichlet distribution `Dir(alpha · 1_d)` used to sample
            scalarization weights over the `d` reward objectives. `alpha=1.0` samples uniformly from the probability
            simplex; smaller values concentrate weight on a single objective, larger values push weights toward the
            uniform mixture.
        answer_separator (`str`, *optional*, defaults to `"\n"`):
            String used to split a completion into its `num_answers` candidate answers. Ignored when `answer_pattern`
            is set. Empty segments (after stripping) are discarded before truncating/padding to `num_answers`.
        answer_pattern (`str`, *optional*):
            Regular expression used to extract candidate answers from a completion. If set, it overrides
            `answer_separator`: every match is treated as an answer (the last capturing group is used when the pattern
            contains groups, otherwise the whole match). Useful for structured outputs, e.g. `r"<answer>(.*?)</answer>"`.

        > Inherited parameters with notes for VPO

        num_generations (`int`, *optional*, defaults to `8`):
            Size of the GRPO group `G`. VPO normalizes the set reward `R̂(S)` within each group, so this must be
            `>= 2` (the inherited GRPO default of `8` is a good starting point).
        reward_weights (`list[float]`, *optional*):
            **Unused by VPO.** Objective trade-offs are handled by the sampled Dirichlet scalarization weights, not by
            fixed reward weights. A value passed here is ignored (with a warning).
    """

    # VPO specific parameters
    num_answers: int = field(
        default=3,
        metadata={
            "help": "Number of candidate answers emitted per rollout (the set size m). The completion is parsed into "
            "this many answers using `answer_separator` / `answer_pattern`."
        },
    )
    num_scalarizations: int = field(
        default=16,
        metadata={
            "help": "Number of Dirichlet weight vectors K sampled per prompt group to estimate the set reward "
            "R_hat(S) = (1/K) * sum_k max_{s in S} w_k . r(x, s). Shared across a group's generations."
        },
    )
    scalarization_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Concentration of the symmetric Dirichlet Dir(alpha * 1_d) used to sample scalarization weights "
            "over the d objectives. 1.0 = uniform over the simplex."
        },
    )
    answer_separator: str = field(
        default="\n",
        metadata={
            "help": "String used to split a completion into its candidate answers. Ignored when `answer_pattern` is "
            "set."
        },
    )
    answer_pattern: str | None = field(
        default=None,
        metadata={
            "help": "Regex used to extract candidate answers from a completion. Overrides `answer_separator`. The last "
            "capturing group is used when present, otherwise the whole match."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.num_answers < 1:
            raise ValueError(f"`num_answers` must be >= 1, got {self.num_answers}.")
        if self.num_scalarizations < 1:
            raise ValueError(f"`num_scalarizations` must be >= 1, got {self.num_scalarizations}.")
        if self.scalarization_alpha <= 0:
            raise ValueError(f"`scalarization_alpha` must be > 0, got {self.scalarization_alpha}.")
