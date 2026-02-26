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
class APOConfig(GRPOConfig):
    r"""
    Configuration class for the [`APOTrainer`].

    This class extends [`GRPOConfig`] with parameters specific to A★-PO training. For a full list of training
    arguments, please refer to the [`GRPOConfig`] and [`~transformers.TrainingArguments`] documentation. Note that
    default values in this class may differ from those in the parent classes.

    Parameters:
        > Parameters specific to A★-PO

        beta1 (`float`, *optional*, defaults to `0.5`):
            Temperature for the offline soft-value estimation. Used to compute the optimal value function:
            `v*(x) = beta1 * log(E_ref[exp(r(x,y) / beta1)])` over a set of pre-generated reference responses. A
            higher `beta1` results in a softer (more averaged) value estimate. Only used when `v_star_column` is not
            present in the dataset and the value function is computed online using the reference model.
        beta2 (`float`, *optional*, defaults to `1e-3`):
            Regression coefficient for the A★-PO least-squares loss. This controls the scale of the log-probability
            ratio in the objective: `L = E[(beta2 * log(π/π_ref) - A*(x, y))²]`, where `A*(x,y) = r(x,y) - v*(x)`.
            Smaller values make the update closer to a supervised regression on the advantage; larger values allow
            more deviation from the reference policy.
        v_star_column (`str`, *optional*, defaults to `"v_star"`):
            Name of the column in the training dataset that contains pre-computed optimal value estimates `v*(x)`.
            These are typically computed offline by sampling multiple responses per prompt from the reference model
            and applying the soft-value formula: `v*(x) = beta1 * log(mean(exp(r_i / beta1)))`. If this column is
            absent from the dataset, `v*(x) = 0` is used (which reduces A★-PO to supervised regression on the raw
            reward signal).

        > Inherited parameters with modified defaults for A★-PO

        num_generations (`int`, *optional*, defaults to `1`):
            A★-PO requires exactly **1 generation per prompt** at training time (Stage 2). The value function is
            estimated offline (Stage 1), so no group of completions is needed during training. Setting this to a
            value greater than `1` is not recommended and will raise a warning.
        beta (`float`, *optional*, defaults to `0.0`):
            This parameter is inherited from `GRPOConfig` but is **not used** as a KL penalty in the A★-PO loss. The
            reference model is always created for computing log-probability ratios required by the A★-PO objective.
            Setting this to a non-zero value will still cause a GRPO-style KL term to be added to the loss on top of
            the A★-PO objective, which is generally not recommended.
    """

    # A-PO specific parameters
    beta1: float = field(
        default=0.5,
        metadata={
            "help": "Temperature for offline soft-value estimation. Used as: "
            "v*(x) = beta1 * log(E_ref[exp(r/beta1)]). Only relevant when computing v_star online."
        },
    )
    beta2: float = field(
        default=1e-3,
        metadata={
            "help": "Regression coefficient in the A-PO squared loss: "
            "L = E[(beta2 * log(pi/pi_ref) - advantage)^2], where advantage = reward - v_star."
        },
    )
    v_star_column: str = field(
        default="v_star",
        metadata={
            "help": "Dataset column containing pre-computed optimal value estimates v*(x). "
            "If absent, v*(x) = 0 is used (reward-only regression)."
        },
    )

    # Override the default for num_generations: A-PO only needs 1 generation per prompt
    num_generations: int = field(
        default=1,
        metadata={
            "help": "Number of completions to generate per prompt. A-PO requires exactly 1."
        },
    )

    def __post_init__(self):
        # GRPOConfig validates that num_generations >= 2, which does not apply to A★-PO.
        # Temporarily set num_generations to 2 to pass GRPO's validation, then restore it.
        _original = self.num_generations
        self.num_generations = max(self.num_generations, 2)
        super().__post_init__()
        self.num_generations = _original
