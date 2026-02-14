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
from typing import Any

from transformers import TrainingArguments


@dataclass
class DQOConfig(TrainingArguments):
    r"""
    Configuration class for the [`DQOTrainer`].

    This class includes only the parameters that are specific to DQO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the
            [`DQOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the full sequence (prompt + completion).
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the sequence exceeds `max_length`. Possible values are `"keep_end"` and
            `"keep_start"`.

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `0.03`):
            KL regularization parameter (temperature) that controls the deviation from the reference model.
            Higher β means less deviation from the reference model.
        lambda_return (`float`, *optional*, defaults to `1.0`):
            Lambda parameter for λ-return in temporal difference learning. Value between 0 and 1.
            λ=1.0 uses Monte Carlo returns (full trajectory), λ=0.0 uses one-step TD.
        importance_sampling_clip (`float`, *optional*, defaults to `10.0`):
            Clipping value for importance sampling ratios to prevent gradient explosion.
            The ratio is clipped to [1/clip, clip].
        value_learning_rate (`float`, *optional*, defaults to `1e-5`):
            Learning rate for the value function network. If `None`, uses the same as the policy learning rate.
        value_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for the value model initialization.
        process_reward_scale (`float`, *optional*, defaults to `1.0`):
            Scale factor for process rewards when they are available in the dataset.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs", "value_model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=5e-7,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model and reference model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DQOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum length of the full sequence (prompt + completion)."},
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={
            "help": "Truncation mode to use when the sequence exceeds `max_length`. Possible values are `'keep_end'` "
            "and `'keep_start'`.",
            "choices": ["keep_end", "keep_start"],
        },
    )

    # Parameters that control the training
    beta: float = field(
        default=0.03,
        metadata={
            "help": "KL regularization parameter (temperature) that controls the deviation from the reference model. "
            "Higher β means less deviation from the reference model."
        },
    )
    lambda_return: float = field(
        default=1.0,
        metadata={
            "help": "Lambda parameter for λ-return in temporal difference learning. Value between 0 and 1. "
            "λ=1.0 uses Monte Carlo returns (full trajectory), λ=0.0 uses one-step TD."
        },
    )
    importance_sampling_clip: float = field(
        default=10.0,
        metadata={
            "help": "Clipping value for importance sampling ratios to prevent gradient explosion. "
            "The ratio is clipped to [1/clip, clip]."
        },
    )
    value_learning_rate: float | None = field(
        default=1e-5,
        metadata={
            "help": "Learning rate for the value function network. If `None`, uses the same as the policy learning rate."
        },
    )
    value_model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Keyword arguments for the value model initialization."},
    )
    process_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scale factor for process rewards when they are available in the dataset."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        if self.lambda_return < 0.0 or self.lambda_return > 1.0:
            raise ValueError(f"lambda_return must be between 0 and 1, got {self.lambda_return}")

        if self.importance_sampling_clip <= 0:
            raise ValueError(f"importance_sampling_clip must be positive, got {self.importance_sampling_clip}")

        super().__post_init__()
