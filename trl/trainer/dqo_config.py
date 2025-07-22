# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Optional, Union

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

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the
            [`DQOTrainer`] is provided as a string.
        ref_model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `ref_model` argument of the
            [`DQOTrainer`] is provided as a string.
        model_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        force_use_ref_model (`bool`, *optional*, defaults to `False`):
            If you provide a PEFT model as the active model and wish to use a different model for the `ref_model`, set
            this flag to `True`.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        padding_value (`int` or `None`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Padding value to use for labels.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt.
        max_completion_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the completion.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the full sequence (prompt + completion).
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the sequence exceeds `max_length`. Possible values are `"keep_end"` and
            `"keep_start"`.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute the log probabilities from the reference model. Setting this to `True` allows
            training without needing the reference model during training, which can help reduce GPU memory usage. If
            set to `False` (default), the reference model will be used during training to compute log probabilities
            on-the-fly.
        precompute_ref_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Batch size to use when precomputing reference model log probabilities. This can be set higher than the
            training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
            training and `per_device_eval_batch_size` for evaluation.
        tools (`Optional[list[Union[dict, Callable]]]`, *optional*, defaults to `None`):
            List of tools (callable functions) that will be accessible to the model. If the template does not support
            function calling, this argument will have no effect.

        > Parameters that control the DQO training

        beta (`float`, *optional*, defaults to `0.1`):
            Temperature parameter for the DQO loss. Higher β means less deviation from the reference model.
        gamma (`float`, *optional*, defaults to `0.99`):
            Discount factor for Q-value computation in DQO. Controls how much future rewards are discounted.
        tau (`float`, *optional*, defaults to `0.005`):
            Soft update parameter for target Q-network updates. Controls the rate of target network updates.
        q_learning_rate (`float`, *optional*, defaults to `3e-4`):
            Learning rate for the Q-network optimizer, separate from the policy learning rate.
        target_update_freq (`int`, *optional*, defaults to `100`):
            Frequency (in steps) for updating the target Q-network.
        use_double_q (`bool`, *optional*, defaults to `True`):
            Whether to use Double Q-learning to reduce overestimation bias.
        q_network_hidden_size (`int`, *optional*, defaults to `256`):
            Hidden size for the Q-network layers.
        q_network_num_layers (`int`, *optional*, defaults to `2`):
            Number of hidden layers in the Q-network.
        exploration_noise (`float`, *optional*, defaults to `0.1`):
            Standard deviation of exploration noise added to actions during training.
        min_q_weight (`float`, *optional*, defaults to `1.0`):
            Weight for the minimum Q-value term in the DQO loss.
        reference_free (`bool`, *optional*, defaults to `False`):
            Whether to ignore the provided reference model and implicitly use a reference model that assigns equal
            probability to all responses.

        > Parameters that control the logging

        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Whether to generate and log completions from both the model and the reference model to W&B or Comet during
            evaluation.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs", "ref_model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DQOTrainer` is provided as a string."
        },
    )
    ref_model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `ref_model` argument "
            "of the `DQOTrainer` is provided as a string."
        },
    )
    model_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
    )
    force_use_ref_model: bool = field(
        default=False,
        metadata={
            "help": "If you provide a PEFT model as the active model and wish to use a different model for the "
            "`ref_model`, set this flag to `True`."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    padding_value: Optional[int] = field(
        default=None,
        metadata={"help": "Padding value to use. If `None`, the padding value of the tokenizer is used."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Padding value to use for labels."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt."},
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the completion."},
    )
    max_length: Optional[int] = field(
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
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute the log probabilities from the reference model. Setting this to `True` "
            "allows training without needing the reference model during training, which can help reduce GPU memory "
            "usage. If set to `False` (default), the reference model will be used during training to compute log "
            "probabilities on-the-fly."
        },
    )
    precompute_ref_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use when precomputing reference model log probabilities. This can be set higher "
            "than the training batch size to speed up preprocessing. If `None`, defaults to "
            "`per_device_train_batch_size` for training and `per_device_eval_batch_size` for evaluation."
        },
    )
    tools: Optional[list[Union[dict, Callable]]] = field(
        default=None,
        metadata={
            "help": "List of tools (callable functions) that will be accessible to the model. If the template does "
            "not support function calling, this argument will have no effect."
        },
    )

    # Parameters that control the DQO training
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Temperature parameter for the DQO loss. Higher β means less deviation from the reference model."
        },
    )
    gamma: float = field(
        default=0.99,
        metadata={
            "help": "Discount factor for Q-value computation in DQO. Controls how much future rewards are discounted."
        },
    )
    tau: float = field(
        default=0.005,
        metadata={
            "help": "Soft update parameter for target Q-network updates. Controls the rate of target network updates."
        },
    )
    q_learning_rate: float = field(
        default=3e-4,
        metadata={
            "help": "Learning rate for the Q-network optimizer, separate from the policy learning rate."
        },
    )
    target_update_freq: int = field(
        default=100,
        metadata={
            "help": "Frequency (in steps) for updating the target Q-network."
        },
    )
    use_double_q: bool = field(
        default=True,
        metadata={
            "help": "Whether to use Double Q-learning to reduce overestimation bias."
        },
    )
    q_network_hidden_size: int = field(
        default=256,
        metadata={
            "help": "Hidden size for the Q-network layers."
        },
    )
    q_network_num_layers: int = field(
        default=2,
        metadata={
            "help": "Number of hidden layers in the Q-network."
        },
    )
    exploration_noise: float = field(
        default=0.1,
        metadata={
            "help": "Standard deviation of exploration noise added to actions during training."
        },
    )
    min_q_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the minimum Q-value term in the DQO loss."
        },
    )
    reference_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the provided reference model and implicitly use a reference model that assigns "
            "equal probability to all responses."
        },
    )

    # Parameters that control the logging
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate and log completions from both the model and the reference model to W&B, MLFLow "
            "or Comet during evaluation."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        super().__post_init__()
