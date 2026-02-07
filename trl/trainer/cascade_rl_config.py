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
from typing import Any

from transformers import TrainingArguments


@dataclass
class CascadeRLStage:
    """
    Configuration for a single stage in Cascade RL training.
    
    Args:
        name (`str`):
            Name of this training stage (e.g., "alignment", "math", "code").
        dataset (`str` or `Dataset`):
            Dataset to use for this stage. Can be a HuggingFace dataset name or a Dataset object.
        trainer_class (`str`):
            Name of the trainer class to use for this stage (e.g., "DPOTrainer", "GRPOTrainer").
        trainer_config (`dict` or `TrainingArguments`):
            Configuration for the trainer at this stage.
        reward_funcs (`list` or `None`, *optional*):
            Reward functions to use for this stage (only applicable for RL stages like GRPO).
        eval_dataset (`str` or `Dataset`, *optional*):
            Evaluation dataset for this stage.
        preserve_checkpoints (`bool`, *optional*, defaults to `True`):
            Whether to save checkpoints from this stage separately.
    """

    name: str
    dataset: str | Any
    trainer_class: str
    trainer_config: dict | TrainingArguments
    reward_funcs: list | None = None
    eval_dataset: str | Any | None = None
    preserve_checkpoints: bool = True


@dataclass
class CascadeRLConfig(TrainingArguments):
    r"""
    Configuration class for the [`CascadeRLTrainer`].

    Cascade RL orchestrates sequential, domain-wise reinforcement learning training, as introduced in the 
    [Nemotron-Cascade paper](https://huggingface.co/papers/2506.xxxxx). This approach trains models stage-by-stage
    across different domains, starting with alignment (RLHF) and progressing through domain-specific RL stages.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        stages (`list[CascadeRLStage]`):
            List of training stages to execute sequentially. Each stage defines a domain-specific training phase
            with its own dataset, trainer, and configuration.
        continuous_training (`bool`, *optional*, defaults to `True`):
            Whether to continue training from the previous stage's checkpoint. If `False`, each stage starts from
            the base model.
        eval_on_all_stages (`bool`, *optional*, defaults to `True`):
            Whether to evaluate on all previous stages' eval datasets after each stage completes. This helps
            monitor performance degradation across domains.
        early_stopping_threshold (`float`, *optional*):
            If specified, stops training if performance on any previous stage degrades by more than this threshold.
            Only applicable when `eval_on_all_stages=True`.
        stage_output_dir (`str`, *optional*):
            Base directory for saving stage-specific outputs. If `None`, uses `output_dir` from base TrainingArguments.
            Each stage will create a subdirectory: `{stage_output_dir}/{stage_name}`.
    """

    stages: list[CascadeRLStage] = field(default_factory=list)
    continuous_training: bool = field(
        default=True,
        metadata={
            "help": "Whether to continue training from the previous stage's checkpoint. If False, each stage starts "
            "from the base model."
        },
    )
    eval_on_all_stages: bool = field(
        default=True,
        metadata={
            "help": "Whether to evaluate on all previous stages' eval datasets after each stage completes. This "
            "helps monitor performance degradation across domains."
        },
    )
    early_stopping_threshold: float | None = field(
        default=None,
        metadata={
            "help": "If specified, stops training if performance on any previous stage degrades by more than this "
            "threshold. Only applicable when eval_on_all_stages=True."
        },
    )
    stage_output_dir: str | None = field(
        default=None,
        metadata={
            "help": "Base directory for saving stage-specific outputs. If None, uses output_dir from base "
            "TrainingArguments. Each stage will create a subdirectory: {stage_output_dir}/{stage_name}."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        
        if not self.stages:
            raise ValueError("At least one training stage must be defined in CascadeRLConfig.stages")
        
        if self.stage_output_dir is None:
            self.stage_output_dir = self.output_dir

    def add_stage(
        self,
        name: str,
        dataset: str | Any,
        trainer_class: str,
        trainer_config: dict | TrainingArguments,
        reward_funcs: list | None = None,
        eval_dataset: str | Any | None = None,
        preserve_checkpoints: bool = True,
    ):
        """
        Add a new training stage to the cascade.
        
        Args:
            name: Name of this training stage
            dataset: Dataset to use for this stage
            trainer_class: Name of the trainer class (e.g., "DPOTrainer", "GRPOTrainer")
            trainer_config: Configuration for the trainer
            reward_funcs: Reward functions for RL stages (optional)
            eval_dataset: Evaluation dataset (optional)
            preserve_checkpoints: Whether to save checkpoints from this stage
        """
        stage = CascadeRLStage(
            name=name,
            dataset=dataset,
            trainer_class=trainer_class,
            trainer_config=trainer_config,
            reward_funcs=reward_funcs,
            eval_dataset=eval_dataset,
            preserve_checkpoints=preserve_checkpoints,
        )
        self.stages.append(stage)
