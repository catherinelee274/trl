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

import os
import textwrap
from pathlib import Path
from typing import Any

from accelerate.logging import get_logger
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from .cascade_rl_config import CascadeRLConfig, CascadeRLStage


logger = get_logger(__name__)


class CascadeRLTrainer:
    """
    Trainer for Cascaded Reinforcement Learning (Cascade RL).
    
    This trainer orchestrates sequential, domain-wise reinforcement learning training as described in the
    Nemotron-Cascade paper. It trains models stage-by-stage across different domains, starting with 
    alignment (RLHF) and progressing through domain-specific RL stages like math, code, and software engineering.
    
    Key advantages of Cascade RL:
    - RLHF substantially improves response quality and boosts reasoning ability
    - Subsequent domain-wise RL stages rarely degrade previous benchmark performance
    - Each stage can have tailored hyperparameters and training curriculum
    - Simpler engineering than joint multi-domain training
    
    Example:
    
    ```python
    from trl import CascadeRLTrainer, CascadeRLConfig, CascadeRLStage
    from trl.rewards import accuracy_reward
    
    # Define training stages
    config = CascadeRLConfig(
        output_dir="./cascade-model",
        stages=[
            CascadeRLStage(
                name="alignment",
                dataset="my-alignment-dataset",
                trainer_class="DPOTrainer",
                trainer_config={"learning_rate": 1e-6, "num_train_epochs": 1}
            ),
            CascadeRLStage(
                name="math",
                dataset="my-math-dataset",
                trainer_class="GRPOTrainer",
                trainer_config={"learning_rate": 2e-6, "num_train_epochs": 1},
                reward_funcs=[accuracy_reward]
            ),
        ]
    )
    
    trainer = CascadeRLTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        config=config,
    )
    trainer.train()
    ```
    
    Args:
        model (`str` or `PreTrainedModel`):
            The base model to train. Can be a model ID or a PreTrainedModel instance.
        config (`CascadeRLConfig`):
            Configuration containing the cascade of training stages.
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`, *optional*):
            Processing class (tokenizer/processor) for the model. If None, will be loaded automatically.
    """
    
    _tag_names = ["trl", "cascade-rl"]
    _name = "Cascade RL"
    _paper = {
        "title": "Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models",
        "id": "2512.13607",
        "citation": textwrap.dedent("""\
            @article{wang2025nemotron,
                title={{Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models}},
                author={Boxin Wang and Chankyu Lee and Nayeon Lee and Sheng-Chieh Lin and Wenliang Dai and Yang Chen and Yangyi Chen and Zhuolin Yang and Zihan Liu and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping},
                year={2025},
                eprint={arXiv:2512.13607},
            }
            """),
    }
    
    def __init__(
        self,
        model: str | PreTrainedModel,
        config: CascadeRLConfig,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
    ):
        self.config = config
        self.base_model = model
        self.processing_class = processing_class
        self.current_model = None
        self.stage_history = []
        self.eval_results = {}
        
    def _get_trainer_class(self, trainer_name: str):
        """Dynamically import and return the trainer class."""
        # Import the trainer from trl.trainer module
        if trainer_name == "DPOTrainer":
            from .dpo_trainer import DPOTrainer
            return DPOTrainer
        elif trainer_name == "GRPOTrainer":
            from .grpo_trainer import GRPOTrainer
            return GRPOTrainer
        elif trainer_name == "RLOOTrainer":
            from .rloo_trainer import RLOOTrainer
            return RLOOTrainer
        elif trainer_name == "SFTTrainer":
            from .sft_trainer import SFTTrainer
            return SFTTrainer
        elif trainer_name == "RewardTrainer":
            from .reward_trainer import RewardTrainer
            return RewardTrainer
        elif trainer_name == "PPOTrainer":
            from .ppo_trainer import PPOTrainer
            return PPOTrainer
        elif trainer_name == "OnlineDPOTrainer":
            from .online_dpo_trainer import OnlineDPOTrainer
            return OnlineDPOTrainer
        else:
            raise ValueError(f"Unknown trainer class: {trainer_name}")
    
    def _load_dataset(self, dataset_spec: str | Dataset) -> Dataset:
        """Load a dataset from a specification (string path or Dataset object)."""
        if isinstance(dataset_spec, Dataset):
            return dataset_spec
        elif isinstance(dataset_spec, str):
            # Try to load from HuggingFace Hub or local path
            try:
                return load_dataset(dataset_spec, split="train")
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_spec}': {e}")
                raise
        else:
            raise ValueError(f"Invalid dataset specification: {type(dataset_spec)}")
    
    def _prepare_stage_config(self, stage: CascadeRLStage, stage_idx: int) -> dict:
        """Prepare the configuration for a specific stage."""
        # Convert trainer_config to dict if it's a TrainingArguments object
        if hasattr(stage.trainer_config, "to_dict"):
            config_dict = stage.trainer_config.to_dict()
        elif isinstance(stage.trainer_config, dict):
            config_dict = stage.trainer_config.copy()
        else:
            config_dict = {}
        
        # Set stage-specific output directory
        stage_output_dir = os.path.join(self.config.stage_output_dir, f"stage_{stage_idx:02d}_{stage.name}")
        config_dict["output_dir"] = stage_output_dir
        
        return config_dict
    
    def _train_stage(self, stage: CascadeRLStage, stage_idx: int):
        """Train a single stage of the cascade."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Stage {stage_idx + 1}/{len(self.config.stages)}: {stage.name}")
        logger.info(f"Trainer: {stage.trainer_class}")
        logger.info(f"{'='*80}\n")
        
        # Load dataset
        train_dataset = self._load_dataset(stage.dataset)
        eval_dataset = self._load_dataset(stage.eval_dataset) if stage.eval_dataset else None
        
        # Get trainer class
        TrainerClass = self._get_trainer_class(stage.trainer_class)
        
        # Prepare configuration
        stage_config_dict = self._prepare_stage_config(stage, stage_idx)
        
        # Determine which model to use for initialization
        if self.config.continuous_training and self.current_model is not None:
            model_to_use = self.current_model
            logger.info(f"Continuing training from previous stage checkpoint")
        else:
            model_to_use = self.base_model
            logger.info(f"Starting from base model")
        
        # Initialize trainer with stage-specific configuration
        trainer_kwargs = {
            "model": model_to_use,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": self.processing_class,
        }
        
        # Add reward functions for RL trainers
        if stage.reward_funcs is not None and "reward_funcs" in TrainerClass.__init__.__code__.co_varnames:
            trainer_kwargs["reward_funcs"] = stage.reward_funcs
        
        # Create config object for the trainer
        # This is a simplified approach - in practice, you'd need to properly instantiate the config class
        try:
            # Try to get the config class for this trainer
            config_class_name = stage.trainer_class.replace("Trainer", "Config")
            if config_class_name == "SFTConfig":
                from .sft_config import SFTConfig
                trainer_config = SFTConfig(**stage_config_dict)
            elif config_class_name == "DPOConfig":
                from .dpo_config import DPOConfig
                trainer_config = DPOConfig(**stage_config_dict)
            elif config_class_name == "GRPOConfig":
                from .grpo_config import GRPOConfig
                trainer_config = GRPOConfig(**stage_config_dict)
            elif config_class_name == "RLOOConfig":
                from .rloo_config import RLOOConfig
                trainer_config = RLOOConfig(**stage_config_dict)
            elif config_class_name == "RewardConfig":
                from .reward_config import RewardConfig
                trainer_config = RewardConfig(**stage_config_dict)
            elif config_class_name == "PPOConfig":
                from .ppo_config import PPOConfig
                trainer_config = PPOConfig(**stage_config_dict)
            elif config_class_name == "OnlineDPOConfig":
                from .online_dpo_config import OnlineDPOConfig
                trainer_config = OnlineDPOConfig(**stage_config_dict)
            else:
                # Fallback to TrainingArguments
                from transformers import TrainingArguments
                trainer_config = TrainingArguments(**stage_config_dict)
            
            trainer_kwargs["args"] = trainer_config
        except Exception as e:
            logger.warning(f"Could not create config object: {e}. Passing config dict directly.")
        
        # Instantiate trainer
        trainer = TrainerClass(**trainer_kwargs)
        
        # Train
        trainer.train()
        
        # Update current model to the trained model
        self.current_model = trainer.model
        
        # Save stage checkpoint if requested
        if stage.preserve_checkpoints:
            checkpoint_dir = os.path.join(stage_config_dict["output_dir"], "final_checkpoint")
            trainer.save_model(checkpoint_dir)
            logger.info(f"Saved stage checkpoint to {checkpoint_dir}")
        
        # Record stage completion
        self.stage_history.append({
            "stage_idx": stage_idx,
            "stage_name": stage.name,
            "trainer_class": stage.trainer_class,
            "output_dir": stage_config_dict["output_dir"],
        })
        
        return trainer
    
    def _evaluate_all_stages(self, current_stage_idx: int) -> dict[str, Any]:
        """Evaluate on all previous stages' eval datasets."""
        if not self.config.eval_on_all_stages:
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating on all previous stages' datasets")
        logger.info(f"{'='*80}\n")
        
        eval_results = {}
        
        for stage_idx in range(current_stage_idx + 1):
            stage = self.config.stages[stage_idx]
            if stage.eval_dataset is None:
                continue
            
            logger.info(f"Evaluating on stage {stage_idx + 1} ({stage.name}) eval dataset")
            
            # Load eval dataset
            eval_dataset = self._load_dataset(stage.eval_dataset)
            
            # Get trainer class and create a temporary trainer for evaluation
            TrainerClass = self._get_trainer_class(stage.trainer_class)
            
            # Create minimal config for evaluation
            eval_config_dict = {
                "output_dir": self.config.stage_output_dir,
                "per_device_eval_batch_size": 8,
            }
            
            # Create trainer for evaluation
            eval_kwargs = {
                "model": self.current_model,
                "eval_dataset": eval_dataset,
                "processing_class": self.processing_class,
            }
            
            # Add reward functions if needed
            if stage.reward_funcs is not None and "reward_funcs" in TrainerClass.__init__.__code__.co_varnames:
                eval_kwargs["reward_funcs"] = stage.reward_funcs
            
            try:
                # Create config
                config_class_name = stage.trainer_class.replace("Trainer", "Config")
                if config_class_name == "GRPOConfig":
                    from .grpo_config import GRPOConfig
                    eval_config = GRPOConfig(**eval_config_dict)
                elif config_class_name == "DPOConfig":
                    from .dpo_config import DPOConfig
                    eval_config = DPOConfig(**eval_config_dict)
                else:
                    from transformers import TrainingArguments
                    eval_config = TrainingArguments(**eval_config_dict)
                
                eval_kwargs["args"] = eval_config
            except Exception as e:
                logger.warning(f"Could not create eval config: {e}")
                continue
            
            try:
                eval_trainer = TrainerClass(**eval_kwargs)
                metrics = eval_trainer.evaluate()
                eval_results[f"stage_{stage_idx}_{stage.name}"] = metrics
                logger.info(f"Stage {stage_idx + 1} ({stage.name}) metrics: {metrics}")
            except Exception as e:
                logger.warning(f"Failed to evaluate on stage {stage_idx} ({stage.name}): {e}")
        
        return eval_results
    
    def _check_early_stopping(self, current_stage_idx: int, eval_results: dict[str, Any]) -> bool:
        """Check if training should stop early due to performance degradation."""
        if self.config.early_stopping_threshold is None or current_stage_idx == 0:
            return False
        
        # Compare current performance with baseline (first stage or previous stage)
        # This is a simplified implementation - in practice, you'd want more sophisticated logic
        for stage_key, metrics in eval_results.items():
            stage_idx = int(stage_key.split("_")[1])
            if stage_idx >= current_stage_idx:
                continue
            
            # Check if we have previous results for this stage
            previous_key = f"{stage_key}_previous"
            if previous_key in self.eval_results:
                previous_metrics = self.eval_results[previous_key]
                
                # Compare metrics (assuming lower loss is better)
                if "eval_loss" in metrics and "eval_loss" in previous_metrics:
                    current_loss = metrics["eval_loss"]
                    previous_loss = previous_metrics["eval_loss"]
                    degradation = (current_loss - previous_loss) / previous_loss
                    
                    if degradation > self.config.early_stopping_threshold:
                        logger.warning(
                            f"Performance degraded by {degradation:.2%} on {stage_key}, "
                            f"exceeding threshold of {self.config.early_stopping_threshold:.2%}"
                        )
                        return True
        
        return False
    
    def train(self):
        """Execute the full cascade RL training pipeline."""
        logger.info(f"Starting Cascade RL training with {len(self.config.stages)} stages")
        
        for stage_idx, stage in enumerate(self.config.stages):
            # Train current stage
            trainer = self._train_stage(stage, stage_idx)
            
            # Evaluate on all previous stages
            eval_results = self._evaluate_all_stages(stage_idx)
            self.eval_results[f"after_stage_{stage_idx}"] = eval_results
            
            # Check early stopping
            if self._check_early_stopping(stage_idx, eval_results):
                logger.warning(f"Early stopping triggered after stage {stage_idx + 1}")
                break
            
            # Store current eval results for next iteration
            for key, value in eval_results.items():
                self.eval_results[f"{key}_previous"] = value
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Cascade RL training completed!")
        logger.info(f"Trained {len(self.stage_history)} stages")
        logger.info(f"Final model available at: {self.stage_history[-1]['output_dir']}")
        logger.info(f"{'='*80}\n")
        
        return self.current_model
    
    def save_model(self, output_dir: str):
        """Save the final trained model."""
        if self.current_model is None:
            raise ValueError("No model to save. Please run train() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if hasattr(self.current_model, "save_pretrained"):
            self.current_model.save_pretrained(output_dir)
        
        # Save processing class
        if self.processing_class is not None and hasattr(self.processing_class, "save_pretrained"):
            self.processing_class.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
