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

import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin

from ..data_utils import is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .base_trainer import BaseTrainer
from .dqo_config import DQOConfig
from .utils import disable_dropout_in_model, empty_cache, flush_left, get_config_model_id, pad


@dataclass
class DataCollatorForDQO(DataCollatorMixin):
    """
    Data collator for DQO training. Supports both full trajectories and step-level process rewards.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        
        # Check if we have rewards
        if "rewards" in examples[0]:
            rewards = [torch.tensor(example["rewards"]) for example in examples]
        else:
            rewards = None
        
        # Pad sequences
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side="left")
        
        if rewards is not None:
            output["rewards"] = pad(rewards, padding_value=0.0, padding_side="left")
        
        return output


class ValueNetwork(nn.Module):
    """
    Value network that estimates the state value V(s) for DQO.
    
    Args:
        input_dim (`int`):
            Dimension of the input hidden states from the language model.
        hidden_dim (`int`, *optional*, defaults to `1024`):
            Dimension of the hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            values: Tensor of shape (batch_size, seq_len)
        """
        return self.net(hidden_states).squeeze(-1)


class DQOTrainer(BaseTrainer):
    """
    Trainer for Direct Q-function Optimization (DQO) method.

    DQO formulates the response generation process as a token-level MDP and uses the soft actor-critic
    framework to optimize a Q-function directly parameterized by the language model. Unlike bandit-based
    methods (e.g., DPO, DRO), DQO can leverage process supervision signals and is more suitable for
    multi-step reasoning tasks.

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either a string (model id) or a [`~transformers.PreTrainedModel`].
        ref_model (`PreTrainedModel`, *optional*):
            Reference model for KL regularization. If not provided, a copy of the model will be used.
        value_model (`nn.Module`, *optional*):
            Value network for estimating V(s). If not provided, will be created automatically.
        args (`DQOConfig`, *optional*):
            Configuration for this trainer.
        data_collator (`DataCollator`, *optional*):
            Data collator for batching. If not provided, [`DataCollatorForDQO`] will be used.
        train_dataset (`Dataset | IterableDataset`, *optional*):
            Training dataset.
        eval_dataset (`Dataset | IterableDataset`, *optional*):
            Evaluation dataset.
        processing_class (`PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin`, *optional*):
            Processing class for tokenization.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics during evaluation.
        callbacks (`list[TrainerCallback]`, *optional*):
            List of callbacks.
        optimizers (`tuple`, *optional*):
            Tuple of (optimizer, scheduler) for the policy model.
        optimizer_cls_and_kwargs (`tuple`, *optional*):
            Tuple of (optimizer_class, kwargs) for customizing the optimizer.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess logits for metrics.
    """

    _tag_names = ["trl", "dqo"]
    _name = "DQO"
    _paper = {
        "title": "Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization",
        "id": "2410.09302",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{ji2024dqo,
                title        = {{Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization}},
                author       = {Kaixuan Ji and Guanlin Liu and Ning Dai and Qingping Yang and Renjie Zheng and Zheng Wu and Chen Dun and Quanquan Gu and Lin Yan},
                year         = 2024,
                journal      = {arXiv preprint arXiv:2410.09302},
            }"""),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        value_model: nn.Module | None = None,
        args: DQOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DQOConfig(f"{model_name}-DQO")

        # Model and reference model setup
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            from ..models.utils import create_model_from_path
            model = create_model_from_path(model, **model_init_kwargs)
        
        model_id = get_config_model_id(model.config)
        
        if ref_model is None:
            ref_model = create_reference_model(model)
        elif isinstance(ref_model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            from ..models.utils import create_model_from_path
            ref_model = create_model_from_path(ref_model, **model_init_kwargs)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)

        # Handle pad token
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
        self.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        if self.pad_token_id is None:
            raise ValueError(
                f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary. "
                "Ensure that the `pad_token` exists in the vocabulary before using it as a padding token."
            )

        # Value network setup
        if value_model is None:
            hidden_size = model.config.hidden_size
            value_model = ValueNetwork(input_dim=hidden_size)
        self.value_model = value_model

        # Disable dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            disable_dropout_in_model(ref_model)

        # Data collator
        if data_collator is None:
            data_collator = DataCollatorForDQO(pad_token_id=self.pad_token_id)

        self.ref_model = ref_model
        self.beta = args.beta
        self.lambda_return = args.lambda_return
        self.importance_sampling_clip = args.importance_sampling_clip
        self.value_learning_rate = args.value_learning_rate or args.learning_rate
        self.process_reward_scale = args.process_reward_scale
        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        self.dataset_num_proc = args.dataset_num_proc

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Dataset preparation
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Set up value optimizer
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.value_learning_rate)

        # Prepare models for distributed training
        if not hasattr(self, "accelerator"):
            raise AttributeError("Your `Trainer` does not have an `accelerator` object.")

        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        self.value_model = self.accelerator.prepare_model(self.value_model)
        self.value_optimizer = self.accelerator.prepare_optimizer(self.value_optimizer)

        # Gradient accumulation requires scaled loss
        self.model_accepts_loss_kwargs = False

        # Add model tags
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args: DQOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Prepare dataset for DQO training."""
        if dataset is None:
            return None

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply chat template if needed
            is_chat = is_conversational(next(iter(dataset)))
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": None},
                **map_kwargs,
            )

            # Tokenize
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"
            
            def tokenize_fn(examples):
                tokenizer = processing_class if isinstance(processing_class, PreTrainedTokenizerBase) else processing_class.tokenizer
                
                # Tokenize prompt and completion together
                full_text = [p + c for p, c in zip(examples["prompt"], examples["completion"], strict=False)]
                encoded = tokenizer(full_text, add_special_tokens=False)
                
                # Tokenize prompts separately to know where they end
                prompt_encoded = tokenizer(examples["prompt"], add_special_tokens=False)
                
                result = {
                    "input_ids": encoded["input_ids"],
                    "prompt_length": [len(ids) for ids in prompt_encoded["input_ids"]],
                }
                
                # Handle rewards (per-token or per-sequence)
                if "rewards" in examples:
                    result["rewards"] = examples["rewards"]
                elif "reward" in examples:
                    # Convert per-sequence reward to per-token
                    result["rewards"] = [
                        [0.0] * prompt_len + [reward / (len(input_ids) - prompt_len)] * (len(input_ids) - prompt_len)
                        for input_ids, prompt_len, reward in zip(encoded["input_ids"], result["prompt_length"], examples["reward"], strict=False)
                    ]
                
                return result

            dataset = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "rewards", "prompt_length"]],
                **map_kwargs,
            )

        return dataset

    def compute_q_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_ref_model: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q-values, log probabilities, and hidden states.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            is_ref_model: Whether to use reference model
        
        Returns:
            Tuple of (q_values, log_probs, hidden_states)
        """
        model = self.ref_model if is_ref_model else self.model
        
        with torch.no_grad() if is_ref_model else nullcontext():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        logits = outputs.logits[:, :-1]  # Shift for next token prediction
        hidden_states = outputs.hidden_states[-1][:, :-1]  # Last layer hidden states
        
        # Get log probabilities for the actual next tokens
        labels = input_ids[:, 1:]  # Shift right
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        
        # Compute value estimates
        if not is_ref_model:
            value_estimates = self.value_model(hidden_states)
        else:
            value_estimates = torch.zeros_like(token_log_probs)
        
        # Q(s,a) = β * log π(a|s) + V(s)
        q_values = self.beta * token_log_probs + value_estimates
        
        return q_values, token_log_probs, hidden_states

    def compute_lambda_return(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute λ-return for variance reduction.
        
        Args:
            rewards: Immediate rewards of shape (batch_size, seq_len)
            values: Value estimates of shape (batch_size, seq_len)
            policy_log_probs: Log probs from policy of shape (batch_size, seq_len)
            ref_log_probs: Log probs from reference of shape (batch_size, seq_len)
            attention_mask: Mask of shape (batch_size, seq_len)
        
        Returns:
            Lambda returns of shape (batch_size, seq_len)
        """
        batch_size, seq_len = rewards.shape
        
        # Compute KL-regularized rewards
        kl_rewards = rewards - self.beta * (policy_log_probs - ref_log_probs)
        
        # Initialize returns
        lambda_returns = torch.zeros_like(values)
        
        # Compute λ-return backward through time
        if self.lambda_return == 1.0:
            # Monte Carlo return (sum all future rewards)
            for t in range(seq_len - 1, -1, -1):
                if t == seq_len - 1:
                    lambda_returns[:, t] = kl_rewards[:, t]
                else:
                    lambda_returns[:, t] = kl_rewards[:, t] + lambda_returns[:, t + 1]
        else:
            # λ-return (exponentially weighted mixture)
            for t in range(seq_len - 1, -1, -1):
                if t == seq_len - 1:
                    lambda_returns[:, t] = kl_rewards[:, t]
                else:
                    n_step_return = kl_rewards[:, t] + values[:, t + 1]
                    lambda_returns[:, t] = (
                        (1 - self.lambda_return) * n_step_return
                        + self.lambda_return * (kl_rewards[:, t] + lambda_returns[:, t + 1])
                    )
        
        # Apply attention mask
        lambda_returns = lambda_returns * attention_mask
        
        return lambda_returns

    def compute_importance_weights(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance sampling weights for offline data.
        
        Args:
            policy_log_probs: Log probs from current policy (batch_size, seq_len)
            ref_log_probs: Log probs from behavior policy (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Importance weights of shape (batch_size,)
        """
        # Compute log importance ratios
        log_ratios = (policy_log_probs - ref_log_probs) * attention_mask
        log_ratio = log_ratios.sum(dim=1)  # Sum over sequence
        
        # Compute importance weights and clip
        importance_weights = torch.exp(log_ratio)
        importance_weights = torch.clamp(
            importance_weights,
            min=1.0 / self.importance_sampling_clip,
            max=self.importance_sampling_clip,
        )
        
        return importance_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute DQO loss."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        rewards = inputs.get("rewards", torch.zeros_like(input_ids, dtype=torch.float32))
        
        # Truncate if needed
        if self.max_length is not None and input_ids.size(1) > self.max_length:
            if self.truncation_mode == "keep_end":
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                rewards = rewards[:, -self.max_length:]
            else:  # keep_start
                input_ids = input_ids[:, :self.max_length]
                attention_mask = attention_mask[:, :self.max_length]
                rewards = rewards[:, :self.max_length]
        
        # Flush left to remove padding
        attention_mask, input_ids, rewards = flush_left(attention_mask, input_ids, rewards)
        
        # Get Q-values and log probs from policy
        q_values, policy_log_probs, hidden_states = self.compute_q_values(
            input_ids, attention_mask, is_ref_model=False
        )
        
        # Get log probs from reference model
        with torch.no_grad():
            _, ref_log_probs, _ = self.compute_q_values(
                input_ids, attention_mask, is_ref_model=True
            )
        
        # Compute value estimates
        value_estimates = self.value_model(hidden_states)
        
        # Prepare for next step (shift by 1)
        next_value_estimates = torch.cat(
            [value_estimates[:, 1:], torch.zeros_like(value_estimates[:, :1])], dim=1
        )
        
        # Adjust masks (we don't have values for the last token's next state)
        seq_attention_mask = attention_mask[:, :-1]
        seq_rewards = rewards[:, :-1]
        seq_policy_log_probs = policy_log_probs
        seq_ref_log_probs = ref_log_probs
        
        # Compute λ-return targets
        lambda_returns = self.compute_lambda_return(
            seq_rewards,
            next_value_estimates,
            seq_policy_log_probs,
            seq_ref_log_probs,
            seq_attention_mask,
        )
        
        # Compute importance weights
        importance_weights = self.compute_importance_weights(
            seq_policy_log_probs,
            seq_ref_log_probs,
            seq_attention_mask,
        )
        
        # Value function loss: L_V = E[(V(s) - G^λ(s))^2]
        value_targets = lambda_returns.detach()
        value_loss = F.mse_loss(
            value_estimates * seq_attention_mask,
            value_targets * seq_attention_mask,
            reduction="none",
        )
        value_loss = (value_loss.sum(dim=1) / seq_attention_mask.sum(dim=1).clamp(min=1)).mean()
        
        # Q-function loss: L_π = E[(Q(s,a) - (r + G^λ(s')))^2]
        # Q(s,a) = β * log π(a|s) + V(s)
        q_targets = seq_rewards + next_value_estimates
        q_targets = q_targets.detach()
        
        q_loss = F.mse_loss(
            q_values * seq_attention_mask,
            q_targets * seq_attention_mask,
            reduction="none",
        )
        
        # Weight by importance sampling and average
        q_loss = q_loss.sum(dim=1) / seq_attention_mask.sum(dim=1).clamp(min=1)
        q_loss = (q_loss * importance_weights).mean()
        
        # Total loss
        loss = q_loss + value_loss
        
        # Update value network separately
        self.value_optimizer.zero_grad()
        self.accelerator.backward(value_loss, retain_graph=True)
        self.value_optimizer.step()
        
        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "q_loss": q_loss.item(),
            "value_loss": value_loss.item(),
            "mean_q_value": (q_values * seq_attention_mask).sum().item() / seq_attention_mask.sum().item(),
            "mean_value": (value_estimates * seq_attention_mask).sum().item() / seq_attention_mask.sum().item(),
            "mean_reward": (seq_rewards * seq_attention_mask).sum().item() / seq_attention_mask.sum().item(),
            "mean_importance_weight": importance_weights.mean().item(),
        }
        
        self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return loss, metrics
        return loss

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        """Store metrics for logging."""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log metrics with stored averages."""
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            if metrics:
                logs[key] = sum(metrics) / len(metrics)
        self._stored_metrics[train_eval].clear()
        return super().log(logs, start_time)

    def _save_checkpoint(self, model, trial):
        """Save checkpoint including value network."""
        # Save value network state dict
        value_model_path = self.args.output_dir + "/value_model.pt"
        torch.save(self.value_model.state_dict(), value_model_path)
        super()._save_checkpoint(model, trial)
