#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Example script for training a model with Direct Q-function Optimization (DQO).

DQO formulates response generation as a token-level MDP and uses soft actor-critic
framework to optimize Q-functions. This makes it particularly suitable for multi-step
reasoning tasks like math problem solving.

Usage:
    python examples/scripts/dqo.py \
        --model_name_or_path facebook/opt-350m \
        --dataset_name trl-lib/tldr \
        --output_dir dqo-model-tldr \
        --per_device_train_batch_size 4 \
        --num_train_epochs 1
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import DQOConfig, DQOTrainer


@dataclass
class ScriptArguments:
    """
    Arguments for the DQO training script.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(
        default="trl-lib/tldr",
        metadata={"help": "Dataset name. Should contain 'prompt', 'completion', and optionally 'reward' fields."},
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training"})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation"})


def prepare_dataset(examples):
    """
    Prepare dataset by ensuring it has the required format.
    
    DQO expects:
    - 'prompt': Input text
    - 'completion': Output text  
    - 'reward' (optional): Scalar reward for the completion
    - 'rewards' (optional): Per-token rewards
    """
    # If dataset has 'query' and 'response' instead of 'prompt' and 'completion'
    if "query" in examples and "prompt" not in examples:
        examples["prompt"] = examples["query"]
    if "response" in examples and "completion" not in examples:
        examples["completion"] = examples["response"]
    
    # Add dummy rewards if not present (will use sparse terminal rewards)
    if "reward" not in examples and "rewards" not in examples:
        # Assign positive reward to all examples (you should replace this with actual rewards)
        examples["reward"] = [1.0] * len(examples["prompt"])
    
    return examples


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, DQOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    dataset = load_dataset(script_args.dataset_name)
    
    # Prepare dataset format
    train_dataset = dataset[script_args.dataset_train_split].map(
        prepare_dataset,
        batched=True,
        desc="Preparing train dataset",
    )
    
    if script_args.dataset_test_split in dataset:
        eval_dataset = dataset[script_args.dataset_test_split].map(
            prepare_dataset,
            batched=True,
            desc="Preparing eval dataset",
        )
    else:
        eval_dataset = None

    # Initialize DQO trainer
    trainer = DQOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print(f"Model saved to {training_args.output_dir}")
