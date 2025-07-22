#!/usr/bin/env python3
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

"""
Example script for training a model using Direct Q-function Optimization (DQO).

DQO formulates the response generation process as a Markov Decision Process (MDP) 
and utilizes the soft actor-critic (SAC) framework to optimize a Q-function directly 
parameterized by the language model. This enables more effective process supervision 
for multi-step reasoning tasks.

Key features of DQO:
- MDP formulation for better multi-step reasoning
- λ-return for bias mitigation  
- Importance sampling for offline data reweighting
- Direct Q-function parameterization by the policy

Usage:
    python examples/scripts/dqo.py \
        --model_name_or_path=gpt2 \
        --dataset_name=Anthropic/hh-rlhf \
        --output_dir=./dqo_model
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import DQOConfig, DQOTrainer


# Enable logging
logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptArguments:
    """
    Arguments for the DQO training script.
    """

    model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Path to the model or model identifier from huggingface.co/models."},
    )
    dataset_name: str = field(
        default="Anthropic/hh-rlhf",
        metadata={"help": "Name of the dataset to use for training."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name of the dataset."},
    )
    output_dir: str = field(
        default="./dqo_model",
        metadata={"help": "Directory to save the trained model."},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for training."},
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Maximum length of the prompt sequence."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "Maximum length of the completion sequence."},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "KL regularization coefficient."},
    )
    gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor for future rewards."},
    )
    tau: float = field(
        default=0.005,
        metadata={"help": "Soft update coefficient for target network."},
    )
    q_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for Q-network optimization."},
    )
    target_update_freq: int = field(
        default=100,
        metadata={"help": "Frequency of target network updates."},
    )
    q_network_hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size for Q-network."},
    )
    q_network_num_layers: int = field(
        default=2,
        metadata={"help": "Number of layers in Q-network."},
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, script_args.dataset_config)
    
    # For demonstration, we'll use a small subset
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = dataset["test"].select(range(100)) if "test" in dataset else None

    # Configure DQO training
    training_args = DQOConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        num_train_epochs=1,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        max_completion_length=script_args.max_completion_length,
        beta=script_args.beta,
        gamma=script_args.gamma,
        tau=script_args.tau,
        q_learning_rate=script_args.q_learning_rate,
        target_update_freq=script_args.target_update_freq,
        q_network_hidden_size=script_args.q_network_hidden_size,
        q_network_num_layers=script_args.q_network_num_layers,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        warmup_steps=100,
        report_to=None,  # Disable wandb/tensorboard for this example
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )

    # Initialize DQO trainer
    trainer = DQOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train the model
    print("Starting DQO training...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    print(f"Model saved to {script_args.output_dir}")

    # Example of how to use the trained model for inference
    print("\nTesting the trained model...")
    test_prompt = "What is the capital of France?"
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response[len(test_prompt):].strip()}")


if __name__ == "__main__":
    main()
