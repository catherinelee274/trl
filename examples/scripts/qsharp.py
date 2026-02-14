"""
Example script for training Q# classifier for math reasoning.

This script demonstrates how to use Q# (Q-Sharp) for value-based RL in LLM post-training.
Q# learns Q or V functions using distributional RL to guide the reference policy.

Reference: https://arxiv.org/abs/2502.20548
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from trl import QSharpConfig
from trl.models import QSharpClassifier, QSharpLogitProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q# classifier for math reasoning")
    
    # Model arguments
    parser.add_argument(
        "--ref_model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Path to reference model",
    )
    parser.add_argument(
        "--classifier_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path to classifier model",
    )
    
    # Q# configuration
    parser.add_argument("--classifier_type", type=str, default="Q", choices=["Q", "V"])
    parser.add_argument("--loss_type", type=str, default="bce", choices=["mse", "bce", "mle"])
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--init_mode", type=str, default="reuse", choices=["zero", "random", "reuse", "warmstart"])
    parser.add_argument("--inference_mode", type=str, default="bernoulli", choices=["expectation", "bernoulli", "disabled"])
    
    # Distributional RL (for mle loss)
    parser.add_argument("--num_atoms", type=int, default=11)
    parser.add_argument("--V_min", type=float, default=0.0)
    parser.add_argument("--V_max", type=float, default=1.0)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./qsharp_checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_token_length", type=int, default=-1)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--ckpt_freq", type=int, default=500)
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--drop_no_variation", action="store_true")
    parser.add_argument("--id_eval_ratio", type=float, default=0.1)
    
    # Inference arguments
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--cd_baseline", action="store_true")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    
    return parser.parse_args()


def prepare_data(dataset, tokenizer, args):
    """
    Prepare data for Q# training.
    This is a simplified example - in practice, you would need to:
    1. Generate rollouts from the reference model
    2. Compute rewards (e.g., correctness of answers)
    3. Format data with prompts, responses, and rewards
    """
    print("Preparing data for Q# training...")
    
    # This is a placeholder - real implementation would involve:
    # - Collecting rollouts from reference model
    # - Computing rewards based on correctness
    # - Creating training samples with (prompt, response, reward) tuples
    
    train_data = []
    eval_data = []
    
    for split, target_list in [("train", train_data), ("test", eval_data)]:
        for example in dataset[split]:
            # Extract question and answer
            question = example.get("question", "")
            answer = example.get("answer", "")
            
            # In practice, you would:
            # 1. Generate multiple responses from the model
            # 2. Compute rewards for each response
            # 3. Store tokenized prompts, responses, and rewards
            
            # Placeholder data structure
            sample = {
                "question": question,
                "answer": answer,
                "prompt_tokens": [],  # Tokenized prompt
                "response_tokens": [],  # Tokenized responses
                "rewards": [],  # Computed rewards
            }
            target_list.append(sample)
    
    print(f"Prepared {len(train_data)} training samples and {len(eval_data)} eval samples")
    return train_data, eval_data


def train_classifier(model, train_data, eval_data, config, accelerator):
    """
    Train the Q# classifier.
    This is a simplified training loop - in practice, you would use
    the full training infrastructure from the original implementation.
    """
    print("Training Q# classifier...")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Prepare with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Training loop (simplified)
    model.train()
    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
        
        # In practice, you would:
        # 1. Create batches from train_data
        # 2. Forward pass through classifier
        # 3. Compute loss
        # 4. Backward pass and optimize
        # 5. Evaluate periodically
        
        # Placeholder for actual training code
        pass
    
    print("Training completed!")
    return model


def evaluate_with_guidance(ref_model, classifier, tokenizer, test_data, args):
    """
    Evaluate the reference model with Q# guidance.
    """
    print("Evaluating with Q# guidance...")
    
    # Create logit processor
    logit_processor = QSharpLogitProcessor(
        eta=args.eta,
        ref_model=ref_model,
        ref_model_tokenizer=tokenizer,
        value_classifier=classifier,
        inference_mode=args.inference_mode,
        top_k=args.top_k,
        cd_baseline=1 if args.cd_baseline else 0,
        use_cache=True,
    )
    
    results = []
    
    for example in test_data[:10]:  # Evaluate on first 10 examples
        question = example.get("question", "")
        
        # Tokenize prompt
        inputs = tokenizer(question, return_tensors="pt").to(ref_model.device)
        
        # Reset classifier state for each generation
        logit_processor.reset_classifier_state()
        
        # Generate with guidance
        from transformers.generation import LogitsProcessorList
        logit_processors = LogitsProcessorList([logit_processor])
        
        outputs = ref_model.generate(
            **inputs,
            logits_processor=logit_processors,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "question": question,
            "generated_answer": generated_text,
        })
        
        print(f"\nQ: {question}")
        print(f"A: {generated_text}\n")
    
    return results


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load tokenizer and models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=accelerator.device,
    )
    
    # Load classifier config and model
    classifier_config = AutoConfig.from_pretrained(args.classifier_model_name_or_path)
    classifier_config.num_labels = len(tokenizer)
    
    classifier = QSharpClassifier(
        classifier_config,
        loss_type=args.loss_type,
        use_bias=args.use_bias,
        classifier_type=args.classifier_type,
        num_atoms=args.num_atoms,
        V_min=args.V_min,
        V_max=args.V_max,
    )
    
    # Initialize classifier weights if needed
    if args.init_mode == "reuse":
        print("Reusing LM head weights for classifier initialization...")
        temp_model = AutoModelForCausalLM.from_pretrained(
            args.classifier_model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        lm_head_weight = temp_model.lm_head.weight.data
        if args.loss_type == "mle":
            vocab_size = lm_head_weight.shape[0]
            classifier.score.weight.data = lm_head_weight.repeat(1, args.num_atoms).view(
                vocab_size * args.num_atoms, -1
            )
        else:
            classifier.score.weight.data = lm_head_weight
        del temp_model
    elif args.init_mode == "zero":
        classifier.zero_init_classifier()
    
    classifier = classifier.to(accelerator.device)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    
    # Prepare data
    train_data, eval_data = prepare_data(dataset, tokenizer, args)
    
    # Create Q# config
    config = QSharpConfig(
        output_dir=args.output_dir,
        classifier_type=args.classifier_type,
        loss_type=args.loss_type,
        use_bias=args.use_bias,
        init_mode=args.init_mode,
        inference_mode=args.inference_mode,
        num_atoms=args.num_atoms,
        V_min=args.V_min,
        V_max=args.V_max,
        eta=args.eta,
        top_k=args.top_k,
        cd_baseline=args.cd_baseline,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_token_length=args.max_token_length,
        drop_no_variation=args.drop_no_variation,
        id_eval_ratio=args.id_eval_ratio,
        eval_freq=args.eval_freq,
        ckpt_freq=args.ckpt_freq,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.ckpt_freq,
        report_to="wandb" if args.use_wandb else "none",
    )
    
    # Train classifier
    classifier = train_classifier(classifier, train_data, eval_data, config, accelerator)
    
    # Save final model
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        classifier.save_pretrained(os.path.join(args.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
        
        # Save config
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # Evaluate with guidance
    results = evaluate_with_guidance(ref_model, classifier, tokenizer, eval_data, args)
    
    # Save results
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
