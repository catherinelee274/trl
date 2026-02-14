# DQO Trainer

TRL supports the Direct Q-function Optimization (DQO) algorithm from the paper [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://huggingface.co/papers/2410.09302) by Ji et al., 2024. DQO formulates response generation as a token-level Markov Decision Process (MDP) and uses the soft actor-critic framework to optimize a Q-function directly parameterized by the language model.

## Key Features

- **Multi-step reasoning**: Unlike bandit-based methods (DPO, DRO), DQO models language generation as an MDP, making it more suitable for tasks requiring long chains of thought.
- **Process supervision**: Can leverage step-by-step reward signals to provide stronger supervision signals during training.
- **Offline learning**: Trains on pre-collected data without requiring online sampling.
- **λ-return**: Uses λ-return for variance reduction in value estimation.
- **Importance sampling**: Reweights offline data to account for distribution mismatch between behavior and target policies.

## Quick Start

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DQOConfig, DQOTrainer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Load dataset - should contain 'prompt', 'completion', and optionally 'reward' or 'rewards'
dataset = load_dataset("your-dataset")

# Configure trainer
training_args = DQOConfig(
    output_dir="dqo-model",
    beta=0.03,  # KL regularization coefficient
    lambda_return=1.0,  # λ for λ-return (1.0 = Monte Carlo)
    learning_rate=5e-7,
    value_learning_rate=1e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,
)

# Initialize trainer
trainer = DQOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)

# Train
trainer.train()
```

## Expected Dataset Format

DQO expects datasets with the following fields:

### Required Fields
- `prompt`: The input prompt/question
- `completion`: The model's response/completion

### Optional Fields
- `reward`: A single scalar reward for the entire completion (will be distributed across tokens)
- `rewards`: A list of per-token rewards matching the length of the completion

### Example

```python
{
    "prompt": "What is 2+2?",
    "completion": "Let me think step by step. 2+2 equals 4.",
    "reward": 1.0  # Correct answer
}
```

For process supervision with per-step rewards:

```python
{
    "prompt": "Solve: 2x + 3 = 7",
    "completion": "Step 1: Subtract 3 from both sides\nStep 2: 2x = 4\nStep 3: Divide by 2\nStep 4: x = 2",
    "rewards": [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25]  # Per-token rewards
}
```

## DQOConfig

[[autodoc]] DQOConfig

## DQOTrainer

[[autodoc]] DQOTrainer

## Algorithm Details

### Token-Level MDP Formulation

DQO treats text generation as a token-level MDP where:
- **State** s_t: All tokens generated so far (prompt + partial completion)
- **Action** a_t: The next token to generate
- **Reward** r_t: Immediate reward (can be sparse or dense)
- **Transition**: Deterministic (next state = current state + new token)

### Q-Function Parameterization

The Q-function is directly parameterized by the policy:

```
Q(s, a) = β * log π(a|s) + V(s)
```

where:
- π is the policy (language model)
- V(s) is a learned value function
- β is the KL regularization coefficient

### Loss Functions

**Value Function Loss:**
```
L_V = E[(V(s) - G^λ(s))^2]
```

**Policy Loss (Q-function):**
```
L_π = E[w(τ) * (Q(s,a) - (r + G^λ(s')))^2]
```

where:
- G^λ(s) is the λ-return target
- w(τ) is the importance sampling weight

### λ-Return

The λ-return provides a trade-off between bias and variance:
- λ=1.0: Monte Carlo return (high variance, low bias)
- λ=0.0: One-step TD (low variance, high bias)

```
G^λ(s_t) = (1-λ) Σ_n λ^(n-1) G^(n)(s_t)
```

## Comparison with Other Methods

| Method | Formulation | Process Rewards | Pairwise Data | Online Sampling |
|--------|-------------|-----------------|---------------|-----------------|
| DPO | Bandit | ✗ | ✓ | ✗ |
| DRO | Bandit | ✗ | ✗ | ✗ |
| PPO | MDP | ✓ | ✗ | ✓ |
| **DQO** | **MDP** | **✓** | **✗** | **✗** |

DQO combines the benefits of offline learning (like DPO/DRO) with the multi-step reasoning capabilities of MDP-based methods (like PPO).

## Tips and Best Practices

1. **β (beta) selection**: 
   - Smaller values (0.01-0.03) for creative tasks
   - Larger values (0.1-0.5) when you want to stay close to the reference model

2. **λ (lambda_return)**:
   - Start with λ=1.0 (Monte Carlo) for maximum use of trajectory information
   - Use λ<1.0 if training is unstable or for shorter sequences

3. **Learning rates**:
   - Policy: 5e-7 to 1e-6 (smaller than typical fine-tuning)
   - Value: 1e-5 to 5e-5 (higher than policy)

4. **Process rewards**:
   - If available, provide per-token rewards for better process supervision
   - Can significantly improve performance on multi-step reasoning tasks

5. **Dataset size**:
   - DQO benefits from diverse trajectories
   - Include both positive and negative examples

## Citation

```bibtex
@article{ji2024dqo,
    title={Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization},
    author={Kaixuan Ji and Guanlin Liu and Ning Dai and Qingping Yang and Renjie Zheng and Zheng Wu and Chen Dun and Quanquan Gu and Lin Yan},
    journal={arXiv preprint arXiv:2410.09302},
    year={2024}
}
```
