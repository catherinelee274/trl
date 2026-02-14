# Q# (Q-Sharp): Value-Based RL for LLM Post-Training

## Overview

Q# (pronounced "Q-sharp") is a value-based reinforcement learning algorithm for LLM post-training that uses distributional RL to learn optimal Q or V functions for KL-regularized RL problems. Unlike policy-based methods like PPO and DPO, Q# guides the reference policy using learned value functions, offering both improved performance and theoretical guarantees.

## Key Features

- **Value-Based Approach**: Learns Q or V functions to guide policy instead of directly optimizing the policy
- **Distributional RL**: Uses distributional RL (via maximum likelihood estimation) for variance-dependent convergence
- **Provably Optimal**: Theoretically principled with convergence guarantees for deterministic MDPs
- **KL-Regularized**: Maintains controlled divergence from reference policy
- **Flexible Inference**: Supports multiple inference modes (expectation, Bernoulli sampling)

## Quick Start

<function_calls>

Here's a basic example of training a Q# classifier:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import QSharpConfig
from trl.models import QSharpClassifier

# Load reference model and tokenizer
ref_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
classifier_model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id)

# Create Q# configuration
config = QSharpConfig(
    output_dir="./qsharp_checkpoints",
    classifier_type="Q",  # or "V" for state-value function
    loss_type="bce",  # or "mse" or "mle" for distributional RL
    init_mode="reuse",  # reuse LM head weights
    inference_mode="bernoulli",
    eta=1.0,  # KL regularization strength
    num_train_epochs=5,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
)

# Load classifier model
from transformers import AutoConfig
classifier_config = AutoConfig.from_pretrained(classifier_model_id)
classifier_config.num_labels = len(tokenizer)

classifier = QSharpClassifier(
    classifier_config,
    loss_type=config.loss_type,
    use_bias=config.use_bias,
    classifier_type=config.classifier_type,
    num_atoms=config.num_atoms,
    V_min=config.V_min,
    V_max=config.V_max,
)
```

## Using Q# for Guided Generation

Once you have a trained Q# classifier, you can use it to guide generation:

```python
from trl.models import QSharpLogitProcessor
from transformers.generation import LogitsProcessorList

# Create logit processor
logit_processor = QSharpLogitProcessor(
    eta=10.0,  # Guidance strength
    ref_model=ref_model,
    ref_model_tokenizer=tokenizer,
    value_classifier=classifier,
    inference_mode="bernoulli",
    top_k=-1,  # Modify all logits
    use_cache=True,
)

# Generate with guidance
prompt = "Solve the following math problem: What is 25 * 4?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

logit_processor.reset_classifier_state()
logit_processors = LogitsProcessorList([logit_processor])

outputs = ref_model.generate(
    **inputs,
    logits_processor=logit_processors,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Configuration Options

### Classifier Configuration

- `classifier_type`: Type of classifier ("Q" for action-value, "V" for state-value)
- `loss_type`: Loss function ("mse", "bce", or "mle" for distributional RL)
- `use_bias`: Whether to use bias in classification layer (default: False)

### Distributional RL (for "mle" loss)

- `num_atoms`: Number of support points (default: 11)
- `V_min`: Minimum value (default: 0.0)
- `V_max`: Maximum value (default: 1.0)

### Initialization

- `init_mode`: 
  - `"zero"`: Zero initialization
  - `"random"`: Random initialization
  - `"reuse"`: Reuse LM head weights (recommended)
  - `"warmstart"`: Load from checkpoint

### Inference

- `inference_mode`: 
  - `"bernoulli"`: Bernoulli sampling (recommended)
  - `"expectation"`: Expectation-based guidance
  - `"disabled"`: No guidance
- `eta`: KL regularization strength (larger = less KL divergence)
- `top_k`: Number of top logits to modify (-1 for all)
- `cd_baseline`: Use Contrastive Decoding baseline

### Data Processing

- `shift_reward`: Shift rewards by constant (subtraction)
- `scale_reward`: Scale rewards by constant (multiplication)
- `drop_no_variation`: Drop samples with no reward variation
- `use_all_ref_tokens`: Use all reference tokens (0/1/2)
- `max_token_length`: Maximum sequence length (-1 for no limit)
- `max_batch_num_tokens`: Enable dynamic batching (-1 for static)

## Mathematical Background

Q# learns the optimal Q-function $Q^*(s, a)$ or V-function $V^*(s)$ for the KL-regularized RL objective:

$$\max_\pi \mathbb{E}_{s,a \sim \pi}[R(s,a)] - \beta \cdot D_{KL}(\pi || \pi_{ref})$$

where:
- $\pi$ is the policy being optimized
- $\pi_{ref}$ is the reference policy
- $R(s,a)$ is the reward function
- $\beta$ is the KL penalty coefficient (related to `eta` by $\eta = 1/\beta$)

During inference, the policy is guided using:

$$\pi(a|s) \propto \pi_{ref}(a|s) \cdot \exp(\eta \cdot Q^*(s,a))$$

For distributional RL with "mle" loss, Q# learns the full return distribution rather than just the expected value, enabling variance-dependent convergence rates.

## Comparison with Other Methods

| Method | Type | Advantages | Disadvantages |
|--------|------|-----------|---------------|
| **Q#** | Value-based | Provably optimal, variance-aware, stable | Requires separate classifier training |
| **PPO** | Policy-based | Direct policy optimization | Can inherit shortcuts, less stable |
| **DPO** | Policy-based | Simple, no reward model | May not fix pre-training shortcuts |
| **GRPO** | Policy-based | Group-based robustness | Still policy-based limitations |

## Examples

See `examples/scripts/qsharp_training.py` for a complete training example on math reasoning tasks.

## Citation

If you use Q# in your research, please cite:

```bibtex
@article{zhou2025qsharp,
  title={Q\#: Provably Optimal Distributional RL for LLM Post-Training},
  author={Zhou, Jin Peng and Wang, Kaiwen and Chang, Jonathan and Gao, Zhaolin and Kallus, Nathan and Weinberger, Kilian Q. and Brantley, Kianté and Sun, Wen},
  journal={arXiv preprint arXiv:2502.20548},
  year={2025}
}
```

## References

- Paper: [Q#: Provably Optimal Distributional RL for LLM Post-Training](https://arxiv.org/abs/2502.20548)
- Original Implementation: [github.com/jinpz/q_sharp](https://github.com/jinpz/q_sharp)
