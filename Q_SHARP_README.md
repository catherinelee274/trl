# Q# (Q-Sharp) Implementation in TRL

This document describes the implementation of Q# (Q-Sharp), a value-based reinforcement learning algorithm for LLM post-training, integrated into the TRL library.

## Overview

Q# is a provably optimal distributional RL algorithm for LLM post-training that learns Q or V functions to guide the reference policy. Unlike policy-based methods (PPO, DPO), Q# uses value-based guidance with theoretical guarantees.

**Paper**: [Q#: Provably Optimal Distributional RL for LLM Post-Training](https://arxiv.org/abs/2502.20548)  
**Original Implementation**: [github.com/jinpz/q_sharp](https://github.com/jinpz/q_sharp)

## Implementation Structure

### Files Added

1. **`trl/models/qsharp_model.py`**
   - `QSharpClassifier`: Main classifier model for learning Q or V functions
   - `QSharpLogitProcessor`: Logit processor for guided generation
   - Supports both Q-functions (action-value) and V-functions (state-value)
   - Implements distributional RL with multiple loss types (MSE, BCE, MLE)

2. **`trl/trainer/qsharp_config.py`**
   - `QSharpConfig`: Configuration class for Q# training
   - Comprehensive parameters for classifier setup, training, and inference
   - Validation logic for parameter combinations

3. **`docs/source/qsharp.md`**
   - Complete documentation for Q# usage
   - Mathematical background and theoretical foundation
   - Configuration options and examples
   - Comparison with other methods (PPO, DPO, GRPO)

4. **`examples/scripts/qsharp.py`**
   - Example training script for math reasoning tasks
   - Demonstrates data preparation, training, and evaluation
   - Can be extended for custom tasks

### Files Modified

1. **`trl/models/__init__.py`**
   - Added exports for `QSharpClassifier` and `QSharpLogitProcessor`

2. **`trl/trainer/__init__.py`**
   - Added export for `QSharpConfig`

## Key Features

### 1. **Flexible Classifier Types**
- **Q-Classifier**: Learns action-value functions Q(s, a)
- **V-Classifier**: Learns state-value functions V(s)

### 2. **Multiple Loss Functions**
- **MSE**: Mean squared error for regression
- **BCE**: Binary cross-entropy for binary rewards
- **MLE**: Maximum likelihood for distributional RL (learns full return distribution)

### 3. **Distributional RL**
When using `loss_type="mle"`, Q# learns the full distribution of returns rather than just the expected value:
- Configurable number of atoms (support points)
- Adjustable value range [V_min, V_max]
- Variance-dependent convergence rates

### 4. **Flexible Initialization**
- **zero**: Zero initialization
- **random**: Random initialization
- **reuse**: Reuse LM head weights (recommended)
- **warmstart**: Load from checkpoint

### 5. **Multiple Inference Modes**
- **bernoulli**: Bernoulli sampling (recommended)
- **expectation**: Expectation-based guidance
- **disabled**: No guidance (baseline)

### 6. **KL Regularization**
- Controlled divergence from reference policy via `eta` parameter
- Larger `eta` → smaller KL divergence
- Theoretically grounded KL-regularized RL

## Usage Example

### Basic Training Setup

```python
from transformers import AutoTokenizer, AutoConfig
from trl import QSharpConfig
from trl.models import QSharpClassifier

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Create configuration
config = QSharpConfig(
    output_dir="./qsharp_checkpoints",
    classifier_type="Q",
    loss_type="bce",
    init_mode="reuse",
    inference_mode="bernoulli",
    eta=1.0,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
)

# Create classifier
classifier_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
classifier_config.num_labels = len(tokenizer)

classifier = QSharpClassifier(
    classifier_config,
    loss_type=config.loss_type,
    use_bias=config.use_bias,
    classifier_type=config.classifier_type,
)
```

### Guided Generation

```python
from trl.models import QSharpLogitProcessor
from transformers.generation import LogitsProcessorList

# Create logit processor
logit_processor = QSharpLogitProcessor(
    eta=10.0,
    ref_model=ref_model,
    ref_model_tokenizer=tokenizer,
    value_classifier=classifier,
    inference_mode="bernoulli",
    top_k=-1,
    use_cache=True,
)

# Generate with guidance
prompt = "What is 25 * 4?"
inputs = tokenizer(prompt, return_tensors="pt")

logit_processor.reset_classifier_state()
logit_processors = LogitsProcessorList([logit_processor])

outputs = ref_model.generate(
    **inputs,
    logits_processor=logit_processors,
    max_new_tokens=512,
)
```

## Mathematical Foundation

Q# solves the KL-regularized RL problem:

$$\max_\pi \mathbb{E}_{s,a \sim \pi}[R(s,a)] - \beta \cdot D_{KL}(\pi || \pi_{ref})$$

The optimal policy is given by:

$$\pi^*(a|s) \propto \pi_{ref}(a|s) \cdot \exp(\eta \cdot Q^*(s,a))$$

where $\eta = 1/\beta$ is the reciprocal of the KL penalty coefficient.

### Distributional RL (MLE Loss)

For distributional RL, Q# learns the full return distribution $Z(s,a)$ represented as a categorical distribution over atoms:

$$Z(s,a) = \sum_{i=1}^{N} p_i \delta_{z_i}$$

This provides:
- **Variance estimation**: Know uncertainty in value estimates
- **Faster convergence**: When reference policy has low variance
- **Better exploration**: Use distribution information for guidance

## Advantages Over Policy-Based Methods

| Aspect | Q# | PPO/DPO |
|--------|-----|---------|
| **Theoretical Guarantees** | Provably optimal with convergence bounds | Heuristic, no guarantees |
| **Stability** | Value function learning is stable | Can be unstable, sensitive to hyperparameters |
| **Shortcuts** | Can fix pre-training shortcuts | May inherit shortcuts |
| **Variance** | Variance-aware (with MLE loss) | Ignores variance |
| **KL Control** | Direct control via eta | Indirect control |

## Integration with TRL Ecosystem

Q# integrates seamlessly with TRL's infrastructure:

- Uses standard `TrainingArguments` as base config
- Compatible with `Accelerate` for distributed training
- Supports Weights & Biases logging
- Works with HuggingFace models and tokenizers
- Can be used with PEFT/LoRA adapters

## Future Enhancements

Potential areas for extension:

1. **Full Trainer Class**: Implement a complete `QSharpTrainer` class with built-in data preparation and training loop
2. **Data Collection Utilities**: Tools for collecting rollouts and computing rewards
3. **Multi-Round Training**: Support for iterative training rounds
4. **Additional Model Architectures**: Support beyond Llama (GPT, Mistral, etc.)
5. **Reward Models**: Integration with reward model training
6. **Advanced Evaluation**: Built-in evaluation metrics for various tasks

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

- [Q# Paper](https://arxiv.org/abs/2502.20548)
- [Original Implementation](https://github.com/jinpz/q_sharp)
- [TRL Documentation](https://huggingface.co/docs/trl)

## License

This implementation follows the same license as TRL (Apache 2.0).

---

**Implementation Date**: February 2026  
**TRL Version**: Compatible with TRL 0.13+  
**Maintainer**: TRL Team
