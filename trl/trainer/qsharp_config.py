"""Configuration for Q# (Q-sharp) Trainer.

This module defines the training configuration for Q# algorithm.

Reference: https://arxiv.org/abs/2502.20548
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class QSharpConfig(TrainingArguments):
    """Configuration class for Q# Trainer.
    
    Q# is a value-based RL algorithm for LLM post-training that uses distributional RL
    to learn optimal Q or V functions for KL-regularized RL.
    
    Args:
        classifier_type: Type of classifier to train ("Q" or "V")
        loss_type: Type of loss function ("mse", "bce", or "mle")
        use_bias: Whether to use bias in the classification layer
        num_atoms: Number of atoms for distributional RL (only for "mle" loss)
        V_min: Minimum value for distributional RL (only for "mle" loss)
        V_max: Maximum value for distributional RL (only for "mle" loss)
        init_mode: Initialization mode ("zero", "random", "reuse", or "warmstart")
        inference_mode: Inference mode for guidance ("expectation", "bernoulli", or "disabled")
        eta: KL regularization strength (reciprocal of KL weight)
        top_k: Only modify top k logits during inference (-1 for all)
        cd_baseline: Whether to use CD baseline
        shift_reward: Shift rewards by this value (subtraction)
        scale_reward: Scale rewards by this value (multiplication)
        drop_no_variation: Whether to drop samples with no reward variation
        use_all_ref_tokens: Whether to use all tokens from reference model
        max_token_length: Maximum token length for training (-1 for no limit)
        id_eval_ratio: Ratio of training data to use for in-distribution eval
        max_batch_num_tokens: Maximum number of tokens per batch (-1 for no limit)
    """
    
    # Classifier configuration
    classifier_type: Literal["Q", "V"] = field(
        default="Q",
        metadata={"help": "Type of classifier to train: Q (action-value) or V (state-value)"}
    )
    loss_type: Literal["mse", "bce", "mle"] = field(
        default="bce",
        metadata={"help": "Loss function type: mse (mean squared error), bce (binary cross-entropy), or mle (maximum likelihood for distributional RL)"}
    )
    use_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in the classification layer"}
    )
    
    # Distributional RL parameters (for mle loss)
    num_atoms: int = field(
        default=11,
        metadata={"help": "Number of atoms for distributional RL (only used with mle loss)"}
    )
    V_min: float = field(
        default=0.0,
        metadata={"help": "Minimum value for distributional RL (only used with mle loss)"}
    )
    V_max: float = field(
        default=1.0,
        metadata={"help": "Maximum value for distributional RL (only used with mle loss)"}
    )
    
    # Initialization
    init_mode: Literal["zero", "random", "reuse", "warmstart"] = field(
        default="reuse",
        metadata={"help": "Initialization mode for classifier: zero (zero init), random (random init), reuse (reuse LM head), warmstart (load from checkpoint)"}
    )
    
    # Inference configuration
    inference_mode: Literal["expectation", "bernoulli", "disabled"] = field(
        default="bernoulli",
        metadata={"help": "Inference mode for guidance: expectation, bernoulli, or disabled"}
    )
    eta: Optional[float] = field(
        default=1.0,
        metadata={"help": "KL regularization strength (reciprocal of KL weight). Larger = smaller KL divergence"}
    )
    top_k: int = field(
        default=-1,
        metadata={"help": "Only modify top k logits during inference. -1 means modify all logits"}
    )
    cd_baseline: bool = field(
        default=False,
        metadata={"help": "Whether to use CD (Contrastive Decoding) baseline"}
    )
    
    # Reward processing
    shift_reward: float = field(
        default=0.0,
        metadata={"help": "Shift rewards by this value (subtraction)"}
    )
    scale_reward: float = field(
        default=1.0,
        metadata={"help": "Scale rewards by this value (multiplication)"}
    )
    
    # Data processing
    drop_no_variation: bool = field(
        default=True,
        metadata={"help": "Whether to drop training samples with no reward variation"}
    )
    use_all_ref_tokens: int = field(
        default=1,
        metadata={"help": "Whether to use all tokens from reference model (0: no, 1: yes, 2: everything including rollout)"}
    )
    max_token_length: int = field(
        default=-1,
        metadata={"help": "Maximum token length for training sequences. -1 for no limit"}
    )
    id_eval_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of training data to use for in-distribution evaluation"}
    )
    max_batch_num_tokens: int = field(
        default=-1,
        metadata={"help": "Maximum number of tokens per batch. -1 for no limit. Enables dynamic batching"}
    )
    
    # Model paths
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model (for tokenizer and generating rollouts)"}
    )
    classifier_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the classifier model (for initialization)"}
    )
    classifier_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to classifier checkpoint for warmstart initialization"}
    )
    
    # Training
    resume_opt_scheduler: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to resume optimizer and scheduler from checkpoint"}
    )
    save_opt_scheduler: bool = field(
        default=False,
        metadata={"help": "Whether to save optimizer and scheduler state"}
    )
    eval_freq: int = field(
        default=500,
        metadata={"help": "Evaluation frequency (in steps). -1 to disable"}
    )
    ckpt_freq: int = field(
        default=500,
        metadata={"help": "Checkpoint save frequency (in steps). -1 to disable"}
    )
    eval_max_size: int = field(
        default=1000,
        metadata={"help": "Maximum number of samples for evaluation. -1 for no limit"}
    )
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate classifier type
        if self.classifier_type not in ["Q", "V"]:
            raise ValueError(f"classifier_type must be 'Q' or 'V', got {self.classifier_type}")
        
        # Validate loss type
        if self.loss_type not in ["mse", "bce", "mle"]:
            raise ValueError(f"loss_type must be 'mse', 'bce', or 'mle', got {self.loss_type}")
        
        # Validate inference mode
        if self.inference_mode not in ["expectation", "bernoulli", "disabled"]:
            raise ValueError(f"inference_mode must be 'expectation', 'bernoulli', or 'disabled', got {self.inference_mode}")
        
        # Validate init mode
        if self.init_mode not in ["zero", "random", "reuse", "warmstart"]:
            raise ValueError(f"init_mode must be 'zero', 'random', 'reuse', or 'warmstart', got {self.init_mode}")
        
        # Validate distributional RL parameters
        if self.loss_type == "mle":
            if self.num_atoms < 2:
                raise ValueError(f"num_atoms must be at least 2 for mle loss, got {self.num_atoms}")
            if self.V_min >= self.V_max:
                raise ValueError(f"V_min must be less than V_max, got V_min={self.V_min}, V_max={self.V_max}")
        
        # Validate checkpoint configuration
        if self.init_mode == "warmstart" and self.classifier_checkpoint_path is None:
            raise ValueError("classifier_checkpoint_path must be provided when init_mode is 'warmstart'")
        
        if self.classifier_checkpoint_path is not None and self.resume_opt_scheduler is None:
            raise ValueError("resume_opt_scheduler must be specified when classifier_checkpoint_path is not None")
