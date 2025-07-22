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

import inspect
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import tqdm
from datasets import Dataset, IterableDataset
from torch import autocast
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations import (
    is_comet_available,
    is_mlflow_available,
    is_wandb_available,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available

from ..data_utils import maybe_apply_chat_template, maybe_extract_prompt
from ..models import create_reference_model, prepare_deepspeed
from ..models.utils import prepare_fsdp
from .callbacks import SyncRefModelCallback
from .dqo_config import DQOConfig
from .utils import (
    disable_dropout_in_model,
    empty_cache,
    flush_left,
    flush_right,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)


if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


if is_wandb_available():
    import wandb

if is_mlflow_available():
    import mlflow


def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right, and pad with pad_token_id"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they are
    not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForPreference

    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]},
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]])
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output


class QNetwork(nn.Module):
    """Q-network for DQO training."""
    
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer (single Q-value)
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQOTrainer(Trainer):
    """
    Trainer for Direct Q-function Optimization (DQO) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
            and loss. If no reference model is provided, the trainer will create a reference model with the same
            architecture as the model to be optimized.
        args ([`DQOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`DataCollatorForPreference`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. DQO supports [preference](#preference) type and. The format of the samples can
            be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "dqo"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DQOConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        model_id = model if isinstance(model, str) else model.config._name_or_path
        if args is None:
            model_name = model_id.split("/")[-1]
            args = DQOConfig(f"{model_name}-DQO")

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)

        if args.padding_value is not None:
            self.padding_value = args.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "`padding_value` is not specified in `DQOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please either set the `padding_value` argument in `DQOConfig`, or set "
                    "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                    "the trainer."
                )

        # Model
        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `DQOConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        if args.ref_model_init_kwargs is not None and not isinstance(ref_model, str):
            warnings.warn(
                "You passed ref_model_init_kwargs to the `DQOConfig`, but your ref_model is already instantiated. "
                "The `ref_model_init_kwargs` will be ignored."
            )
        if isinstance(ref_model, str):
            ref_model = self._create_model_from_path(ref_model, args, is_ref=True)

        # PEFT configuration and model wrapping
        model = self._prepare_peft_model(model, ref_model, peft_config, args)

        if args.generate_during_eval and not (is_wandb_available() or is_comet_available() or is_mlflow_available()):
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases, MLFlow or Comet to be installed."
                " Please install `wandb`, `mlflow` or `comet-ml` to resolve."
            )

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize Q-networks
        hidden_dim = model.config.hidden_size
        self.q_network = QNetwork(
            input_dim=hidden_dim,
            hidden_size=args.q_network_hidden_size,
            num_layers=args.q_network_num_layers
        )
        self.target_q_network = QNetwork(
            input_dim=hidden_dim,
            hidden_size=args.q_network_hidden_size,
            num_layers=args.q_network_num_layers
        )
        
        # Initialize target network with same weights
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Q-network optimizer
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=args.q_learning_rate)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DQO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Data collator
        if data_collator is None:
            data_collator = DataCollatorForPreference(pad_token_id=self.padding_value)

        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # DQO specific parameters
        self.beta = args.beta
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_update_freq = args.target_update_freq
        self.use_double_q = args.use_double_q
        self.exploration_noise = args.exploration_noise
        self.min_q_weight = args.min_q_weight
        self.dataset_num_proc = args.dataset_num_proc
        self._step_count = 0

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

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Prepare Q-networks for distributed training
        self.q_network = self.accelerator.prepare_model(self.q_network)
        self.target_q_network = self.accelerator.prepare_model(self.target_q_network, evaluation_mode=True)

    def _create_model_from_path(self, model_path: str, args: DQOConfig, is_ref: bool = False) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        if not is_ref:
            model_init_kwargs = args.model_init_kwargs or {}
        else:
            model_init_kwargs = args.ref_model_init_kwargs or {}

        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `DQOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )

        # Create model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_peft_model(
        self, model: PreTrainedModel, ref_model: PreTrainedModel, peft_config: Any, args: DQOConfig
    ) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DQO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DQOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            else:
                model = self._prepare_gradient_checkpointing(model, args)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        else:
            model = self._prepare_gradient_checkpointing(model, args)

        return model

    def _prepare_gradient_checkpointing(self, model: PreTrainedModel, args: DQOConfig):
        """Prepare the gradienting checkpointing for the model."""
        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if args.gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: DQOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    @staticmethod
    def process_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer  # the processing class is a processor
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"][0]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]

        return output

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DQOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "image_sizes",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            batch_size = self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

                # Unnecessary cache clearing to avoid OOM
                empty_cache()
                self.accelerator.free_memory()

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            self.train_dataset = self.train_dataset.add_column(
                name="ref_rejected_logps", column=all_ref_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            batch_size = self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            eval_dataset = eval_dataset.add_column(name="ref_rejected_logps", column=all_ref_rejected_logps)

            # Save calculated ref_chosen_logps and ref_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log probabilities of the reference model for a single padded batch of a DQO specific dataset."""
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt and
        completion sequences.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )
        if "image_sizes" in batch:
            output["image_sizes"] = torch.cat([batch["image_sizes"], batch["image_sizes"]], dim=0)

        # Concatenate the chosen and rejected completions
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return output

    def compute_lambda_return(
        self, 
        trajectory: dict[str, torch.Tensor], 
        rewards: torch.Tensor, 
        values: torch.Tensor,
        lambda_param: float = 1.0
    ) -> torch.Tensor:
        """
        Compute λ-return for bias mitigation as described in Section 4.2 of the DQO paper.
        
        Args:
            trajectory: Dictionary containing trajectory information
            rewards: Tensor of shape (batch_size, seq_len) containing rewards
            values: Tensor of shape (batch_size, seq_len) containing value estimates
            lambda_param: λ parameter for λ-return computation
            
        Returns:
            λ-return values of shape (batch_size, seq_len)
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device
        
        # Initialize λ-returns
        lambda_returns = torch.zeros_like(rewards)
        
        # Compute λ-return for each position
        for t in range(seq_len):
            if lambda_param == 1.0:
                # Monte Carlo return (sum of all future rewards)
                lambda_returns[:, t] = rewards[:, t:].sum(dim=1)
            else:
                # Weighted combination of n-step returns
                lambda_return = torch.zeros(batch_size, device=device)
                
                for n in range(1, seq_len - t + 1):
                    # n-step return: sum of next n rewards + discounted value
                    n_step_reward = rewards[:, t:t+n].sum(dim=1)
                    if t + n < seq_len:
                        n_step_return = n_step_reward + (self.gamma ** n) * values[:, t+n]
                    else:
                        n_step_return = n_step_reward
                    
                    # Weight for this n-step return
                    if n == seq_len - t:
                        weight = lambda_param ** (n - 1)
                    else:
                        weight = (1 - lambda_param) * (lambda_param ** (n - 1))
                    
                    lambda_return += weight * n_step_return
                
                lambda_returns[:, t] = lambda_return
        
        return lambda_returns

    def compute_importance_weights(
        self, 
        batch: dict[str, torch.Tensor], 
        current_policy_logps: torch.Tensor,
        behavior_policy_logps: torch.Tensor,
        clip_ratio: float = 2.0
    ) -> torch.Tensor:
        """
        Compute importance sampling weights as described in Section 4.3 of the DQO paper.
        
        Args:
            batch: Batch of data
            current_policy_logps: Log probabilities under current policy
            behavior_policy_logps: Log probabilities under behavior policy (reference)
            clip_ratio: Maximum ratio for clipping (e in the paper)
            
        Returns:
            Importance sampling weights
        """
        # Compute importance ratio: π(τ|s₁) / μ(τ|s₁)
        log_ratio = current_policy_logps - behavior_policy_logps
        ratio = torch.exp(log_ratio)
        
        # Clip the ratio to avoid gradient explosion
        clipped_ratio = torch.clamp(ratio, max=clip_ratio)
        
        return clipped_ratio

    def dqo_loss(
        self,
        model_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the DQO loss as described in Section 4.1 of the paper.
        
        This implements the core DQO objective with:
        1. Q-function parameterized by the policy: Q_θ(s_h, a_h) = β log π_θ(a_h|s_h) + V_φ(s_h)
        2. Value function loss with λ-return for bias mitigation
        3. Importance sampling for offline data reweighting
        """
        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        chosen_values = model_output.get("chosen_values", torch.zeros_like(chosen_logps))
        rejected_values = model_output.get("rejected_values", torch.zeros_like(rejected_logps))
        
        # Compute rewards (KL-regularized)
        chosen_rewards = chosen_logps - ref_chosen_logps if not self.reference_free else chosen_logps
        rejected_rewards = rejected_logps - ref_rejected_logps if not self.reference_free else rejected_logps
        
        # Compute Q-values using the DQO parameterization: Q(s,a) = β log π(a|s) + V(s)
        chosen_q_values = self.beta * chosen_logps + chosen_values
        rejected_q_values = self.beta * rejected_logps + rejected_values
        
        # Compute importance sampling weights for offline data reweighting
        chosen_importance_weights = self.compute_importance_weights(
            batch, chosen_logps, ref_chosen_logps
        )
        rejected_importance_weights = self.compute_importance_weights(
            batch, rejected_logps, ref_rejected_logps
        )
        
        # Prepare trajectory data for λ-return computation
        batch_size = chosen_logps.shape[0]
        seq_len = 1  # For simplicity, treating each response as single step
        
        chosen_rewards_expanded = chosen_rewards.unsqueeze(1)  # (batch_size, 1)
        rejected_rewards_expanded = rejected_rewards.unsqueeze(1)
        chosen_values_expanded = chosen_values.unsqueeze(1)
        rejected_values_expanded = rejected_values.unsqueeze(1)
        
        # Compute λ-returns for bias mitigation (Section 4.2)
        chosen_lambda_returns = self.compute_lambda_return(
            {}, chosen_rewards_expanded, chosen_values_expanded, lambda_param=1.0
        ).squeeze(1)
        rejected_lambda_returns = self.compute_lambda_return(
            {}, rejected_rewards_expanded, rejected_values_expanded, lambda_param=1.0
        ).squeeze(1)
        
        # Value function loss (Equation 4.6 in the paper)
        chosen_value_loss = chosen_importance_weights * (chosen_lambda_returns - chosen_values) ** 2
        rejected_value_loss = rejected_importance_weights * (rejected_lambda_returns - rejected_values) ** 2
        value_loss = (chosen_value_loss + rejected_value_loss).mean()
        
        # Policy loss (Equation 4.7 in the paper)
        # L_π(θ) = E[(V_φ(s_h) + β log π_θ(a_h|s_h) - r(s_h,a_h) - G^λ(s_h+1))²]
        chosen_policy_target = chosen_rewards + chosen_lambda_returns
        rejected_policy_target = rejected_rewards + rejected_lambda_returns
        
        chosen_policy_loss = chosen_importance_weights * (
            chosen_values + self.beta * chosen_logps - chosen_policy_target
        ) ** 2
        rejected_policy_loss = rejected_importance_weights * (
            rejected_values + self.beta * rejected_logps - rejected_policy_target
        ) ** 2
        
        policy_loss = (chosen_policy_loss + rejected_policy_loss).mean()
        
        # Total DQO loss
        total_loss = policy_loss + value_loss
        
        # Compute rewards for logging
        chosen_rewards_final = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards_final = self.beta * (rejected_logps - ref_rejected_logps).detach()
        
        metrics = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "chosen_rewards": chosen_rewards_final,
            "rejected_rewards": rejected_rewards_final,
            "chosen_q_values": chosen_q_values.detach(),
            "rejected_q_values": rejected_q_values.detach(),
            "importance_weights": (chosen_importance_weights + rejected_importance_weights).mean().detach() / 2,
        }
        
        return total_loss, metrics

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False}

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            model_kwargs["attention_mask"] = attention_mask
            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute mean logits for logging
        mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        # Add value estimates (simplified - in practice would use separate value network)
        if not is_ref_model:
            # Use Q-network to estimate values from hidden states
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            if hidden_states is not None:
                # Pool hidden states (mean pooling over sequence)
                pooled_hidden = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
                q_values = self.q_network(pooled_hidden).squeeze(-1)  # (batch_size,)
                
                output["chosen_values"] = q_values[:num_examples]
                output["rejected_values"] = q_values[num_examples:]
            else:
                # Fallback: use zeros if no hidden states available
                output["chosen_values"] = torch.zeros_like(output["chosen_logps"])
                output["rejected_values"] = torch.zeros_like(output["rejected_logps"])

        return output

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DQO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        # Compute DQO loss
        total_loss, dqo_metrics = self.dqo_loss(model_output, batch, ref_chosen_logps, ref_rejected_logps)

        # Update target Q-network periodically
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.update_target_network()

        # Compute reward accuracies
        reward_accuracies = (dqo_metrics["chosen_rewards"] > dqo_metrics["rejected_rewards"]).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(dqo_metrics["chosen_rewards"]).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(dqo_metrics["rejected_rewards"]).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(dqo_metrics["chosen_rewards"] - dqo_metrics["rejected_rewards"]).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}q_values/chosen"] = (
            self.accelerator.gather_for_metrics(dqo_metrics["chosen_q_values"]).detach().mean().item()
        )
        metrics[f"{prefix}q_values/rejected"] = (
            self.accelerator.gather_for_metrics(dqo_metrics["rejected_q_values"]).detach().mean().item()
        )
        metrics[f"{prefix}policy_loss"] = (
            self.accelerator.gather_for_metrics(dqo_metrics["policy_loss"]).detach().mean().item()
        )
        metrics[f"{prefix}value_loss"] = (
            self.accelerator.gather_for_metrics(dqo_metrics["value_loss"]).detach().mean().item()
        )
        metrics[f"{prefix}importance_weights"] = (
            self.accelerator.gather_for_metrics(dqo_metrics["importance_weights"]).detach().mean().item()
        )

        return total_loss.mean(), metrics

    def update_target_network(self):
        """Update target Q-network using soft updates (τ parameter)."""
        with torch.no_grad():
            for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.padding_value,
            )

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.padding_value,
                        )
                else:
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.padding_value,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.padding_value)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        ref_output = pad_to_length(ref_output, self.max_length, self.padding_value)
        ref_output_decoded = self.processing_class.batch_decode(ref_output, skip_special_tokens=True)

        return policy_output_decoded, ref_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch. Prediction/evaluation loop, shared by
        `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.generate_from_model_and_ref(self.model, random_batch)

            table = pd.DataFrame(
                columns=["Prompt", "Policy", "Ref Model"],
                data=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(
                        random_batch_dataset["prompt"], policy_output_decoded, ref_output_decoded
                    )
                ],
            )
            if "wandb" in self.args.report_to and self.accelerator.is_main_process:
                wandb.log({"game_log": wandb.Table(data=table)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(
                    name="game_log.csv",
                    table=table,
                )

            if "mlflow" in self.args.report_to and self.accelerator.is_main_process:
                mlflow.log_table(data=table, artifact_file="game_log.json")

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent(
            """\
            @article{ji2024enhancing,
                title        = {{Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization}},
                author       = {Kaixuan Ji and Guanlin Liu and Ning Dai and Qingping Yang and Renjie Zheng and Zheng Wu and Chen Dun and Quanquan Gu and Lin Yan},
                year         = 2024,
                journal      = {arXiv preprint arXiv:2410.09302},
                url          = {https://arxiv.org/abs/2410.09302},
            }"""
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="DQO",
            trainer_citation=citation,
            paper_title="Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization",
            paper_id="2410.09302",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
