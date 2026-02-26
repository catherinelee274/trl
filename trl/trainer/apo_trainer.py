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
import warnings
from typing import Any

import torch
from accelerate.utils import gather, gather_object, is_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from ..extras.profiling import profiling_decorator
from ..models import prepare_deepspeed, prepare_fsdp
from ..models.utils import disable_gradient_checkpointing
from .apo_config import APOConfig
from .grpo_trainer import GRPOTrainer, RewardFunc, RolloutFunc
from .utils import (
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    nanstd,
    pad,
    use_adapter,
)

if is_peft_available():
    from peft import PeftConfig, PeftModel


class APOTrainer(GRPOTrainer):
    r"""
    Trainer for the A★-PO (Policy Optimization via Optimal Advantage Regression) method. This algorithm was
    proposed in the paper [A★-PO: Policy Optimization via Optimal Advantage
    Regression](https://huggingface.co/papers/2505.20686).

    A★-PO operates in two stages:

    1. **Stage 1 (offline)**: For each prompt, sample multiple responses from the reference policy, compute their
       rewards, and estimate the optimal value function via the soft-max formula:
       `v*(x) = β₁ · log(E_ref[exp(r(x,y) / β₁)])`. Store `v*(x)` in the training dataset.
    2. **Stage 2 (online training)**: For each prompt, generate a single response from the current policy and
       minimize the least-squares regression loss:
       `L = E[(β₂ · log(π(y|x)/π_ref(y|x)) − (r(x,y) − v*(x)))²]`

    This achieves competitive performance compared to PPO and GRPO while using only **1 generation per prompt**
    at training time, reducing training time and peak memory usage significantly.

    Example:

    ```python
    from trl import APOTrainer, APOConfig
    from trl.rewards import accuracy_reward

    # Dataset must include a "v_star" column with pre-computed optimal value estimates.
    # See the A-PO paper for how to generate this offline using the reference policy.
    dataset = load_dataset("my_org/my_dataset_with_v_star")

    trainer = APOTrainer(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        reward_funcs=accuracy_reward,
        args=APOConfig(output_dir="apo-model", beta2=1e-3),
        train_dataset=dataset["train"],
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            The model to train. See [`GRPOTrainer`] for details.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to compute rewards during training. See [`GRPOTrainer`] for details.
        args ([`APOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`]):
            Dataset to use for training. Must include a `"prompt"` column and the column specified by
            `args.v_star_column` (default: `"v_star"`) containing pre-computed optimal value estimates. If the
            `v_star_column` is absent, `v*(x) = 0` is used, reducing A★-PO to regression on raw rewards.
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for evaluation. Must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Tokenizer or processor. See [`GRPOTrainer`] for details.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or list, *optional*):
            Processing classes for reward models. See [`GRPOTrainer`] for details.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            Training callbacks. See [`GRPOTrainer`] for details.
        optimizers (`tuple`, *optional*):
            Optimizer and scheduler tuple. See [`GRPOTrainer`] for details.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration. See [`GRPOTrainer`] for details.
        rollout_func (`RolloutFunc`, *optional*):
            Custom rollout function. See [`GRPOTrainer`] for details.
    """

    _tag_names = ["trl", "apo"]
    _name = "APO"
    _paper = {
        "title": "A*-PO: Policy Optimization via Optimal Advantage Regression",
        "id": "2505.20686",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{gao2025apo,
                title        = {{A\\textsuperscript{\\textbf{*}}-PO: Policy Optimization via Optimal Advantage Regression}},
                author       = {Zhaolin Gao and Jonathan D. Chang and Wenhao Zhan and Gokul Swamy and Kiant{\\'e} Brantley and Thorsten Joachims and J. Andrew Bagnell and Jason D. Lee and Wen Sun},
                year         = 2025,
                eprint       = {arXiv:2505.20686},
            }
            """),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: APOConfig | None = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = APOConfig(f"{model_name}-APO")

        if args.num_generations != 1:
            warnings.warn(
                f"APOTrainer requires `num_generations=1` (got {args.num_generations}). "
                "A★-PO generates a single completion per prompt at training time; the value function is "
                "estimated offline. Overriding to `num_generations=1`.",
                UserWarning,
                stacklevel=2,
            )
            args.num_generations = 1

        # APOTrainer does not support tools or environments (no multi-turn rollouts).
        # These are inherited from GRPOTrainer but not applicable to A-PO's single-turn setting.
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        # Store A-PO specific parameters
        self.beta2 = args.beta2
        self.v_star_column = args.v_star_column

        # A★-PO always requires a reference model to compute log(π/π_ref).
        # GRPOTrainer only creates self.ref_model when beta != 0 (for KL penalty).
        # If it was not created, create and prepare it here.
        if self.ref_model is None and not is_peft_model(self.model):
            model_init_kwargs = args.model_init_kwargs or {}
            if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)
            if args.disable_dropout:
                disable_dropout_in_model(self.ref_model)
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # Keep the v_star column in the dataset so it is available in each batch.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images", self.v_star_column]

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # Extract v_star from the dataset batch. Falls back to 0.0 if the column is absent.
        v_star_local = torch.tensor(
            [float(inp.get(self.v_star_column, 0.0)) for inp in inputs],
            dtype=torch.float32,
            device=device,
        )

        # --- Image handling (same as GRPOTrainer) ---
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        # --- Generation (same as GRPOTrainer) ---
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # --- Convert token ID lists to padded tensors (same as GRPOTrainer) ---
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(lp, device=device) for lp in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        num_images = [len(img_list) for img_list in images] if images is not None else None

        # --- Multimodal forward_kwargs (same as GRPOTrainer) ---
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # --- Old log-probs for importance sampling (same as GRPOTrainer) ---
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            # --- vLLM importance sampling correction (same as GRPOTrainer) ---
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True) if sequence_level_is else per_token_logps_diff
                vllm_importance_sampling_ratio = torch.exp(logps_diff)
                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling mode: {self.vllm_importance_sampling_mode}."
                    )

            # --- Reference model log-probs (A-PO always needs these) ---
            # A★-PO requires log(π/π_ref) for the regression loss regardless of the KL coefficient (self.beta).
            if self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
            else:
                # PEFT: disable the adapter to recover the reference (base model) behaviour.
                model = self.accelerator.unwrap_model(self.model)
                with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )

        # --- Decode (same as GRPOTrainer) ---
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # --- Merge extra_fields from rollout_func (same as GRPOTrainer) ---
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # --- Reward computation ---
        # Rewards are gathered across all processes (shape: [num_processes * B]).
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # --- A★-PO advantage: A*(x, y) = r(x, y) − v*(x) ---
        # v_star is per-prompt (offline estimate). Gather across processes to align with `rewards`.
        v_star = gather(v_star_local)
        advantages = rewards - v_star

        # Slice to the local process subset
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_advantages = advantages.clone()  # keep for logging
        advantages = advantages[process_slice]

        # --- Logging (adapted from GRPOTrainer) ---
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(nanstd(rewards_per_func[:, i]).item())
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["v_star"].append(v_star.mean().item())
        self._metrics[mode]["advantage"].append(all_advantages.mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        # --- Build output dict ---
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "ref_per_token_logps": ref_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

    def _compute_loss(self, model, inputs):
        """A★-PO least-squares regression loss.

        Minimizes: `L = E[(β₂ · log(π(y|x)/π_ref(y|x)) − A*(x,y))²]`

        where `A*(x,y) = r(x,y) − v*(x)` is the A★-PO advantage computed in
        `_generate_and_score_completions`.
        """
        mode = "train" if self.model.training else "eval"

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute per-token log-probs for the current policy
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Sequence-level log-probability ratio: log π(y|x) − log π_ref(y|x)
        ref_per_token_logps = inputs["ref_per_token_logps"]
        log_pi = (per_token_logps * mask).sum(-1)       # (B,)
        log_pi_ref = (ref_per_token_logps * mask).sum(-1)  # (B,)
        log_ratio = log_pi - log_pi_ref                  # (B,)

        # A★-PO advantage: r(x,y) − v*(x), pre-computed in _generate_and_score_completions
        advantage = inputs["advantages"]  # (B,)

        # A★-PO least-squares regression loss
        # L = E[(β₂ · log(π/π_ref) − A*(x,y))²]
        loss = (self.beta2 * log_ratio - advantage) ** 2

        # Optional: add GRPO-style KL penalty (self.beta > 0 in APOConfig)
        # This is generally not recommended for A★-PO but is kept for flexibility.
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            kl_per_seq = (per_token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            loss = loss + self.beta * kl_per_seq
            mean_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = loss.mean() / normalizer

        # Logging
        completion_token_count = mask.sum().clamp(min=1.0)
        mean_entropy = (entropies * mask).sum() / completion_token_count
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        self._metrics[mode]["log_ratio"].append(self.accelerator.gather(log_ratio.mean()).mean().item())

        return loss
