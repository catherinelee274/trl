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

import inspect
import re
import textwrap
import warnings

import torch
from accelerate.utils import gather
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from transformers.utils import is_peft_available

from ..data_utils import is_conversational
from ..extras.profiling import profiling_context
from .grpo_trainer import GRPOTrainer, RewardFunc, RolloutFunc
from .utils import get_config_model_id, nanstd
from .vpo_config import VPOConfig


if is_peft_available():
    from peft import PeftConfig, PeftModel


def vpo_set_reward(*args, **kwargs):
    """Placeholder reward function name shown in logs as ``rewards/vpo_set_reward/*``.

    VPO computes its set-level scalarized reward internally in
    [`VPOTrainer._calculate_rewards`]; this callable is never actually invoked. It only exists so the parent
    [`GRPOTrainer`] is configured with a single, well-named reward column.
    """
    raise RuntimeError("`vpo_set_reward` is a placeholder and should never be called; VPO scores rewards internally.")


class VPOTrainer(GRPOTrainer):
    r"""
    Trainer for the Vector Policy Optimization (VPO) method. This algorithm was proposed in the paper [Vector Policy
    Optimization: Training for Diversity Improves Test-Time Search](https://huggingface.co/papers/2605.22817).

    VPO is **GRPO with a set-level, stochastically-scalarized reward**. Where GRPO scores a single completion with a
    scalar reward, VPO:

    1. **Multi-answer rollouts** — the policy emits a *set* of `num_answers` candidate answers in one autoregressive
       chain.
    2. **Vector reward** — each answer is scored by a *vector* reward `r(x, y) ∈ ℝ^d`, one component per reward
       function (the `d` objectives passed as `reward_funcs`).
    3. **Stochastic scalarization** — for each prompt group, `num_scalarizations` weight vectors `w ~ Dir(alpha)` are
       sampled (shared across the group) and the set is scored with
       `R̂(S) = (1/K) Σ_k max_{s ∈ S} w_kᵀ r(x, s)`.
    4. **GRPO** — `R̂(S)` is consumed as the per-rollout scalar reward (group-relative normalization + clipped
       policy-gradient loss, all inherited unchanged from [`GRPOTrainer`]).

    Optimizing the set reward rewards *diverse, competent* sets rather than collapsing to a single mode, which improves
    test-time search such as best-of-N and pass@k.

    Example:

    ```python
    from datasets import load_dataset
    from trl import VPOTrainer, VPOConfig

    # Each objective is a standard TRL reward function scoring a *single* answer.
    def closeness_to_target_0(completions, target_0, **kwargs):
        return [-abs(int(c) - t) for c, t in zip(completions, target_0)]

    def closeness_to_target_1(completions, target_1, **kwargs):
        return [-abs(int(c) - t) for c, t in zip(completions, target_1)]

    dataset = load_dataset("my_org/multi_target", split="train")

    trainer = VPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=[closeness_to_target_0, closeness_to_target_1],  # d = 2 objectives
        args=VPOConfig(output_dir="vpo-model", num_answers=3, num_scalarizations=16),
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            The model to train. See [`GRPOTrainer`] for details.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            The `d` reward objectives. Each must be a **callable** with the standard TRL reward-function signature
            (`(prompts, completions, completion_ids, **kwargs) -> list[float | None]`) that scores a *single* answer.
            VPO parses each rollout into `num_answers` answers, scores every answer with every objective to build the
            vector reward `r(x, y) ∈ ℝ^d`, and combines them via stochastic scalarization. Reward *models* (string ids
            or `nn.Module`) are not supported as objectives.
        args ([`VPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`]):
            Dataset to use for training. Must include a `"prompt"` column. Any extra columns are forwarded to the
            objective reward functions (e.g. per-objective targets).
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for evaluation. Must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Tokenizer or processor. See [`GRPOTrainer`] for details.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            Training callbacks. See [`GRPOTrainer`] for details.
        optimizers (`tuple`, *optional*):
            Optimizer and scheduler tuple. See [`GRPOTrainer`] for details.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration. See [`GRPOTrainer`] for details.
        rollout_func (`RolloutFunc`, *optional*):
            Custom rollout function. See [`GRPOTrainer`] for details.
    """

    _tag_names = ["trl", "vpo"]
    _name = "VPO"
    _paper = {
        "title": "Vector Policy Optimization: Training for Diversity Improves Test-Time Search",
        "id": "2605.22817",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{bahlousboldi2026vpo,
                title        = {{Vector Policy Optimization: Training for Diversity Improves Test-Time Search}},
                author       = {Mahdi Bahlous-Boldi and Raviteja Puri and Idan Shenfeld and Aviral Kumar and Mehul Damani and Sebastian Risi and Omar Khattab and Joey Hong and Lakshya A. Agrawal},
                year         = 2026,
                eprint       = {arXiv:2605.22817},
            }
            """),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: VPOConfig | None = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = VPOConfig(f"{model_name}-VPO")

        # The `d` objectives. Each must be a plain callable scoring a single answer; reward models are not supported.
        objective_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        for func in objective_funcs:
            if not callable(func) or isinstance(func, str | PreTrainedModel):
                raise ValueError(
                    "VPOTrainer objectives must be callable reward functions scoring a single answer. Reward models "
                    f"(string ids or `nn.Module`) are not supported as VPO objectives, got {type(func)}."
                )
        self.objective_funcs = objective_funcs
        self.objective_func_names = [func.__name__ for func in objective_funcs]
        self.num_objectives = len(objective_funcs)  # d

        if args.reward_weights is not None:
            warnings.warn(
                "`reward_weights` is ignored by VPOTrainer: objective trade-offs are handled by the sampled "
                "Dirichlet scalarization weights, not by fixed reward weights.",
                UserWarning,
                stacklevel=2,
            )
            args.reward_weights = None

        # Present GRPO with a single, well-named composite reward column. The real per-objective scoring and the
        # set-level stochastic scalarization happen in `_calculate_rewards`, which fills that column with R̂(S).
        super().__init__(
            model=model,
            reward_funcs=[vpo_set_reward],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        # VPO-specific knobs
        self.num_answers = args.num_answers  # m
        self.num_scalarizations = args.num_scalarizations  # K
        self.scalarization_alpha = args.scalarization_alpha
        self.answer_separator = args.answer_separator
        self.answer_pattern = re.compile(args.answer_pattern, re.DOTALL) if args.answer_pattern else None

    def _parse_answers(self, completion) -> list[str]:
        """Parse a single completion into exactly `num_answers` answer strings.

        Fewer parsed answers are padded by repeating the last one; extra answers are truncated.
        """
        # Extract the raw text from the completion (conversational completions are lists of messages).
        if isinstance(completion, list):
            text = "".join(msg.get("content", "") for msg in completion if isinstance(msg, dict))
        else:
            text = completion

        if self.answer_pattern is not None:
            matches = self.answer_pattern.findall(text)
            # `findall` returns the last group when groups are present, otherwise the whole match.
            answers = [m[-1] if isinstance(m, tuple) else m for m in matches]
            answers = [a.strip() for a in answers if a.strip()]
        else:
            answers = [seg.strip() for seg in text.split(self.answer_separator) if seg.strip()]

        if not answers:
            answers = [text.strip()]
        if len(answers) >= self.num_answers:
            return answers[: self.num_answers]
        # Pad by repeating the last answer so the set has exactly `num_answers` elements (duplicates don't change the
        # `max` over the set).
        return answers + [answers[-1]] * (self.num_answers - len(answers))

    def _sample_scalarization_weights(self, num_groups: int, device: torch.device) -> torch.Tensor:
        """Sample `num_scalarizations` Dirichlet weight vectors per group, shape `(num_groups, K, d)`.

        Sampling is seeded by the global step so that every process draws *identical* weights for the same group,
        keeping a prompt group's set rewards comparable even when its generations are split across processes.
        """
        concentration = torch.full((self.num_objectives,), float(self.scalarization_alpha))
        # Seed deterministically and identically across ranks, then restore the global RNG state.
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(int(self.args.seed) + int(self.state.global_step))
            weights = torch.distributions.Dirichlet(concentration).sample((num_groups, self.num_scalarizations))
        finally:
            torch.set_rng_state(rng_state)
        return weights.to(device)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Compute the VPO set-level, stochastically-scalarized reward `R̂(S)` for every rollout.

        Returns a gathered tensor of shape `(num_processes * B, 1)` so the inherited
        [`GRPOTrainer._generate_and_score_completions`] can treat it as a single-column reward and apply its usual
        group-relative advantage normalization.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        conversational = is_conversational(inputs[0])

        # Build reward kwargs exactly like GRPOTrainer (forward dataset columns + trainer hooks to the objectives).
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state
        reward_kwargs["log_extra"] = self._log_completion_extra
        reward_kwargs["log_metric"] = self._log_metric
        if self.environments is not None:
            reward_kwargs["environments"] = self.environments

        # Parse each rollout into `num_answers` answers, then flatten to (B * m) so objectives score one answer at a
        # time. Dataset columns / prompts / completion ids are repeated to stay aligned with the flattened answers.
        n = len(prompts)
        m = self.num_answers
        parsed = [self._parse_answers(c) for c in completions]  # B x m
        if conversational:
            answers_flat = [[{"role": "assistant", "content": a}] for ans in parsed for a in ans]
        else:
            answers_flat = [a for ans in parsed for a in ans]
        prompts_rep = [p for p in prompts for _ in range(m)]
        completion_ids_rep = [cid for cid in completion_ids_list for _ in range(m)]
        reward_kwargs_rep = {}
        for key, value in reward_kwargs.items():
            if isinstance(value, list) and len(value) == n:
                reward_kwargs_rep[key] = [v for v in value for _ in range(m)]
            else:
                reward_kwargs_rep[key] = value

        # Vector reward r(x, y) ∈ ℝ^d for every answer: shape (B, m, d).
        rewards_per_obj = torch.zeros(n * m, self.num_objectives, device=device)
        for j, (func, func_name) in enumerate(zip(self.objective_funcs, self.objective_func_names, strict=True)):
            if inspect.iscoroutinefunction(func):
                raise ValueError(
                    f"VPO objective '{func_name}' is async, which is not supported. Provide synchronous objectives."
                )
            with profiling_context(self, func_name):
                output = func(
                    prompts=prompts_rep,
                    completions=answers_flat,
                    completion_ids=completion_ids_rep,
                    **reward_kwargs_rep,
                )
            output = [r if r is not None else torch.nan for r in output]
            rewards_per_obj[:, j] = torch.tensor(output, dtype=torch.float32, device=device)
        rewards_per_obj = rewards_per_obj.view(n, m, self.num_objectives)

        # Gather across processes so prompt groups are contiguous and scalarization weights can be shared per group.
        rewards_per_obj = gather(rewards_per_obj)  # (N, m, d)
        total = rewards_per_obj.size(0)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        num_groups = total // num_generations
        grouped = rewards_per_obj.view(num_groups, num_generations, m, self.num_objectives)

        # Stochastic scalarization: R̂(S) = (1/K) Σ_k max_{s ∈ S} w_kᵀ r(x, s), weights shared within each group.
        weights = self._sample_scalarization_weights(num_groups, device)  # (num_groups, K, d)
        # (num_groups, G, m, d) x (num_groups, K, d) -> (num_groups, G, m, K)
        scalarized = torch.einsum("ngmd,nkd->ngmk", grouped, weights)  # (num_groups, G, m, K)
        set_reward = scalarized.max(dim=2).values.mean(dim=-1)  # max over set, mean over K -> (num_groups, G)
        set_reward = set_reward.reshape(total)

        # Log per-objective reward means and a reward-space diversity proxy (mean spread of the best answer per group).
        for j, name in enumerate(self.objective_func_names):
            self._metrics[mode][f"objectives/{name}/mean"].append(torch.nanmean(grouped[..., j]).item())
        self._metrics[mode]["set_reward"].append(torch.nanmean(set_reward).item())
        # Diversity: across a group's sets, how much does the argmax answer index vary per scalarization.
        best_answer = scalarized.max(dim=2).indices.float()  # (num_groups, G, K)
        self._metrics[mode]["answer_diversity"].append(nanstd(best_answer).item())

        return set_reward.unsqueeze(1)  # (N, 1)
