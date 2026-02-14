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

import gc

import pytest
import torch
from accelerate.utils.memory import release_memory
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import backend_empty_cache, torch_device

from trl import DQOConfig, DQOTrainer

from .testing_utils import TrlTestCase, require_peft, require_torch_accelerator


class TestDQOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _create_dummy_dataset(self, n_samples=10):
        """Create a dummy dataset for DQO training."""
        return Dataset.from_dict({
            "prompt": ["What is 2+2?" for _ in range(n_samples)],
            "completion": [" The answer is 4." for _ in range(n_samples)],
            "reward": [1.0 for _ in range(n_samples)],
        })

    def test_dqo_trainer_init(self):
        """Test that DQOTrainer initializes correctly."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        # Check that trainer components are initialized
        assert trainer.model is not None
        assert trainer.ref_model is not None
        assert trainer.value_model is not None
        assert trainer.value_optimizer is not None
        assert trainer.beta == 0.03
        assert trainer.lambda_return == 1.0
        assert trainer.importance_sampling_clip == 10.0

    def test_dqo_train(self):
        """Test basic DQO training."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=1e-6,
            value_learning_rate=1e-5,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        # Store initial parameters
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        previous_value_params = {n: param.clone() for n, param in trainer.value_model.named_parameters()}
        
        trainer.train()
        
        # Check that loss was logged
        assert trainer.state.log_history[-1]["loss"] is not None
        
        # Check that policy parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)
        
        # Check that value parameters have changed
        for n, param in previous_value_params.items():
            new_param = trainer.value_model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("beta", [0.01, 0.03, 0.1])
    def test_dqo_with_different_betas(self, beta):
        """Test DQO training with different beta values."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            beta=beta,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        assert trainer.state.log_history[-1]["loss"] is not None
        assert trainer.beta == beta

    @pytest.mark.parametrize("lambda_return", [0.0, 0.5, 1.0])
    def test_dqo_with_different_lambda(self, lambda_return):
        """Test DQO training with different lambda values."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            lambda_return=lambda_return,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        assert trainer.state.log_history[-1]["loss"] is not None
        assert trainer.lambda_return == lambda_return

    def test_dqo_with_process_rewards(self):
        """Test DQO training with per-token process rewards."""
        # Create dataset with per-token rewards
        n_samples = 10
        dataset = Dataset.from_dict({
            "prompt": ["What is 2+2?" for _ in range(n_samples)],
            "completion": [" The answer is 4." for _ in range(n_samples)],
            "rewards": [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(n_samples)],  # per-token rewards
        })
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        assert trainer.state.log_history[-1]["loss"] is not None

    def test_dqo_importance_sampling(self):
        """Test importance sampling weight computation."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            importance_sampling_clip=5.0,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        # Test importance weight computation
        policy_log_probs = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ref_log_probs = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        
        weights = trainer.compute_importance_weights(
            policy_log_probs, ref_log_probs, attention_mask.float()
        )
        
        # Check that weights are within clip range
        assert torch.all(weights >= 1.0 / trainer.importance_sampling_clip)
        assert torch.all(weights <= trainer.importance_sampling_clip)
        assert weights.shape == (2,)

    def test_dqo_lambda_return_computation(self):
        """Test lambda-return computation."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            lambda_return=0.95,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        # Test lambda-return computation
        batch_size, seq_len = 2, 5
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        policy_log_probs = torch.randn(batch_size, seq_len)
        ref_log_probs = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)
        
        lambda_returns = trainer.compute_lambda_return(
            rewards, values, policy_log_probs, ref_log_probs, attention_mask
        )
        
        assert lambda_returns.shape == (batch_size, seq_len)
        assert torch.isfinite(lambda_returns).all()

    def test_dqo_value_network(self):
        """Test value network architecture and forward pass."""
        from trl.trainer.dqo_trainer import ValueNetwork
        
        hidden_size = 128
        batch_size, seq_len = 2, 10
        
        value_net = ValueNetwork(input_dim=hidden_size, hidden_dim=256)
        
        # Test forward pass
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        values = value_net(hidden_states)
        
        assert values.shape == (batch_size, seq_len)
        assert torch.isfinite(values).all()

    def test_dqo_config_validation(self):
        """Test DQOConfig parameter validation."""
        # Test invalid lambda_return
        with pytest.raises(ValueError, match="lambda_return must be between 0 and 1"):
            DQOConfig(
                output_dir=self.tmp_dir,
                lambda_return=1.5,  # Invalid: > 1
            )
        
        with pytest.raises(ValueError, match="lambda_return must be between 0 and 1"):
            DQOConfig(
                output_dir=self.tmp_dir,
                lambda_return=-0.1,  # Invalid: < 0
            )
        
        # Test invalid importance_sampling_clip
        with pytest.raises(ValueError, match="importance_sampling_clip must be positive"):
            DQOConfig(
                output_dir=self.tmp_dir,
                importance_sampling_clip=-1.0,  # Invalid: negative
            )

    def test_dqo_metrics_logging(self):
        """Test that all expected metrics are logged."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            logging_steps=1,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        
        # Check that expected metrics are logged
        last_log = trainer.state.log_history[-1]
        expected_metrics = ["loss", "q_loss", "value_loss", "mean_q_value", "mean_value", "mean_reward", "mean_importance_weight"]
        
        for metric in expected_metrics:
            assert metric in last_log, f"Expected metric '{metric}' not found in logs"

    def test_dqo_dataset_without_rewards(self):
        """Test DQO with dataset that has no rewards (should use zeros)."""
        dataset = Dataset.from_dict({
            "prompt": ["What is 2+2?" for _ in range(10)],
            "completion": [" The answer is 4." for _ in range(10)],
            # No rewards field
        })
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        assert trainer.state.log_history[-1]["loss"] is not None

    def test_dqo_save_and_load(self):
        """Test that DQO trainer can save and load models."""
        dataset = self._create_dummy_dataset()
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            save_strategy="steps",
            save_steps=1,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        
        # Check that model and value network are saved
        import os
        assert os.path.exists(os.path.join(self.tmp_dir, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(self.tmp_dir, "model.safetensors"))
        assert os.path.exists(os.path.join(self.tmp_dir, "value_model.pt"))

    def test_dqo_with_eval_dataset(self):
        """Test DQO with evaluation dataset."""
        train_dataset = self._create_dummy_dataset(n_samples=20)
        eval_dataset = self._create_dummy_dataset(n_samples=10)
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=2,
            eval_strategy="steps",
            eval_steps=1,
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        
        # Check that evaluation metrics are logged
        assert any("eval_loss" in log for log in trainer.state.log_history)

    def test_dqo_data_collator(self):
        """Test the DQO data collator."""
        from trl.trainer.dqo_trainer import DataCollatorForDQO
        
        collator = DataCollatorForDQO(pad_token_id=0)
        
        # Create dummy examples
        examples = [
            {"input_ids": [1, 2, 3, 4], "rewards": [0.1, 0.2, 0.3, 0.4]},
            {"input_ids": [5, 6], "rewards": [0.5, 0.6]},
        ]
        
        batch = collator(examples)
        
        # Check batch structure
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "rewards" in batch
        
        # Check shapes (padded to longest sequence)
        assert batch["input_ids"].shape == (2, 4)
        assert batch["attention_mask"].shape == (2, 4)
        assert batch["rewards"].shape == (2, 4)
        
        # Check padding
        assert batch["input_ids"][1, 0] == 0  # Padded
        assert batch["attention_mask"][1, 0] == 0  # Padded


@pytest.mark.slow
@require_torch_accelerator
class TestDQOTrainerSlow(TrlTestCase):
    def setup_method(self):
        self.dataset = Dataset.from_dict({
            "prompt": ["Solve: x + 2 = 5" for _ in range(50)],
            "completion": [" Step 1: Subtract 2 from both sides\nStep 2: x = 3" for _ in range(50)],
            "reward": [1.0 for _ in range(50)],
        })
        self.max_length = 128

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @pytest.mark.parametrize("lambda_return", [0.0, 0.95, 1.0])
    @pytest.mark.parametrize("model_id", [
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "trl-internal-testing/tiny-MistralForCausalLM-0.2",
    ])
    def test_dqo_full_training(self, model_id, lambda_return):
        """Test full DQO training on different models."""
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            max_steps=5,
            learning_rate=1e-6,
            value_learning_rate=1e-5,
            lambda_return=lambda_return,
            beta=0.03,
            max_length=self.max_length,
            logging_strategy="no",
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        release_memory(model, trainer)

    @require_peft
    @pytest.mark.parametrize("model_id", [
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    ])
    def test_dqo_with_peft(self, model_id):
        """Test DQO training with PEFT (LoRA)."""
        from peft import LoraConfig
        
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        training_args = DQOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            max_steps=5,
            learning_rate=1e-6,
            value_learning_rate=1e-5,
            gradient_checkpointing=True,
            max_length=self.max_length,
            logging_strategy="no",
            report_to="none",
        )
        
        trainer = DQOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        release_memory(model, trainer)


class TestDQOConfig(TrlTestCase):
    def test_dqo_config_defaults(self):
        """Test DQOConfig default values."""
        config = DQOConfig(output_dir=self.tmp_dir)
        
        assert config.beta == 0.03
        assert config.lambda_return == 1.0
        assert config.importance_sampling_clip == 10.0
        assert config.value_learning_rate == 1e-5
        assert config.learning_rate == 5e-7
        assert config.max_length == 1024
        assert config.truncation_mode == "keep_end"

    def test_dqo_config_custom_values(self):
        """Test DQOConfig with custom values."""
        config = DQOConfig(
            output_dir=self.tmp_dir,
            beta=0.05,
            lambda_return=0.95,
            importance_sampling_clip=5.0,
            value_learning_rate=5e-6,
            learning_rate=1e-7,
            max_length=512,
        )
        
        assert config.beta == 0.05
        assert config.lambda_return == 0.95
        assert config.importance_sampling_clip == 5.0
        assert config.value_learning_rate == 5e-6
        assert config.learning_rate == 1e-7
        assert config.max_length == 512
