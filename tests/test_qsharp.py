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

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from trl import QSharpConfig
from trl.models import QSharpClassifier, QSharpLogitProcessor

from .testing_utils import TrlTestCase


class TestQSharpConfig(TrlTestCase):
    """Test cases for QSharpConfig validation and initialization."""

    def test_default_config(self):
        """Test QSharpConfig with default values."""
        config = QSharpConfig(output_dir=self.tmp_dir)
        assert config.classifier_type == "Q"
        assert config.loss_type == "bce"
        assert config.use_bias is False
        assert config.init_mode == "reuse"
        assert config.inference_mode == "bernoulli"
        assert config.eta == 1.0
        assert config.num_atoms == 11
        assert config.V_min == 0.0
        assert config.V_max == 1.0

    def test_custom_config(self):
        """Test QSharpConfig with custom values."""
        config = QSharpConfig(
            output_dir=self.tmp_dir,
            classifier_type="V",
            loss_type="mle",
            use_bias=True,
            num_atoms=21,
            V_min=-1.0,
            V_max=1.0,
            eta=10.0,
            inference_mode="expectation",
        )
        assert config.classifier_type == "V"
        assert config.loss_type == "mle"
        assert config.use_bias is True
        assert config.num_atoms == 21
        assert config.V_min == -1.0
        assert config.V_max == 1.0
        assert config.eta == 10.0
        assert config.inference_mode == "expectation"

    def test_invalid_classifier_type(self):
        """Test that invalid classifier_type raises ValueError."""
        with pytest.raises(ValueError, match="classifier_type must be 'Q' or 'V'"):
            QSharpConfig(output_dir=self.tmp_dir, classifier_type="invalid")

    def test_invalid_loss_type(self):
        """Test that invalid loss_type raises ValueError."""
        with pytest.raises(ValueError, match="loss_type must be 'mse', 'bce', or 'mle'"):
            QSharpConfig(output_dir=self.tmp_dir, loss_type="invalid")

    def test_invalid_inference_mode(self):
        """Test that invalid inference_mode raises ValueError."""
        with pytest.raises(ValueError, match="inference_mode must be 'expectation', 'bernoulli', or 'disabled'"):
            QSharpConfig(output_dir=self.tmp_dir, inference_mode="invalid")

    def test_invalid_init_mode(self):
        """Test that invalid init_mode raises ValueError."""
        with pytest.raises(ValueError, match="init_mode must be 'zero', 'random', 'reuse', or 'warmstart'"):
            QSharpConfig(output_dir=self.tmp_dir, init_mode="invalid")

    def test_mle_with_invalid_num_atoms(self):
        """Test that mle loss with num_atoms < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_atoms must be at least 2"):
            QSharpConfig(output_dir=self.tmp_dir, loss_type="mle", num_atoms=1)

    def test_mle_with_invalid_value_range(self):
        """Test that mle loss with V_min >= V_max raises ValueError."""
        with pytest.raises(ValueError, match="V_min must be less than V_max"):
            QSharpConfig(output_dir=self.tmp_dir, loss_type="mle", V_min=1.0, V_max=0.0)

    def test_warmstart_without_checkpoint(self):
        """Test that warmstart mode without checkpoint path raises ValueError."""
        with pytest.raises(ValueError, match="classifier_checkpoint_path must be provided"):
            QSharpConfig(output_dir=self.tmp_dir, init_mode="warmstart")

    def test_checkpoint_without_resume_flag(self):
        """Test that checkpoint path without resume_opt_scheduler flag raises ValueError."""
        with pytest.raises(ValueError, match="resume_opt_scheduler must be specified"):
            QSharpConfig(output_dir=self.tmp_dir, classifier_checkpoint_path="/some/path")


class TestQSharpClassifier(TrlTestCase):
    """Test cases for QSharpClassifier model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.config = AutoConfig.from_pretrained(self.model_id)
        self.config.num_labels = len(self.tokenizer)

    @pytest.mark.parametrize("classifier_type", ["Q", "V"])
    @pytest.mark.parametrize("loss_type", ["mse", "bce", "mle"])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_classifier_initialization(self, classifier_type, loss_type, use_bias):
        """Test that classifier initializes correctly with different configurations."""
        classifier = QSharpClassifier(
            self.config,
            loss_type=loss_type,
            use_bias=use_bias,
            classifier_type=classifier_type,
            num_atoms=11,
            V_min=0.0,
            V_max=1.0,
        )
        assert classifier.classifier_type == classifier_type
        assert classifier.loss_type == loss_type
        assert classifier.use_bias == use_bias

        if loss_type == "mle":
            assert classifier.num_atoms == 11
            assert classifier.V_min == 0.0
            assert classifier.V_max == 1.0
            assert len(classifier.atoms) == 11

    def test_invalid_classifier_type(self):
        """Test that invalid classifier_type raises AssertionError."""
        with pytest.raises(AssertionError):
            QSharpClassifier(
                self.config,
                loss_type="bce",
                use_bias=False,
                classifier_type="invalid",
            )

    def test_invalid_loss_type(self):
        """Test that invalid loss_type raises AssertionError."""
        with pytest.raises(AssertionError):
            QSharpClassifier(
                self.config,
                loss_type="invalid",
                use_bias=False,
                classifier_type="Q",
            )

    def test_zero_init_classifier(self):
        """Test zero initialization of classifier weights."""
        classifier = QSharpClassifier(
            self.config,
            loss_type="bce",
            use_bias=True,
            classifier_type="Q",
        )
        classifier.zero_init_classifier()
        
        assert torch.all(classifier.score.weight == 0)
        assert torch.all(classifier.score.bias == 0)

    @pytest.mark.parametrize("loss_type", ["mse", "bce"])
    def test_q_classifier_output_shape(self, loss_type):
        """Test Q-classifier output shape for different loss types."""
        classifier = QSharpClassifier(
            self.config,
            loss_type=loss_type,
            use_bias=False,
            classifier_type="Q",
        )
        
        # Create dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.num_labels, (batch_size, seq_len))
        
        # Forward pass (inference mode)
        with torch.no_grad():
            outputs = classifier(input_ids=input_ids, use_cache=False)
        
        assert outputs.logits is not None

    def test_q_classifier_mle_output_shape(self):
        """Test Q-classifier output shape for MLE loss."""
        num_atoms = 11
        classifier = QSharpClassifier(
            self.config,
            loss_type="mle",
            use_bias=False,
            classifier_type="Q",
            num_atoms=num_atoms,
        )
        
        # Create dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.num_labels, (batch_size, seq_len))
        
        # Forward pass (inference mode) with logit_indices
        top_k = 5
        logit_indices = torch.randint(0, self.config.num_labels, (batch_size, top_k))
        
        with torch.no_grad():
            outputs = classifier(input_ids=input_ids, logit_indices=logit_indices, use_cache=False)
        
        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, top_k, num_atoms)

    def test_v_classifier_output_shape(self):
        """Test V-classifier output shape."""
        classifier = QSharpClassifier(
            self.config,
            loss_type="bce",
            use_bias=False,
            classifier_type="V",
        )
        
        # Create dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.num_labels, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        # Forward pass (inference mode) with logit_indices
        top_k = 5
        logit_indices = torch.randint(0, self.config.num_labels, (batch_size, top_k))
        
        with torch.no_grad():
            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logit_indices=logit_indices,
                use_cache=False
            )
        
        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, top_k)

    @pytest.mark.parametrize("loss_type", ["mse", "bce", "mle"])
    def test_calculate_loss(self, loss_type):
        """Test loss calculation for different loss types."""
        num_atoms = 11 if loss_type == "mle" else 1
        classifier = QSharpClassifier(
            self.config,
            loss_type=loss_type,
            use_bias=False,
            classifier_type="Q",
            num_atoms=num_atoms,
        )
        
        # Create dummy data
        batch_size, seq_len = 4, 10
        logits = torch.randn(batch_size, seq_len, num_atoms)
        labels = torch.rand(batch_size)
        loss_weights = torch.ones(batch_size)
        loss_mask = torch.ones(batch_size, seq_len)
        
        # Calculate loss
        loss = classifier.calculate_loss(logits, labels, loss_weights, loss_mask)
        
        assert loss is not None
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("loss_type", ["mse", "bce", "mle"])
    def test_calculate_predictions(self, loss_type):
        """Test prediction calculation for different loss types."""
        num_atoms = 11 if loss_type == "mle" else 1
        classifier = QSharpClassifier(
            self.config,
            loss_type=loss_type,
            use_bias=False,
            classifier_type="Q",
            num_atoms=num_atoms,
        )
        
        # Create dummy logits
        batch_size, seq_len = 4, 10
        logits = torch.randn(batch_size, seq_len, num_atoms)
        
        # Calculate predictions
        predictions = classifier.calculate_predictions(logits)
        
        assert predictions is not None
        assert predictions.shape == (batch_size, seq_len)
        assert torch.all((predictions >= 0) & (predictions <= 1))

    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        classifier = QSharpClassifier(
            self.config,
            loss_type="bce",
            use_bias=False,
            classifier_type="Q",
        )
        
        # Create dummy training data
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.num_labels, (batch_size, seq_len))
        labels = torch.rand(batch_size)
        loss_weights = torch.ones(batch_size)
        loss_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass in training mode
        outputs = classifier(
            input_ids=input_ids,
            labels=labels,
            loss_weights=loss_weights,
            loss_mask=loss_mask,
        )
        
        assert outputs.loss is not None
        assert torch.isfinite(outputs.loss)
        assert outputs.logits is not None


class TestQSharpLogitProcessor(TrlTestCase):
    """Test cases for QSharpLogitProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        
        config = AutoConfig.from_pretrained(self.model_id)
        config.num_labels = len(self.tokenizer)
        
        self.classifier = QSharpClassifier(
            config,
            loss_type="bce",
            use_bias=False,
            classifier_type="Q",
        )

    @pytest.mark.parametrize("inference_mode", ["expectation", "bernoulli", "disabled"])
    def test_logit_processor_initialization(self, inference_mode):
        """Test logit processor initialization with different inference modes."""
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode=inference_mode,
            top_k=-1,
            use_cache=True,
        )
        
        assert logit_processor.inference_mode == inference_mode
        assert logit_processor.eta == 1.0
        assert logit_processor.modify_top_k == -1

    def test_logit_processor_disabled_mode(self):
        """Test that disabled mode returns original logits unchanged."""
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode="disabled",
            top_k=-1,
            use_cache=True,
        )
        
        # Create dummy inputs
        batch_size = 2
        vocab_size = len(self.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, 5))
        ref_logits = torch.randn(batch_size, vocab_size)
        
        # Process logits
        logit_processor.reset_classifier_state()
        processed_logits = logit_processor(input_ids, ref_logits)
        
        # In disabled mode, logits should be unchanged
        assert torch.equal(processed_logits, ref_logits)

    @pytest.mark.parametrize("inference_mode", ["expectation", "bernoulli"])
    def test_logit_processor_modifies_logits(self, inference_mode):
        """Test that logit processor modifies logits in active modes."""
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode=inference_mode,
            top_k=10,
            use_cache=False,
        )
        
        # Create dummy inputs
        batch_size = 1
        vocab_size = len(self.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, 5))
        ref_logits = torch.randn(batch_size, vocab_size)
        
        # Process logits
        logit_processor.reset_classifier_state()
        processed_logits = logit_processor(input_ids, ref_logits)
        
        # Logits should be modified (not equal to original)
        assert processed_logits.shape == ref_logits.shape

    def test_logit_processor_with_mle_loss(self):
        """Test logit processor with MLE loss type."""
        config = AutoConfig.from_pretrained(self.model_id)
        config.num_labels = len(self.tokenizer)
        
        classifier = QSharpClassifier(
            config,
            loss_type="mle",
            use_bias=False,
            classifier_type="Q",
            num_atoms=11,
        )
        
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=classifier,
            inference_mode="bernoulli",
            top_k=10,
            use_cache=False,
        )
        
        # Create dummy inputs
        batch_size = 1
        vocab_size = len(self.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, 5))
        ref_logits = torch.randn(batch_size, vocab_size)
        
        # Process logits
        logit_processor.reset_classifier_state()
        processed_logits = logit_processor(input_ids, ref_logits)
        
        assert processed_logits.shape == ref_logits.shape

    def test_reset_classifier_state(self):
        """Test that reset_classifier_state properly resets internal state."""
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode="bernoulli",
            top_k=-1,
            use_cache=True,
        )
        
        # Initial state
        assert logit_processor.classifier_state["first_pass"] is True
        assert logit_processor.classifier_state["input_ids"] is None
        
        # Process some logits
        batch_size = 1
        vocab_size = len(self.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, 5))
        ref_logits = torch.randn(batch_size, vocab_size)
        
        logit_processor(input_ids, ref_logits)
        
        # State should be updated
        assert logit_processor.classifier_state["first_pass"] is False
        
        # Reset state
        logit_processor.reset_classifier_state()
        
        # State should be reset
        assert logit_processor.classifier_state["first_pass"] is True
        assert logit_processor.classifier_state["input_ids"] is None

    def test_get_classifier_values(self):
        """Test get_classifier_values method."""
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode="bernoulli",
            top_k=10,
            use_cache=False,
        )
        
        # Create dummy inputs
        batch_size = 1
        vocab_size = len(self.tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, 5))
        top_k_indices = torch.randint(0, vocab_size, (batch_size, 10))
        
        # Get classifier values
        classifier_values = logit_processor.get_classifier_values(input_ids, top_k_indices)
        
        assert classifier_values is not None
        assert classifier_values.shape[0] == batch_size
        assert classifier_values.shape[1] == 10

    def test_modify_top_k_logits(self):
        """Test modify_top_k_logits method."""
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=self.ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=self.classifier,
            inference_mode="bernoulli",
            top_k=10,
            use_cache=False,
        )
        
        # Create dummy data
        batch_size = 2
        vocab_size = len(self.tokenizer)
        ref_logits = torch.randn(batch_size, vocab_size)
        logit_offset = torch.randn(batch_size, 10)
        top_k_indices = torch.randint(0, vocab_size, (batch_size, 10))
        
        # Modify logits
        modified_logits = logit_processor.modify_top_k_logits(ref_logits, logit_offset, top_k_indices)
        
        assert modified_logits.shape == ref_logits.shape


class TestQSharpIntegration(TrlTestCase):
    """Integration tests for Q# components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @pytest.mark.parametrize("classifier_type", ["Q", "V"])
    @pytest.mark.parametrize("loss_type", ["bce", "mse", "mle"])
    def test_end_to_end_generation(self, classifier_type, loss_type):
        """Test end-to-end generation with Q# guidance."""
        # Load models
        ref_model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        
        # Create classifier
        config = AutoConfig.from_pretrained(self.model_id)
        config.num_labels = len(self.tokenizer)
        
        classifier = QSharpClassifier(
            config,
            loss_type=loss_type,
            use_bias=False,
            classifier_type=classifier_type,
            num_atoms=11 if loss_type == "mle" else 1,
        )
        
        # Create logit processor
        logit_processor = QSharpLogitProcessor(
            eta=1.0,
            ref_model=ref_model,
            ref_model_tokenizer=self.tokenizer,
            value_classifier=classifier,
            inference_mode="bernoulli",
            top_k=10,
            use_cache=True,
        )
        
        # Generate with guidance
        prompt = "Hello, world!"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        logit_processor.reset_classifier_state()
        
        from transformers.generation import LogitsProcessorList
        logit_processors = LogitsProcessorList([logit_processor])
        
        with torch.no_grad():
            outputs = ref_model.generate(
                **inputs,
                logits_processor=logit_processors,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        assert outputs is not None
        assert outputs.shape[1] > inputs["input_ids"].shape[1]

    def test_config_and_classifier_compatibility(self):
        """Test that QSharpConfig and QSharpClassifier work together."""
        config = QSharpConfig(
            output_dir=self.tmp_dir,
            classifier_type="Q",
            loss_type="mle",
            use_bias=True,
            num_atoms=21,
            V_min=-1.0,
            V_max=1.0,
        )
        
        # Create classifier with config parameters
        model_config = AutoConfig.from_pretrained(self.model_id)
        model_config.num_labels = len(self.tokenizer)
        
        classifier = QSharpClassifier(
            model_config,
            loss_type=config.loss_type,
            use_bias=config.use_bias,
            classifier_type=config.classifier_type,
            num_atoms=config.num_atoms,
            V_min=config.V_min,
            V_max=config.V_max,
        )
        
        assert classifier.classifier_type == config.classifier_type
        assert classifier.loss_type == config.loss_type
        assert classifier.use_bias == config.use_bias
        assert classifier.num_atoms == config.num_atoms
