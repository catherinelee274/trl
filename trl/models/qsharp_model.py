"""Q# (Q-sharp) Classifier Model for Value-Based RL in LLM Post-Training.

This module implements the classifier model for Q# algorithm, which supports both Q-function
and V-function learning using distributional RL.

Reference: https://arxiv.org/abs/2502.20548
"""

from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers import LlamaPreTrainedModel, LlamaModel, LogitsProcessor
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache

# Try to import the attention mask preparation function, use fallback if not available
try:
    from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position
except ImportError:
    # Fallback for older transformers versions
    try:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_with_cache_position
    except ImportError:
        # If neither import works, we'll define a simple fallback
        def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask, sequence_length, target_length, dtype, device, min_dtype, cache_position, batch_size
        ):
            # Simple fallback: create a causal mask
            causal_mask = torch.full((target_length, target_length), min_dtype, dtype=dtype, device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place operations
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length, :mask_length] + attention_mask[:, None, None, :]
                causal_mask[..., :mask_length, :mask_length] = padding_mask
            return causal_mask


class QSharpClassifier(LlamaPreTrainedModel):
    """Q# Classifier for learning Q or V functions using distributional RL.
    
    Args:
        config: Model configuration
        loss_type: Type of loss function ("mse", "bce", or "mle")
        use_bias: Whether to use bias in the classification layer
        classifier_type: Type of classifier ("Q" or "V")
        num_atoms: Number of atoms for distributional RL (only for "mle" loss)
        V_min: Minimum value for distributional RL (only for "mle" loss)
        V_max: Maximum value for distributional RL (only for "mle" loss)
    """
    
    def __init__(
        self,
        config,
        loss_type: str,
        use_bias: bool,
        classifier_type: str,
        *,
        num_atoms: int = 11,
        V_min: float = 0.0,
        V_max: float = 1.0
    ):
        assert classifier_type in ["Q", "V"], f"classifier_type must be 'Q' or 'V', got {classifier_type}"
        assert loss_type in ["mse", "bce", "mle"], f"loss_type must be 'mse', 'bce', or 'mle', got {loss_type}"
        
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier_type = classifier_type
        self.model = LlamaModel(config)
        
        # Set up loss function and output layer based on loss type
        if loss_type == "mse":
            self.loss_fct = MSELoss(reduction="none")
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, 1, bias=use_bias)
        elif loss_type == "bce":
            self.loss_fct = BCEWithLogitsLoss(reduction="none")
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, 1, bias=use_bias)
        elif loss_type == "mle":
            self.num_atoms = num_atoms
            self.V_min = V_min
            self.V_max = V_max
            # linspace includes V_min (i=0) and V_max (i=-1)
            self.atoms = torch.linspace(self.V_min, self.V_max, self.num_atoms).float()
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels * self.num_atoms, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, self.num_atoms, bias=use_bias)
        
        self.loss_type = loss_type
        self.use_bias = use_bias
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def zero_init_classifier(self):
        """Initialize classifier weights to zero."""
        nn.init.zeros_(self.score.weight)
        if self.use_bias:
            nn.init.zeros_(self.score.bias)
    
    def calculate_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: torch.Tensor,
        loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss for training the classifier.
        
        Args:
            logits: Classifier outputs [bs, seqlen, 1] or [bs, seqlen, num_atoms]
            labels: Target rewards [bs]
            loss_weights: Per-example loss weights [bs]
            loss_mask: Mask for valid positions [bs, seqlen]
        
        Returns:
            Scalar loss value
        """
        assert len(logits.shape) == 3
        bs, seqlen, _ = logits.shape
        assert loss_mask.shape == (bs, seqlen)
        assert labels.shape == (bs,)
        assert loss_weights.shape == (bs,)
        
        if self.loss_type == "mse":
            # For MSE we will explicitly regress with sigmoid
            relevant_logits = torch.sigmoid(logits).squeeze(-1)  # [bs, seqlen]
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = self.loss_fct(relevant_logits, labels_expanded.to(relevant_logits.dtype))
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # [bs]
        elif self.loss_type == "bce":
            # For BCE, sigmoid is implicitly applied in the loss function
            assert logits.shape[2] == 1
            logits = logits.squeeze(-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = self.loss_fct(logits, labels_expanded)
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # [bs]
        elif self.loss_type == "mle":
            log_pmfs = F.log_softmax(logits, dim=-1)  # [bs, seqlen, num_atoms]
            # Round labels to nearest atom and clamp underflow and overflows to V_min, V_max
            label_indices = torch.round(labels * (self.num_atoms - 1)).long()
            label_indices = torch.clamp(label_indices, 0, self.num_atoms - 1)  # [bs]
            # For each batch_idx, select the corresponding index of num_atoms
            loss = -log_pmfs[torch.arange(bs), :, label_indices]  # [bs, seqlen]
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # [bs]
        else:
            raise RuntimeError("Impossible to reach.")
        
        assert loss.shape == loss_weights.shape
        loss = loss * loss_weights
        loss = loss.mean()
        return loss
    
    def calculate_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Use logits to predict E[R].
        
        Args:
            logits: [bs, seqlen, 1] or [bs, seqlen, num_atoms]
        
        Returns:
            Expected rewards [bs, seqlen]
        """
        if self.loss_type in ["mse", "bce"]:
            # For both MSE and BCE, we will use sigmoid to predict the value
            return torch.sigmoid(logits).squeeze(-1)  # [bs, seqlen]
        elif self.loss_type == "mle":
            pmfs = torch.softmax(logits, dim=-1)  # [bs, seqlen, num_atoms]
            if self.atoms.device != pmfs.device:
                self.atoms = self.atoms.to(pmfs.device)
            return (pmfs * self.atoms).sum(dim=-1)  # [bs, seqlen]
        else:
            raise NotImplementedError()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        logit_indices: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """Forward pass of the Q# classifier.
        
        Two use cases:
        1. Training: input_ids [bs, seq_len] with labels and loss_mask
        2. Inference: input_ids [bs, seq_len] for pre-fill, [bs, 1] for AR
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs = input_ids.size(0)
        
        if self.classifier_type == "Q":
            if labels is not None:
                # Training mode
                bs, seqlen = input_ids.shape
                transformer_outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = transformer_outputs[0]  # [bs, seqlen, hidden_dim]
                logits = self.score(hidden_states)  # [bs, seqlen, vocab_size] or [bs, seqlen, vocab_size * num_atoms]
                
                if self.loss_type in ["mse", "bce"]:
                    logits = logits.unsqueeze(-1)  # [bs, seqlen, vocab_size, 1]
                elif self.loss_type == "mle":
                    logits = logits.view(bs, seqlen, self.num_labels, self.num_atoms)
                
                indexed_logits = logits[:, :-1][torch.arange(bs)[:, None], torch.arange(seqlen-1), input_ids[:, 1:]]
                indexed_logits = indexed_logits.float()
                loss = self.calculate_loss(indexed_logits, labels, loss_weights, loss_mask[:, 1:])
                return SequenceClassifierOutputWithPast(loss=loss, logits=indexed_logits)
            else:
                # Inference mode
                bs, _ = input_ids.shape
                top_k = self.num_labels
                if logit_indices is not None:
                    top_k = logit_indices.size(1)
                
                transformer_outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = transformer_outputs[0]  # [bs, seqlen, hidden_dim] or [bs, 1, hidden_dim]
                hidden_states = hidden_states[:, -1]  # [bs, hidden_size]
                logits = self.score(hidden_states)  # [bs, vocab_size * (1 or num_atoms)]
                
                if self.loss_type in ["mse", "bce"]:
                    if logit_indices is not None:
                        logits = logits[torch.arange(bs)[:, None], logit_indices]
                elif self.loss_type == "mle":
                    if logit_indices is not None:
                        offsets = torch.arange(self.num_atoms, device=logit_indices.device)
                        expanded = logit_indices.unsqueeze(-1) * self.num_atoms + offsets
                        expanded = expanded.view(bs, -1)
                        logits = logits[torch.arange(bs)[:, None], expanded]
                    logits = logits.float().view(bs, top_k, self.num_atoms)
                
                return SequenceClassifierOutputWithPast(
                    logits=logits,
                    past_key_values=transformer_outputs.past_key_values,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )
        
        elif self.classifier_type == "V":
            assert return_dict, "V classifier must return dict"
            if labels is not None:
                # Training mode
                assert logit_indices is None
                assert loss_mask is not None
                transformer_outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states = transformer_outputs[0]  # [bs, seq_len, hidden_size]
                logits = self.score(hidden_states).float()  # [bs, seq_len, 1] or [bs, seq_len, num_atoms]
                loss = self.calculate_loss(logits, labels, loss_weights, loss_mask)
                return SequenceClassifierOutputWithPast(loss=loss, logits=logits)
            else:
                # Inference mode
                top_k = logit_indices.size(1)
                transformer_outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
                output_past_key_values = transformer_outputs.past_key_values
                dtype, device = output_past_key_values[0][0].dtype, output_past_key_values[0][0].device
                min_dtype = torch.finfo(dtype).min
                next_input_ids = logit_indices.to(input_ids.device)
                expanded_attention_mask = torch.cat(
                    [attention_mask, torch.ones((bs, top_k), dtype=torch.long, device=attention_mask.device)],
                    dim=1
                )
                cache_position = torch.arange(attention_mask.shape[1], expanded_attention_mask.shape[1], device=device)
                actual_position_ids = (torch.ones((1, top_k)) * attention_mask.shape[1]).to(
                    dtype=attention_mask.dtype, device=device
                )
                actual_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    expanded_attention_mask,
                    top_k,
                    expanded_attention_mask.shape[1],
                    dtype=dtype,
                    device=device,
                    min_dtype=min_dtype,
                    cache_position=cache_position,
                    batch_size=input_ids.shape[0]
                )
                diagonal_mask = torch.full((top_k, top_k), min_dtype)
                diagonal_mask.fill_diagonal_(0)
                diagonal_mask = diagonal_mask.to(dtype=actual_attention_mask.dtype, device=device)
                actual_attention_mask[:, :, :, -top_k:] = diagonal_mask
                transformer_outputs = self.model(
                    next_input_ids,
                    attention_mask=actual_attention_mask,
                    position_ids=actual_position_ids,
                    past_key_values=output_past_key_values,
                    use_cache=True,
                    cache_position=cache_position
                )
                hidden_states = transformer_outputs[0]
                hidden_states = hidden_states[:, -top_k:]
                logits = self.score(hidden_states)
                if self.loss_type == "mle":
                    assert logits.shape == (bs, top_k, self.num_atoms)
                else:
                    logits = logits.squeeze(-1)
                return SequenceClassifierOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=output_past_key_values,
                )


def log1p_exp(x: torch.Tensor) -> torch.Tensor:
    """Compute log(1 + exp(x)) in a numerically stable way."""
    return torch.logaddexp(x, torch.tensor(0.0).to(x.device))


class QSharpLogitProcessor(LogitsProcessor):
    """Logit processor for Q# guided generation.
    
    This processor modifies the logits during generation to guide the policy
    using the learned Q or V function.
    
    Args:
        eta: Reciprocal of KL divergence weight (larger = smaller KL)
        ref_model: Reference model for generation
        ref_model_tokenizer: Tokenizer for the reference model
        value_classifier: Trained Q# classifier
        inference_mode: Inference mode ("expectation", "bernoulli", or "disabled")
        top_k: Only modify the top k logits (-1 for all)
        cd_baseline: Whether to use CD baseline (0 or 1)
        use_cache: Whether to use KV cache for efficiency
    """
    
    def __init__(
        self,
        eta: float,
        ref_model,
        ref_model_tokenizer,
        value_classifier: QSharpClassifier,
        inference_mode: str,
        top_k: int,
        cd_baseline: int = 0,
        use_cache: bool = True
    ):
        self.eta = eta
        self.ref_model = ref_model
        self.ref_model_tokenizer = ref_model_tokenizer
        self.inference_mode = inference_mode
        self.modify_top_k = top_k
        assert self.inference_mode in ['expectation', 'bernoulli', 'disabled']
        self.cd_baseline = cd_baseline
        self.value_classifier = value_classifier
        self.loss_type = value_classifier.loss_type
        self.use_cache = use_cache
        self.classifier_state = {
            "input_ids": None,
            "attention_mask": None,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True
        }
    
    def reset_classifier_state(self):
        """Reset the classifier state for a new generation."""
        self.classifier_state = {
            "input_ids": None,
            "attention_mask": None,
            "use_cache": self.use_cache,
            "past_key_values": None,
            "first_pass": True
        }
    
    def get_classifier_values(self, input_ids: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Get classifier values for the given input and top-k indices."""
        if self.classifier_state['first_pass']:
            assert self.classifier_state['input_ids'] is None
            assert self.classifier_state['attention_mask'] is None
            assert self.classifier_state['past_key_values'] is None
            self.classifier_state['first_pass'] = False
            self.classifier_state['input_ids'] = input_ids
            pad_token_id = self.ref_model_tokenizer.pad_token_id
            attention_mask = input_ids.ne(pad_token_id).long()
            self.classifier_state['attention_mask'] = attention_mask.to(input_ids.dtype)
        else:
            # Update attention_mask and input_ids
            attention_mask = torch.cat(
                [self.classifier_state["attention_mask"], torch.ones_like(input_ids[:, -1:], dtype=torch.long)],
                dim=1
            )
            if not self.classifier_state["use_cache"]:
                input_ids = torch.cat([self.classifier_state["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]  # Due to cache, we only need the last token
            self.classifier_state["input_ids"] = input_ids
            self.classifier_state["attention_mask"] = attention_mask
        
        with torch.no_grad():
            classifier_outputs = self.value_classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=self.classifier_state["use_cache"],
                logit_indices=top_k_indices,
                past_key_values=self.classifier_state["past_key_values"]
            )
        
        if self.classifier_state['use_cache']:
            assert classifier_outputs.past_key_values is not None
            self.classifier_state['past_key_values'] = classifier_outputs.past_key_values
        
        return classifier_outputs.logits
    
    def modify_top_k_logits(
        self,
        ref_model_logits: torch.Tensor,
        logit_offset: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Modify the top-k logits with the given offset."""
        return torch.scatter_add(ref_model_logits, 1, top_k_indices.to(ref_model_logits.device), logit_offset)
    
    def __call__(self, input_ids: torch.Tensor, ref_model_logits: torch.Tensor) -> torch.Tensor:
        """Process logits during generation."""
        if self.inference_mode == 'disabled':
            return ref_model_logits
        
        if self.modify_top_k == -1:
            top_k_indices = torch.arange(ref_model_logits.size(-1)).unsqueeze(0).expand(ref_model_logits.size(0), -1)
        else:
            _, top_k_indices = torch.topk(ref_model_logits, self.modify_top_k, dim=-1)
        
        if self.loss_type == "mle":
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            log_pmfs = F.log_softmax(classifier_logits, dim=-1)
            atoms = self.value_classifier.atoms.float()
            if atoms.device != log_pmfs.device:
                atoms = atoms.to(log_pmfs.device)
            
            logit_offset = torch.logsumexp(log_pmfs + self.eta * atoms, dim=-1)
            logit_offset = logit_offset - logit_offset.min(dim=-1, keepdim=True).values
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)
        
        elif self.inference_mode == 'expectation':
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                logit_offset = F.logsigmoid(classifier_logits)
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)
        
        elif self.inference_mode == 'bernoulli':
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                log_numerator = log1p_exp(self.eta + classifier_logits)
                log_denominator = log1p_exp(classifier_logits)
                logit_offset = log_numerator - log_denominator
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)
        
        else:
            raise ValueError(f"Invalid inference mode: {self.inference_mode}")
        
        return combined_logits
