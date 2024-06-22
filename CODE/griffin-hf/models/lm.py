"""
Modified LMHead built on top of Griffin
"""

# install the transformer library from source to access the latest models
# pip install git+https://github.com/huggingface/transformers

from dataclasses import dataclass
from typing import Union
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RecurrentGemmaForCausalLM, RecurrentGemmaConfig

@dataclass
class GriffinCustomConfig:
    """
    Configuration class for the Griffin model.
    """
    d_model: int = 192
    n_layers: int = 3
    vocab_size: int = 1024
    n_heads: int = 8
    local_attention_window: int = 16 # image_size
    dropout: float = 0.0

# inherit from RecurrentGemmaForCausalLM
class LM(RecurrentGemmaForCausalLM):
    def __init__(self, model_config: GriffinCustomConfig, vocab_size: int):
        
        # extract config parameters from GriffinCustomConfig to RecurrentGemmaConfig
        config = RecurrentGemmaConfig(
            num_hidden_layers=model_config.n_layers,
            vocab_size=vocab_size,
            hidden_size=model_config.d_model,
            intermediate_size=model_config.d_model*3,
            num_attention_heads=model_config.n_heads,
            attention_window_size=model_config.local_attention_window,
            attention_dropout=model_config.dropout,
            pad_token_id=None,
            eos_token_id=None,
            bos_token_id=None,
            # HAWK: only recurrent blocks: 
            # block_types=("recurrent", "recurrent", "recurrent")
            # GRIFFIN: 2 reccurent and 1 local attention block
            # block_types=("recurrent", "recurrent", "attention")
        )

        # call the parent class constructor
        super().__init__(config=config)

    # taken from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # any parameters that is 2D will be weight decayed, otherwise no. (i.e. all weight tensors in matmuls + embeddings decay, all biases and rmnsnorms don't)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # subtract the position embeddings because they are not used in the final layer
            # here self.model = RecurrentGemmaModel
            n_params -= self.model.embed_tokens.weight.numel()
        return n_params