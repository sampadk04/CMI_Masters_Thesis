"""
Universal language model, which accepts as its core a Transformer.

The Transformer is implemented in PyTorch and supports FlashAttention-2
"""

from typing import Union
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import Transformer, TransformerConfig, RMSNorm

# todo : inference function, with no grad, with kv cache for transformer, step() for mamba

class LM(nn.Module):
    def __init__(self, model_config: Union[TransformerConfig], vocab_size: int):
        super().__init__()

        self.config = model_config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.config.d_model, padding_idx=0)
        
        if isinstance(self.config, TransformerConfig):
            self.core = Transformer(self.config)
        else:
            raise ValueError(f"Unsupported model configuration: {model_config}")

        self.out_norm = RMSNorm(self.config.d_model, self.config.norm_eps)

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.core(x)
        x = self.out_norm(x)
        logits = self.lm_head(x)

        return logits
    
    def forward_up_to(self, tokens, layer):
        # tokens : (B, L)
        # layer (1->n_layers): will stop the forward pass just after this layer

        # x : (B, L, D) activations after {layer}

        x = self.embedding(tokens)
        x = self.core(x, stop_at_layer=layer)

        return x
    
    # taken from llama2.c
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    @torch.no_grad()
    def generate(self, idx, block_size, max_new_tokens, temperature=1.0, top_k=None, spl_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            idx: (B, L) tensor of indices
            block_size: maximum context length
            max_new_tokens: the number of tokens to generate
            temperature: temperature for sampling
            top_k: top_k for sampling
            spl_token: special token to stop generation
        
        Return:
            generated_idx: (B, L + max_new_tokens)
        """
        B,L = idx.size()
        
        for i in range(max_new_tokens):
            idx_cond = idx  if idx.size(1) <= block_size else idx[:, -block_size:]

            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)

            # pluck the logits at the final step and scaled by the desired temperature
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            # resample if `spl_token` is generated
            if spl_token:
                for j in range(B):
                    if idx_next[j].item() == spl_token:
                        # change the probability of the special token to 0
                        probs[j, spl_token] = 0 # alternatively can also set the corresponding logits to -inf
                        idx_next[j] = torch.multinomial(probs[j], num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx # generated_idx


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
            n_params -= self.embedding.weight.numel()
        return n_params