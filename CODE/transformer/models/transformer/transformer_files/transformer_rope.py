import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    """
    Configuration class for the Transformer model.

    Attributes:
        d_model (int): The dimensionality of the model.
        n_layers (int): The number of layers in the model.
        n_heads (int): The number of attention heads in the model.
        max_len (int): The maximum sequence length for positional embedding.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        bias (bool, optional): Whether to include bias terms. Defaults to False.
        norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-5.
        flash (bool, optional): Whether to enable flash training mode. Defaults to True.
    """

    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    dropout: float = 0.1
    bias: bool = False
    norm_eps: float = 1e-5
    flash: bool = True

    def __post_init__(self):
        """
        Performs post-initialization checks and calculations for the Transformer model.

        Raises:
            AssertionError: If `d_model` is not a multiple of `n_heads`.
        """
        assert self.d_model % self.n_heads == 0, "d_model must be a multiple of n_heads"
        # calculate the dimensionality of each head
        self.d_head = self.d_model // self.n_heads

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Initializes a Transformer model.

        It does the following:
        - Initializes the positional embedding layer.
        - Initializes the decoder layers.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__()

        self.config = config

        # self.PE = nn.Embedding(config.max_len, config.d_model)
        # use relative positional encodings
        self.in_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, X, stop_at_layer: int = None):
        """
        Performs forward pass of the Transformer model.
        
        It does the following:
        - Adds positional embeddings to the input tensor.
        - Passes the input tensor through the decoder layers.

        Args:
            X (torch.Tensor): Input tensor of shape (B, L, D).
            stop_at_layer (int, optional): If set, returns the activations after the {layer}-th layer.

        Returns:
            torch.Tensor: Output tensor of shape (B, L, D) or (B, L, d_model) if stop_at_layer is specified.
        """
        _, L, _ = X.size()

        # pos_emb = self.PE(torch.arange(0, L, dtype=torch.long, device=X.device))
        # X = self.in_dropout(X + pos_emb)
        X = self.in_dropout(X)

        for i, layer in enumerate(self.layers):
            X = layer(X) # (B, L, d_model)

            if stop_at_layer == i+1:
                return X
        
        return X
    
class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Initializes a DecoderLayer object.

        It does the following:
        - Initializes the self-attention layer.
        - Initializes the multi-layer perceptron (MLP) layer.

        Args:
            config (TransformerConfig): The configuration object for the Transformer model.
        """
        super().__init__()

        self.config = config

        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.sa = SelfAttentionMultiHead(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)
        
    def forward(self, X):
        """
        Performs a forward pass of the DecoderLayer.

        It does the following:
        - Passes the input tensor through the self-attention layer.
        - Adds the residual connection and layer normalization.
        - Passes the tensor through the multi-layer perceptron (MLP) layer.
        - Adds the residual connection and layer normalization.

        Args:
            X (torch.Tensor): The input tensor of shape (B, L, D).

        Returns:
            torch.Tensor: The output tensor of shape (B, L, D).
        """
        X = X + self.sa(self.attention_norm(X))
        X = X + self.mlp(self.mlp_norm(X))

        return X
    
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Multi-Layer Perceptron (MLP) module for the Transformer model.

        It does the following:
        - Initializes the first linear layer.
        - Initializes the second linear layer.
        - Initializes the third linear layer.
        - Initializes the dropout layer.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__()

        self.fc_1 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.fc_2 = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.fc_3 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the MLP module.

        It does the following:
        - Passes the input tensor through the first linear layer and applies SiLU activation function.
        - Also passes the input through the third linear layer.
        - Element-wise multiplication of the output of the first layer post activation and the output of the third layer.
        - Passes the result through the second linear layer.
        - Applies dropout to the output and returns the result.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor after passing through the MLP.
        """
        return self.dropout(self.fc_2(F.silu(self.fc_1(x)) * self.fc_3(x)))

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Initializes a SelfAttentionMultiHead module.

        It does the following:
        - Initializes the key, query, and value projections for all heads.
        - Initializes the output projection.
        - Initializes the regularization layers.

        Args:
            config (TransformerConfig): The configuration object for the transformer model.
        """
        super().__init__()

        self.config = config

        # key, query, value projections for all heads
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False) # d_query = n_heads*d_head as in the Transformer paper
        self.key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.query_rope = RotaryPositionalEmbeddings(config.d_head)
        self.key_rope = RotaryPositionalEmbeddings(config.d_head)

        if not config.flash:
            # compute the mask once and for all here 
            # registrer treats it like a parameter (device, state_dict...) without training
            mask = torch.full((1, 1, config.max_len, config.max_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)

        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        """
        Performs a forward pass of the SelfAttentionMultiHead module.

        It does the following:
        - Passes the input tensor through the key, query, and value projections.
        - Computes the scaled dot-product attention.
        - Passes the output through the output projection.
        - Applies dropout regularization.

        Args:
            X (torch.Tensor): The input tensor of shape (B, L, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (B, L, d_model).
        """
        B, L, _ = X.size()

        Q = self.query_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_query)
        K = self.key_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_key)
        V = self.value_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_head=d_value)

        Q = self.query_rope(Q)  # Apply RoPE to query
        K = self.key_rope(K)  # Apply RoPE to key

        if self.config.flash:
            attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        else:
            QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, L, L)
            QK_T = QK_T + self.mask[:, :, :L, :L]

            attention_scores = torch.softmax(QK_T / math.sqrt(self.config.d_head), dim=3) # (B, n_heads, L, L)
            attention = self.attn_drop(attention_scores) @ V # (B, n_heads, L, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, L, n_heads, d_head)
        y = attention.contiguous().view(B, L, self.config.d_model) # n_heads * d_head = d_model

        y = self.resid_dropout(self.c_proj(y))

        return y


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        """
        Initializes an instance of the RMSNorm module.

        It does the following:
        - Initializes the weight parameter.
        - Initializes the epsilon value for numerical stability.

        Args:
            dim (int): The input dimension.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Applies RMS normalization to the input tensor.

        It does the following:
        - Computes the mean of the squared values.
        - Computes the square root of the mean.
        - Normalizes the input tensor.
        - Scales the output by the weight parameter.
        - Returns the normalized tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Performs forward pass of the RMSNorm module.

        It does the following:
        - Applies RMS normalization to the input tensor.
        - Scales the output by the weight parameter.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMS normalization and scaling by the weight parameter.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.base = base
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        seq_len = x.shape[2]  # Sequence length is the third dimension
        d_head = x.shape[3]  # Dimension per head
        
        theta = 1. / (self.base ** (torch.arange(0, d_head, 2).float() / d_head)).to(x.device)
        
        seq_idx = torch.arange(seq_len, device=x.device).float()
        
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[None, None, :, :]
        self.sin_cached = idx_theta2.sin()[None, None, :, :]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d_model // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d_model], x[..., self.d_model:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached) + (neg_half_x * self.sin_cached)
        return torch.cat((x_rope, x_pass), dim=-1)