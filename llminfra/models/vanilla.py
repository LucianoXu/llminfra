from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.rand(1, dim))

    def forward(self, x):
        rms = torch.sqrt(torch.square(x).sum(-1, keepdim=True)/self.dim + self.epsilon)
        return x / rms * self.weight

SQRT2 = np.sqrt(2)

def gelu(x: torch.FloatTensor) -> torch.FloatTensor:
    return x * 0.5 * (1.0 + torch.erf(x / SQRT2))   # type: ignore

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(gelu(self.w1(x)))

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    subtracted_x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(subtracted_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, dropout: Optional[float] = None):
    d_k = k.size(-1)

    score = torch.matmul(q, k.transpose(-2, -1))/np.sqrt(d_k)

    if mask is not None:
        score += torch.where(mask, -1e9, 0.)

    attention_weights = softmax(score, -1)

    if dropout is not None:
        attention_weights = F.dropout(attention_weights, dropout)
    
    return attention_weights @ v
    


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.attn_pdrop = attn_pdrop
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):

        batch_shape = tuple(x.shape[:-1])
        seq_len = x.shape[1]

        q = (self.wq(x)).view(*batch_shape, self.num_heads, self.d_head).transpose(1, 2)
        k = (self.wk(x)).view(*batch_shape, self.num_heads, self.d_head).transpose(1, 2)
        v = (self.wv(x)).view(*batch_shape, self.num_heads, self.d_head).transpose(1, 2)

        casual_mask = torch.triu(torch.full((seq_len, seq_len), True, device = x.device), 1)

        output = scaled_dot_product_attention(k, q, v, casual_mask, self.attn_pdrop).transpose(1, 2).reshape(*batch_shape, -1)

        return self.wo(output)

        
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff, attn_pdrop: float|None = None, residual_pdrop: float|None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop if residual_pdrop is not None else 0.

        self.rmsnorm1 = RMSNorm(d_model)

        self.multi_atten = MultiHeadSelfAttention(d_model, num_heads, self.attn_pdrop)

        self.rmsnorm2 = RMSNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor):

        temp_x = self.rmsnorm1(x)
        temp_x = F.dropout(self.multi_atten(temp_x), self.residual_pdrop)

        x2 = temp_x + x

        temp_x = self.rmsnorm2(x2)
        temp_x = F.dropout(self.ffn(temp_x), self.residual_pdrop)

        return temp_x + x2
    

class Vanilla(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 dim: int,
                 num_heads: int,
                 d_ff: int,
                 attn_pdrop: float|None = None, 
                 residual_pdrop: float|None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_lenght = context_length
        self.num_layers = num_layers

        self.dim = dim
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop if residual_pdrop is not None else 0.


        self.embedding = nn.Embedding(vocab_size, dim)
        self.abs_pos_embedding = nn.Parameter(torch.rand(context_length, dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(
                dim,
                num_heads,
                d_ff,
                attn_pdrop,
                residual_pdrop
            ))

        self.rmsnorm = RMSNorm(dim)

        self.linear = nn.Linear(dim, vocab_size, bias=False)

    
    def forward(self, x: torch.Tensor):

        token_embedding = self.embedding(x)
        seq_len = x.shape[-1]
        pos_embedding = self.abs_pos_embedding[:seq_len, :]

        x = F.dropout(token_embedding + pos_embedding, self.residual_pdrop)

        for layer in self.layers:
            x = layer(x)

        x = self.linear(self.rmsnorm(x))

        return x