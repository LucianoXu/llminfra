import math
from dataclasses import dataclass
from typing import Optional, Tuple


import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 256 * 4
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float = 1.5
    norm_eps: float = 1e-5
    rope_theta: float = 50000.0

    context_length: int = 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = 'cpu'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device = 'cpu'):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device = device)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freq_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: str = 'cpu'):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            device=device
        )

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        device = 'cpu'
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False, device = device
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False, device = device
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False, device = device
        )
    
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: str = 'cpu'):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.attention = Attention(args, device)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            device = device
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, device = device)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device = device)

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama3(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 context_length: int,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 device: str = 'cpu'):
        
        super().__init__()

        params = ModelArgs()
        params.dim = dim
        params.vocab_size = vocab_size
        params.n_layers = num_layers
        params.n_heads = num_heads
        params.ffn_dim_multiplier = d_ff / dim
        params.context_length = context_length

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim, device=device
        )

        self.layers = torch.nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, device=device))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=device)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False, device=device
        )

        freqs_cis = precompute_freqs_cis(
            dim=params.dim // params.n_heads,
            end=params.context_length * 2,
            theta=params.rope_theta,
            device=device
        )

        # register precomputed as buffer
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @property
    def device(self):
        return self.tok_embeddings.weight.device

    @device.setter
    def device(self, device):
        self.tok_embeddings = self.tok_embeddings.to(device)
        self.norm = self.norm.to(device)
        self.output = self.output.to(device)
        self.freqs_cis = self.freqs_cis.to(device)
        for layer in self.layers:
            layer = layer.to(device)

    def reset_parameters(self):
        self.tok_embeddings.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()
        self.output.reset_parameters()


    def forward(self, tokens: torch.Tensor, attention_masks: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[: seqlen]


        mask = torch.full((seqlen, seqlen), float("-inf"), device=self.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # get around the bug in MPS
        if self.device.type == 'mps':
            mask = torch.nan_to_num(mask, nan=0.0)

        if attention_masks is not None:
            attention_masks = attention_masks.to(self.device)
            # Convert attention mask from (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
            attention_masks = attention_masks[:, None, None, :]
            padding_mask = torch.where(attention_masks == 0, 
                torch.tensor(-1e-9, device=self.device),
                torch.tensor(0.0, device=self.device))
            mask = mask[None, None, :, :]
            mask = mask + padding_mask

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h).float()
        return output
