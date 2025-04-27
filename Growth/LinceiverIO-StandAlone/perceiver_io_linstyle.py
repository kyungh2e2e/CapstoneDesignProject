from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class LinformerAttention(nn.Module):
    def __init__(self, dim, seq_len, k=128, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by number of heads"

        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.proj_k = nn.Parameter(torch.randn(seq_len, k))
        self.proj_v = nn.Parameter(torch.randn(seq_len, k))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None):
        b, n, d, h, k = x.shape[0], self.seq_len, self.dim_head, self.heads, self.k
        context = x if context is None else context

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        k = torch.einsum('bnd,nk->bkd', k, self.proj_k)
        v = torch.einsum('bnd,nk->bkd', v, self.proj_v)
        k, v = self.dropout(k), self.dropout(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b k (h d) -> b h k d', h=h)
        v = rearrange(v, 'b k (h d) -> b h k d', h=h)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn + 1e-6)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PerceiverIOLinstyle(nn.Module):
    def __init__(self, depth, dim, queries_dim, logits_dim, num_latents=64, latent_dim=512,
                 cross_heads=1, latent_heads=8, cross_dim_head=64, latent_dim_head=64, seq_len=784, k=128, decoder_ff=False):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = LinformerAttention(latent_dim, seq_len, k, heads=cross_heads, dim_head=cross_dim_head)
        self.cross_ff = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.GELU())

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinformerAttention(latent_dim, num_latents, k, heads=latent_heads, dim_head=latent_dim_head),
                nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.GELU())
            ]))

        self.decoder_cross_attn = LinformerAttention(queries_dim, num_latents, k, heads=cross_heads, dim_head=cross_dim_head)
        self.decoder_ff = nn.Sequential(nn.LayerNorm(queries_dim), nn.Linear(queries_dim, queries_dim), nn.GELU()) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if logits_dim else nn.Identity()

    def forward(self, data, queries):
        b = data.shape[0]
        x = repeat(self.latents, 'n d -> b n d', b=b)

        x = self.cross_attn(x, context=data) + x
        x = self.cross_ff(x) + x

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        latents = self.decoder_cross_attn(queries, context=x)
        if self.decoder_ff:
            latents = latents + self.decoder_ff(latents)

        return self.to_logits(latents)
