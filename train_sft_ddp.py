import os
import random

import pyarrow.parquet as pq
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info



tokenizer = tiktoken.get_encoding("gpt2")
V = tokenizer.n_vocab
D, L, FF = 768, 6, 3072
Q_HEADS, KV_HEADS = 8, 8
HEAD_DIM = D // Q_HEADS
KV_DIM = KV_HEADS * HEAD_DIM
assert D % Q_HEADS == 0
assert Q_HEADS % KV_HEADS == 0


def rope(x, start_pos=0):
    dtype = x.dtype
    T, d = x.shape[-2:]
    positions = torch.arange(start_pos, start_pos + T,
                             device=x.device, dtype=torch.float32)
    inv_freq = 10000 ** (-torch.arange(0, d, 2, device=x.device,
                                      dtype=torch.float32) / d)
    angles = torch.outer(positions, inv_freq)
    cis = torch.polar(torch.ones_like(angles), angles)
    x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], d // 2, 2))
    return torch.view_as_real(x * cis).flatten(-2).to(dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, D + 2 * KV_DIM, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x, cache=None, start_pos=0):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split([D, KV_DIM, KV_DIM], dim=-1)
        q = q.view(B, T, Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rope(q, start_pos), rope(k, start_pos)

        if cache is not None:
            k_cache, v_cache = cache
            end_pos = start_pos + T
            k_cache[:, :, start_pos:end_pos].copy_(k)
            v_cache[:, :, start_pos:end_pos].copy_(v)
            k, v = k_cache[:, :, :end_pos], v_cache[:, :, :end_pos]

        causal = cache is None or start_pos == 0
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v,
                is_causal=causal, enable_gqa=True)
        return self.out(x.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(D, 2 * FF, bias=False)
        self.down = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.attn, self.ffn = Attention(), SwiGLU()

    def forward(self, x, cache=None, start_pos=0):
        x = x + self.attn(self.n1(x), cache=cache, start_pos=start_pos)
        return x + self.ffn(self.n2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.embed.weight

    def make_cache(self, batch_size, max_length, dtype=torch.bfloat16):
        device = next(self.parameters()).device
        return [(
            torch.empty(batch_size, KV_HEADS, max_length, HEAD_DIM,
                        device=device, dtype=dtype),
            torch.empty(batch_size, KV_HEADS, max_length, HEAD_DIM,
                        device=device, dtype=dtype),
        ) for _ in range(L)]

    def forward(self, tokens, cache=None, start_pos=0):
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache[i] if cache is not None else None,
                      start_pos=start_pos)
        return self.head(self.norm(x))





