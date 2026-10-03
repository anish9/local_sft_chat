"""Self-contained TinyGPT SFT: messages data, model, DDP training and warmup."""

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



# Configuration: same architecture, data paths and optimizer settings.
tokenizer = tiktoken.get_encoding("gpt2")
V = tokenizer.n_vocab
D, L, FF = 768, 6, 3072
Q_HEADS, KV_HEADS = 8, 8
HEAD_DIM = D // Q_HEADS
KV_DIM = KV_HEADS * HEAD_DIM
assert D % Q_HEADS == 0
assert Q_HEADS % KV_HEADS == 0

TRAIN_DATA_DIR = Path("../sft_datasets/sft_local_oct2/train/")
TEST_DATA_DIR = Path("../sft_datasets/sft_local_oct2/eval/")
FOUNDATION_CKPT = "95m_param_model_ckpts/checkpoint_45600.pt"
SFT_CKPT_DIR = Path("sft_checkpoints_mul")

SEQ_LEN = 2048
BATCH_SIZE = 2            # Per GPU.
NUM_WORKERS = 0          # Same value on every rank; start with zero.
MAX_STEPS = 782500       # Your original limit, measured in optimizer updates.
LOG_EVERY = 10
EVAL_EVERY = 50
EVAL_MAX_BATCHES = 75    # Per GPU; set None to evaluate the entire split.

PEAK_LR = 5e-5
START_LR = 1e-6
WARMUP_STEPS = 300       # Starting setting; zero disables warmup.


# Data: one messages conversation per row.
IGNORE = -100
PAD_ID = tokenizer.eot_token


def encode_conversation(messages, tokenizer):
    """Return unshifted tokens and labels, supervising ALL assistant replies."""
    tokens, labels = [], []
    for message in messages:
        role, content = message["role"], message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("Every message content must be a string; serialize JSON first.")
        header = tokenizer.encode_ordinary(f"### {role.capitalize()}:\n")
        body = tokenizer.encode_ordinary(content) + [tokenizer.eot_token]
        tokens.extend(header + body)
        labels.extend([IGNORE] * len(header))
        labels.extend(body if role == "assistant" else [IGNORE] * len(body))
    return tokens, labels


class SFTParquetDataset(IterableDataset):
    def __init__(self, files, seq_len, messages_column="messages", *,
                 rank=0, world_size=1, shuffle=False, seed=42,
                 parquet_batch_size=1000, shuffle_buffer_size=256):
        super().__init__()
        self.files = sorted(Path(file) for file in files)
        if not self.files:
            raise FileNotFoundError("No Parquet files found for this dataset.")
        if seq_len < 1 or parquet_batch_size < 1 or shuffle_buffer_size < 1:
            raise ValueError("Lengths and buffer sizes must be positive.")
        if not 0 <= rank < world_size:
            raise ValueError("Require 0 <= rank < world_size.")
        self.seq_len = seq_len
        self.messages_column = messages_column
        self.rank, self.world_size = rank, world_size
        self.shuffle, self.seed = shuffle, seed
        self.parquet_batch_size = parquet_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _samples(self, tokenizer, shard_id, num_shards):
        files = list(self.files)
        # ALL ranks/workers must agree on file ordering BEFORE row sharding.
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(files)

        row_start = 0
        for file in files:
            with pq.ParquetFile(file) as parquet:
                for batch in parquet.iter_batches(
                    columns=[self.messages_column],
                    batch_size=self.parquet_batch_size,
                ):
                    # Global row i belongs to exactly one rank/worker shard.
                    first = (shard_id - row_start) % num_shards
                    indices = list(range(first, batch.num_rows, num_shards))
                    row_start += batch.num_rows
                    if not indices:
                        continue
                    column = batch.column(self.messages_column)
                    messages_rows = column.take(indices).to_pylist()

                    for messages in messages_rows:
                        if not messages or messages[-1]["role"] != "assistant":
                            continue
                        tokens, labels = encode_conversation(messages, tokenizer)
                        # Skip long conversations rather than cut an answer in half.
                        if len(tokens) > self.seq_len + 1:
                            continue
                        x_tokens, y_tokens = tokens[:-1], labels[1:]
                        if not x_tokens or not any(y != IGNORE for y in y_tokens):
                            continue
                        yield (
                            torch.tensor(x_tokens, dtype=torch.long),
                            torch.tensor(y_tokens, dtype=torch.long),
                        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        # Use the same num_workers on every rank.
        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers
        tokenizer = tiktoken.get_encoding("gpt2")
        samples = self._samples(tokenizer, shard_id, num_shards)

        if not self.shuffle:
            yield from samples
            return

        # Bounded local shuffle; do not keep the whole dataset in memory.
        rng = random.Random(self.seed + self.epoch * num_shards + shard_id)
        buffer = []
        for sample in samples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = sample
        rng.shuffle(buffer)
        yield from buffer


def sft_collate_fn(batch):
    x_list, y_list = zip(*batch)
    x = pad_sequence(x_list, batch_first=True, padding_value=PAD_ID)
    y = pad_sequence(y_list, batch_first=True, padding_value=IGNORE)
    return x, y


# Network: same computations as the supplied TinyGPT.
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


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def warmup_lr(step):
    fraction = min(step / WARMUP_STEPS, 1.0) if WARMUP_STEPS > 0 else 1.0
    return START_LR + (PEAK_LR - START_LR) * fraction


@torch.no_grad()
def evaluate(model, eval_loader, device, max_batches=EVAL_MAX_BATCHES):
    was_training = model.training
    model.eval()
    eval_model = unwrap(model)   # No DDP forward collectives during uneven eval.
    totals = torch.zeros(2, device=device, dtype=torch.float64)

    for i, (x, y) in enumerate(eval_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = eval_model(x)
            loss_sum = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                       ignore_index=-100, reduction="sum")
        totals[0] += loss_sum.double()
        totals[1] += y.ne(-100).sum()

    if dist.is_initialized():
        dist.all_reduce(totals)
    model.train(was_training)
    if totals[1].item() == 0:
        raise RuntimeError("Evaluation has no supervised tokens; check the evaluation data.")
    return (totals[0] / totals[1]).item()


def save_sft_checkpoint(step, epoch, model, optimizer, loss, keep_last=4):
    # Called by rank 0 only. Save normal parameter names, without 'module.'.
    path = SFT_CKPT_DIR / f"sft_checkpoint_{step}.pt"
    torch.save({
        "step": step, "epoch": epoch,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(), "loss": loss,
        "peak_lr": PEAK_LR, "start_lr": START_LR, "warmup_steps": WARMUP_STEPS,
    }, path)
    print(f"saved: {path}", flush=True)
    checkpoints = sorted(SFT_CKPT_DIR.glob("sft_checkpoint_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    while len(checkpoints) > keep_last:
        old = checkpoints.pop(0)
        old.unlink()
        print(f"deleted old: {old}", flush=True)


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl")

    try:
        train_files = sorted(TRAIN_DATA_DIR.glob("*.parquet"))
        eval_files = sorted(TEST_DATA_DIR.glob("*.parquet"))
        if not train_files or not eval_files:
            raise FileNotFoundError("Training and evaluation folders must both contain Parquet files.")

        train_dataset = SFTParquetDataset(train_files, seq_len=SEQ_LEN,
            rank=rank, world_size=world_size, shuffle=True)
        eval_dataset = SFTParquetDataset(eval_files, seq_len=SEQ_LEN,
            rank=rank, world_size=world_size, shuffle=False)

        loader_options = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            collate_fn=sft_collate_fn, pin_memory=True, persistent_workers=False)
        if NUM_WORKERS > 0:
            loader_options["multiprocessing_context"] = "spawn"
        train_loader = DataLoader(train_dataset, drop_last=True, **loader_options)
        eval_loader = DataLoader(eval_dataset, drop_last=False, **loader_options)

        model = TinyGPT()
        ckpt = torch.load(FOUNDATION_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        del ckpt
        model = model.to(device)  # FP32 parameters; BF16 forward via autocast.
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=START_LR, weight_decay=0.1)
        if rank == 0:
            SFT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

        model.train()
        step, epoch = 0, 0
        while step < MAX_STEPS:
            train_dataset.set_epoch(epoch)
            iterator = iter(train_loader)
            epoch_steps = 0

            while step < MAX_STEPS:
                batch = next(iterator, None)
                # All ranks agree to stop BEFORE the next DDP forward/backward.
                available = torch.tensor(int(batch is not None), device=device)
                if world_size > 1:
                    dist.all_reduce(available, op=dist.ReduceOp.MIN)
                if available.item() == 0:
                    break

                x, y = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                lr = warmup_lr(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss_sum = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                               ignore_index=-100, reduction="sum")

                # Global token mean, rather than an average of per-GPU means.
                stats = torch.stack((loss_sum.detach().double(), y.ne(-100).sum().double()))
                if world_size > 1:
                    dist.all_reduce(stats)
                # DDP averages gradients across ranks: compensate by world_size.
                loss = loss_sum * world_size / stats[1].to(loss_sum.dtype)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss = (stats[0] / stats[1]).item()
                if step % LOG_EVERY == 0 and rank == 0:
                    print(f"step={step:4d} | loss={train_loss:.4f} | lr={lr:.2e} | "
                          f"grad_norm={grad_norm.item():.3f} | batch_per_gpu={x.shape[0]} | "
                          f"seq_len_rank0={x.shape[1]} | supervised_tokens_all_gpus={int(stats[1].item())}",
                          flush=True)
                step += 1
                epoch_steps += 1

                if step % EVAL_EVERY == 0:
                    # Every rank evaluates its shard, then metric totals are reduced.
                    val_loss = evaluate(model, eval_loader, device)
                    if rank == 0:
                        print(f"step={step} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}",
                              flush=True)
                        save_sft_checkpoint(step, epoch, model, optimizer, train_loss)
                    if world_size > 1:
                        dist.barrier()  # Wait for rank 0 to finish saving.

            if epoch_steps == 0:
                raise RuntimeError("No common full training batch; check filtering, batch size and workers.")
            del iterator
            epoch += 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
