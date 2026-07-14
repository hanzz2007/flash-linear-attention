# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Platform-neutral distributed benchmark for GDN context parallelism."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import statistics
import time
from contextlib import suppress
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

with suppress(ImportError):
    importlib.import_module("torch_npu")

import triton

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import IS_NPU, device, device_torch_lib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-seq-len", type=int, default=16384)
    parser.add_argument("--lengths", type=str, default=None, help="Comma-separated global variable-length sequences")
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--v-heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd"), default="fwd_bwd")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _synchronize() -> None:
    device_torch_lib.synchronize()


def _reset_peak_memory() -> None:
    if hasattr(device_torch_lib, "reset_peak_memory_stats"):
        device_torch_lib.reset_peak_memory_stats()


def _peak_memory() -> int:
    if hasattr(device_torch_lib, "max_memory_allocated"):
        return int(device_torch_lib.max_memory_allocated())
    return 0


def _package_version(name: str) -> str | None:
    with suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(name)
    return None


def _build_lengths(args: argparse.Namespace) -> list[int]:
    if args.lengths is None:
        return [args.total_seq_len]
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError(f"Invalid lengths: {args.lengths}")
    if sum(lengths) != args.total_seq_len:
        raise ValueError(f"The lengths sum to {sum(lengths)}, expected {args.total_seq_len}")
    return lengths


def _make_inputs(args: argparse.Namespace, rank: int, world_size: int, lengths: list[int]):
    if args.total_seq_len % world_size != 0:
        raise ValueError(f"total_seq_len={args.total_seq_len} must be divisible by world_size={world_size}")
    if args.v_heads % args.q_heads != 0:
        raise ValueError(f"v_heads={args.v_heads} must be divisible by q_heads={args.q_heads}")

    dtype = getattr(torch, args.dtype)
    local_seq_len = args.total_seq_len // world_size
    device_obj = torch.device(device, rank)
    torch.manual_seed(42 + rank)

    q = F.normalize(
        torch.randn(1, local_seq_len, args.q_heads, args.key_dim, device=device_obj, dtype=torch.float32),
        p=2,
        dim=-1,
    ).to(dtype).requires_grad_(True)
    k = F.normalize(
        torch.randn(1, local_seq_len, args.q_heads, args.key_dim, device=device_obj, dtype=torch.float32),
        p=2,
        dim=-1,
    ).to(dtype).requires_grad_(True)
    v = torch.randn(
        1,
        local_seq_len,
        args.v_heads,
        args.value_dim,
        device=device_obj,
        dtype=dtype,
        requires_grad=True,
    )
    g = F.logsigmoid(
        torch.randn(1, local_seq_len, args.v_heads, device=device_obj, dtype=torch.float32)
    ).to(dtype).requires_grad_(True)
    beta = torch.randn(
        1,
        local_seq_len,
        args.v_heads,
        device=device_obj,
        dtype=torch.float32,
    ).sigmoid().requires_grad_(True)
    do = torch.randn(1, local_seq_len, args.v_heads, args.value_dim, device=device_obj, dtype=dtype)

    cu_seqlens_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    cu_seqlens = cu_seqlens_cpu.to(device_obj)
    cp_context = build_cp_context(
        cu_seqlens,
        group=dist.group.WORLD,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return (q, k, v, g, beta), do, cp_context


def _run_once(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, ...],
    do: torch.Tensor,
    cp_context,
) -> torch.Tensor:
    for tensor in inputs:
        tensor.grad = None
    q, k, v, g, beta = inputs
    output, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cp_context=cp_context,
        state_v_first=args.state_v_first,
        chunk_size=args.chunk_size,
    )
    if args.mode == "fwd_bwd":
        output.backward(do)
    return output


def _measure(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, ...],
    do: torch.Tensor,
    cp_context,
    device_obj: torch.device,
) -> tuple[list[float], int]:
    # Compile or load every kernel before warmup and formal timing.
    result = _run_once(args, inputs, do, cp_context)
    _synchronize()
    tensors = [result]
    if args.mode == "fwd_bwd":
        tensors.extend(tensor.grad for tensor in inputs)
    finite = torch.tensor(
        float(all(torch.isfinite(tensor).all().item() for tensor in tensors)),
        device=device_obj,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise AssertionError("GDN CP benchmark produced a non-finite output or gradient")
    dist.barrier()

    for _ in range(args.warmup):
        _run_once(args, inputs, do, cp_context)
    _synchronize()
    dist.barrier()

    _reset_peak_memory()
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        _synchronize()
        started = time.perf_counter_ns()
        _run_once(args, inputs, do, cp_context)
        _synchronize()
        local_ms = (time.perf_counter_ns() - started) / 1e6
        # HCCL 9.0.0 does not support float64 reductions. Float32 retains
        # sub-microsecond resolution for the millisecond-scale samples here.
        critical_ms = torch.tensor(local_ms, dtype=torch.float32, device=device_obj)
        dist.all_reduce(critical_ms, op=dist.ReduceOp.MAX)
        samples.append(critical_ms.item())

    peak_memory = torch.tensor(_peak_memory(), dtype=torch.int64, device=device_obj)
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    return samples, int(peak_memory.item())


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    visible_var = "ASCEND_RT_VISIBLE_DEVICES" if IS_NPU else "CUDA_VISIBLE_DEVICES"
    visible_devices = os.environ.get(visible_var, "").split(",")
    device_key = visible_devices[local_rank].strip() if len(visible_devices) > local_rank else str(local_rank)
    os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/fla-triton-cache-{device}-{device_key}")
    backend = "hccl" if IS_NPU else "nccl"
    if IS_NPU:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=20))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_torch_lib.set_device(local_rank)
    device_obj = torch.device(device, local_rank)

    lengths = _build_lengths(args)
    inputs, do, cp_context = _make_inputs(args, local_rank, world_size, lengths)
    samples, peak_memory = _measure(args, inputs, do, cp_context, device_obj)

    if rank == 0:
        mean_ms = statistics.fmean(samples)
        std_ms = statistics.pstdev(samples)
        median_ms = statistics.median(samples)
        result = {
            "backend": backend,
            "device": device_torch_lib.get_device_name(local_rank),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "triton_ascend": _package_version("triton-ascend"),
            "world_size": world_size,
            "mode": args.mode,
            "dtype": args.dtype,
            "total_seq_len": args.total_seq_len,
            "local_seq_len": args.total_seq_len // world_size,
            "lengths": lengths,
            "q_heads": args.q_heads,
            "v_heads": args.v_heads,
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "chunk_size": args.chunk_size,
            "state_v_first": args.state_v_first,
            "warmup": args.warmup,
            "samples": args.samples,
            "latency_ms": {
                "min": min(samples),
                "p10": _percentile(samples, 10),
                "median": median_ms,
                "mean": mean_ms,
                "std": std_ms,
                "p90": _percentile(samples, 90),
                "max": max(samples),
                "cv": std_ms / mean_ms if mean_ms else 0.0,
            },
            "global_tokens_per_second": args.total_seq_len / (median_ms / 1e3),
            "peak_memory_bytes": peak_memory,
        }
        print(f"GDN_CP_BENCH_RESULT={json.dumps(result, sort_keys=True)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
