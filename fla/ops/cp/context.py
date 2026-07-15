# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from fla.utils import tensor_cache

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass
class FLACPContext:
    """FLA Context Parallel Context - Operator-level context management."""

    group: ProcessGroup | None = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None

    def copy_for_backward(self) -> FLACPContext:
        """Create a copy for backward pass (useful when PP_SIZE > 1)."""
        return FLACPContext(
            group=self.group,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            cu_seqlens_cpu=self.cu_seqlens_cpu.clone() if self.cu_seqlens_cpu is not None else None,
            is_last_rank=self.is_last_rank,
            pre_num_ranks=self.pre_num_ranks,
            is_first_rank=self.is_first_rank,
            post_num_ranks=self.post_num_ranks,
            conv1d_kernel_size=self.conv1d_kernel_size,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
        )

    @property
    def num_seqs(self) -> int:
        """Number of sequences in this rank."""
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        """Whether context parallel is enabled."""
        return self.group is not None


@tensor_cache
def get_cp_cu_seqlens(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: dist.ProcessGroup | None = None,
    conv1d_kernel_size: int | None = None,
) -> FLACPContext:
    if world_size is None:
        if group is None:
            raise ValueError("group is required when world_size and rank are not provided")
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
    elif rank is None:
        raise ValueError("rank is required when world_size is provided")

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
    if conv1d_kernel_size is not None and (
        isinstance(conv1d_kernel_size, bool) or not isinstance(conv1d_kernel_size, int) or conv1d_kernel_size <= 0
    ):
        raise ValueError(f"conv1d_kernel_size must be a positive integer, got {conv1d_kernel_size!r}")

    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError(f"cu_seqlens must be a tensor, got {type(cu_seqlens).__name__}")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cu_seqlens must have dtype int32 or int64, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be one-dimensional with at least two entries, got {tuple(cu_seqlens.shape)}")

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.detach().cpu()
    else:
        if not isinstance(cu_seqlens_cpu, torch.Tensor):
            raise TypeError(f"cu_seqlens_cpu must be a tensor, got {type(cu_seqlens_cpu).__name__}")
        if cu_seqlens_cpu.device.type != "cpu":
            raise ValueError(f"cu_seqlens_cpu must be on CPU, got {cu_seqlens_cpu.device}")
        if cu_seqlens_cpu.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"cu_seqlens_cpu must have dtype int32 or int64, got {cu_seqlens_cpu.dtype}")
        if cu_seqlens_cpu.ndim != 1 or cu_seqlens_cpu.numel() < 2:
            raise ValueError(
                f"cu_seqlens_cpu must be one-dimensional with at least two entries, got {tuple(cu_seqlens_cpu.shape)}"
            )

    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)
    device_cu_seqlens_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    if not torch.equal(device_cu_seqlens_cpu, cu_seqlens_cpu):
        raise ValueError("cu_seqlens and cu_seqlens_cpu must contain identical cumulative lengths")
    if cu_seqlens_cpu[0].item() != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens_cpu[0].item()}")
    total_tokens = cu_seqlens_cpu[-1].item()
    if total_tokens <= 0:
        raise ValueError(f"total tokens must be positive, got {total_tokens}")
    if not torch.all(cu_seqlens_cpu[1:] > cu_seqlens_cpu[:-1]).item():
        raise ValueError("cu_seqlens must be strictly increasing")

    if total_tokens < world_size:
        raise ValueError(f"total tokens ({total_tokens}) must be at least world_size ({world_size})")
    if total_tokens % world_size != 0:
        raise ValueError(f"total tokens ({total_tokens}) must be divisible by world_size ({world_size})")
    part_len = total_tokens // world_size
    rank_start = part_len * rank
    rank_end = rank_start + part_len

    start_seq_idx = int(torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side="right").item())
    end_seq_idx = int(torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side="left").item())
    subset_cu_seqlens = cu_seqlens_cpu[start_seq_idx : end_seq_idx + 1]

    local_cu_seqlens_cpu = (
        (subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start).unique_consecutive().to(torch.int32)
    )
    local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(
        device=cu_seqlens.device,
        non_blocking=True,
    )

    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()
    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)
    first_rank_of_first_seq = first_seq_global_start // part_len
    pre_num_ranks = rank - first_rank_of_first_seq
    is_first_rank = rank == first_rank_of_first_seq
    last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
    post_num_ranks = last_rank_of_last_seq - rank
    is_last_rank = rank == last_rank_of_last_seq

    return FLACPContext(
        group=group,
        cu_seqlens=local_cu_seqlens_gpu,
        cu_seqlens_cpu=local_cu_seqlens_cpu,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        is_first_rank=is_first_rank,
        post_num_ranks=post_num_ranks,
        conv1d_kernel_size=conv1d_kernel_size,
        pre_num_conv_tokens=pre_num_conv_tokens,
    )


def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: ProcessGroup,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext:
    """Build a CP context for the given cu_seqlens and process group.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor (before partition).
        group: Process group for CP communication.
        conv1d_kernel_size: Kernel size for convolution (optional).
        cu_seqlens_cpu: CPU version of cu_seqlens to avoid d2h transfer (optional).

    Returns:
        FLACPContext with computed cu_seqlens and rank information.
    """
    return get_cp_cu_seqlens(
        cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        group=group,
        conv1d_kernel_size=conv1d_kernel_size,
    )
