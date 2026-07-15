# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

_CONV_COMM_ENV = "FLA_CP_CONV_COMM"
_CONV_COMM_METHODS = ("all_gather", "p2p")


def _resolve_conv_comm_method(method: str | None) -> str:
    method = method or os.environ.get(_CONV_COMM_ENV, "all_gather")
    if method not in _CONV_COMM_METHODS:
        raise ValueError(f"{_CONV_COMM_ENV} must be one of {_CONV_COMM_METHODS}, got {method!r}")
    return method


def _p2p_exchange(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    send_group_peer: int | None,
    recv_group_peer: int | None,
) -> torch.Tensor:
    """Exchange one fixed-shape tensor using ranks local to ``group``."""
    send_tensor = send_tensor.contiguous()
    recv_tensor = torch.zeros_like(send_tensor)
    operations = []
    if recv_group_peer is not None:
        operations.append(dist.P2POp(dist.irecv, recv_tensor, group=group, group_peer=recv_group_peer))
    if send_group_peer is not None:
        operations.append(dist.P2POp(dist.isend, send_tensor, group=group, group_peer=send_group_peer))
    if not operations:
        return recv_tensor
    for request in dist.batch_isend_irecv(operations):
        request.wait()
    return recv_tensor


def all_gather_into_tensor(
    inp: torch.Tensor, out: torch.Tensor | None = None, group: ProcessGroup | None = None, async_op: bool = False
) -> tuple[torch.Tensor, dist.Work | None]:
    """
    All-gather a tensor across ranks.

    Args:
        inp: Input tensor to gather
        out: Optional output tensor of shape [world_size, *inp.shape]
        group: Process group
        async_op: Whether to perform async operation

    Returns:
        Tuple of (output tensor, handle if async_op else None)
    """
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def all_reduce_sum(
    inp: torch.Tensor, group: ProcessGroup | None = None, async_op: bool = False
) -> tuple[torch.Tensor, dist.Work | None]:
    """
    All-reduce sum a tensor across ranks.

    Args:
        inp: Input tensor to reduce (modified in-place)
        group: Process group
        async_op: Whether to perform async operation

    Returns:
        Tuple of (reduced tensor, handle if async_op else None)
    """
    handle = dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
    return inp, handle


def send_recv_fwd(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    recv_from_prev: bool = True,
    method: str | None = None,
) -> torch.Tensor:
    """
    Forward pass communication: send tensor to next rank, receive from previous rank.

    The default uses all-gather. Set ``FLA_CP_CONV_COMM=p2p`` to use
    group-local point-to-point neighbors for benchmark experiments.

    Args:
        send_tensor: Tensor to send (e.g., tails for conv1d)
        group: Process group
        recv_from_prev: If True, receive from previous rank; if False, receive from next rank
        method: Optional explicit ``all_gather`` or ``p2p`` override

    Returns:
        Received tensor from the specified rank (zeros if no valid source)
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if _resolve_conv_comm_method(method) == "p2p":
        if recv_from_prev:
            send_group_peer = rank + 1 if rank + 1 < world_size else None
            recv_group_peer = rank - 1 if rank > 0 else None
        else:
            send_group_peer = rank - 1 if rank > 0 else None
            recv_group_peer = rank + 1 if rank + 1 < world_size else None
        return _p2p_exchange(send_tensor, group, send_group_peer, recv_group_peer)

    # All-gather to ensure all ranks participate
    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)

    if recv_from_prev:
        # Receive from previous rank
        if rank == 0:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank - 1].clone()
    else:
        # Receive from next rank
        if rank == world_size - 1:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank + 1].clone()


def send_recv_bwd(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    recv_from_next: bool = True,
    method: str | None = None,
) -> torch.Tensor:
    """
    Backward pass communication: send gradient to previous rank, receive from next rank.

    The default uses all-gather. Set ``FLA_CP_CONV_COMM=p2p`` to use
    group-local point-to-point neighbors for benchmark experiments.

    Args:
        send_tensor: Gradient tensor to send
        group: Process group
        recv_from_next: If True, receive from next rank; if False, receive from previous rank
        method: Optional explicit ``all_gather`` or ``p2p`` override

    Returns:
        Received gradient tensor from the specified rank (zeros if no valid source)
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if _resolve_conv_comm_method(method) == "p2p":
        if recv_from_next:
            send_group_peer = rank - 1 if rank > 0 else None
            recv_group_peer = rank + 1 if rank + 1 < world_size else None
        else:
            send_group_peer = rank + 1 if rank + 1 < world_size else None
            recv_group_peer = rank - 1 if rank > 0 else None
        return _p2p_exchange(send_tensor, group, send_group_peer, recv_group_peer)

    # All-gather to ensure all ranks participate
    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)

    if recv_from_next:
        # Receive from next rank
        if rank == world_size - 1:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank + 1].clone()
    else:
        # Receive from previous rank
        if rank == 0:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank - 1].clone()


# ============ Convenience aliases for conv1d CP ============


def conv_cp_send_recv_fwd(tails: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """
    Conv1d CP forward: each rank sends its tails, receives previous rank's tails as heads.

    Args:
        tails: [W-1, D] or [N, D, W-1] - tail tokens from current rank
        group: Process group

    Returns:
        heads: Same shape as tails - head tokens from previous rank (zeros for rank 0)
    """
    return send_recv_fwd(tails, group, recv_from_prev=True)


def conv_cp_send_recv_bwd(d_initial_state: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """
    Conv1d CP backward: each rank sends d_initial_state, receives from next rank.

    The received gradient should be added to the last W-1 tokens' gradient.

    Args:
        d_initial_state: [W-1, D] or [N, D, W-1] - gradient w.r.t. initial state
        group: Process group

    Returns:
        recv_grad: Same shape - gradient from next rank (zeros for last rank)
    """
    return send_recv_bwd(d_initial_state, group, recv_from_next=True)
