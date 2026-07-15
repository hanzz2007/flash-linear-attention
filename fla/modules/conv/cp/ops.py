# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.distributed as dist

from fla.ops.cp import FLACPContext, all_gather_into_tensor, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd
from fla.ops.utils import prepare_chunk_indices


class CausalConv1dFunctionCP(torch.autograd.Function):
    """
    Context Parallel version of CausalConv1dFunction.

    Forward:
        1. Get tails from previous rank to construct initial_state
        2. Call causal_conv1d_fwd

    Backward:
        1. Call causal_conv1d_bwd to get dx
        2. Sync communication: add next rank's first W-1 token gradients to current rank's last W-1 tokens
    """

    @staticmethod
    def _right_aligned_halo(x: torch.Tensor, halo_len: int) -> torch.Tensor:
        """Return a fixed-size halo with short inputs padded on the left."""
        assert x.dim() == 2, f"halo source must be [T, D], got {x.shape}"
        halo = x.new_zeros(halo_len, x.shape[-1])
        valid_len = min(halo_len, x.shape[0])
        if valid_len > 0:
            halo[-valid_len:].copy_(x[-valid_len:])
        return halo

    @staticmethod
    def _assemble_previous_halo(
        gathered_halos: torch.Tensor,
        *,
        rank: int,
        local_t: int,
        needed: int,
    ) -> torch.Tensor:
        """Assemble up to W-1 preceding tokens from one or more earlier ranks."""
        _, halo_len, D = gathered_halos.shape
        result = gathered_halos.new_zeros(halo_len, D)
        if needed == 0:
            return result
        valid_per_rank = min(halo_len, local_t)
        history = gathered_halos[:rank, -valid_per_rank:].reshape(-1, D)
        if history.shape[0] < needed:
            raise RuntimeError(f"CP halo requires {needed} prior tokens, but only {history.shape[0]} are available")
        result[-needed:] = history[-needed:]
        return result

    @staticmethod
    def _accumulate_multi_rank_halo_gradients(
        dx: torch.Tensor,
        gathered_gradients: torch.Tensor,
        *,
        rank: int,
    ) -> None:
        """Map right-aligned source halos back to their global token owners."""
        local_t = dx.shape[1]
        halo_len = gathered_gradients.shape[1]
        target_start = rank * local_t
        target_end = target_start + local_t
        correction = torch.zeros_like(dx, dtype=torch.float32)
        for source_rank in range(rank + 1, gathered_gradients.shape[0]):
            source_start = source_rank * local_t - halo_len
            source_end = source_rank * local_t
            overlap_start = max(target_start, source_start)
            overlap_end = min(target_end, source_end)
            if overlap_start >= overlap_end:
                continue
            dx_start = overlap_start - target_start
            grad_start = overlap_start - source_start
            length = overlap_end - overlap_start
            correction[0, dx_start : dx_start + length].add_(
                gathered_gradients[source_rank, grad_start : grad_start + length].float()
            )
        dx.copy_((dx.float() + correction).to(dx.dtype))

    @staticmethod
    def _prepare_initial_state_for_cp(
        x: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        context: FLACPContext,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor | None:
        """Prepare initial_state for CP forward pass by communicating with previous rank.

        Args:
            x: Input tensor of shape [1, T, D]
            weight: Weight tensor of shape [D, W]
            cu_seqlens: Cumulative sequence lengths
            context: CP context
            group: Process group for communication

        Returns:
            initial_state: Initial state tensor of shape [N, D, W] or None
        """
        if group is None:
            return None

        W = weight.shape[-1]  # weight: [D, W]
        D = weight.shape[0]
        assert x.dim() == 3 and x.shape[0] == 1, f"CP requires [1, T, D], got {x.shape}"
        tails = CausalConv1dFunctionCP._right_aligned_halo(x.squeeze(0), W - 1)
        if x.shape[1] < W - 1:
            gathered_halos, _ = all_gather_into_tensor(tails, group=group)
            heads = CausalConv1dFunctionCP._assemble_previous_halo(
                gathered_halos,
                rank=dist.get_rank(group),
                local_t=x.shape[1],
                needed=min(W - 1, context.pre_num_conv_tokens),
            )
        else:
            heads = conv_cp_send_recv_fwd(tails, group)
        if context.is_first_rank:
            return None

        # Non-first rank needs initial_state.
        N = len(cu_seqlens) - 1
        initial_state = torch.zeros(N, D, W, device=x.device, dtype=x.dtype)
        valid_len = min(W - 1, context.pre_num_conv_tokens)
        if valid_len > 0:
            # heads[-valid_len:]: [valid_len, D] -> [D, valid_len]
            initial_state[0, :, -valid_len:] = heads[-valid_len:].T
        return initial_state

    @staticmethod
    def _correct_dx_for_cp(
        dx: torch.Tensor,
        dh0: torch.Tensor | None,
        W: int,
        group: dist.ProcessGroup | None,
        is_first_rank: bool,
        pre_num_conv_tokens: int = 0,
    ) -> None:
        """Correct dx gradients for CP backward pass by communicating with next rank.

        Args:
            dx: Gradient tensor to be corrected, shape [1, T, D]
            dh0: Gradient w.r.t. initial_state, shape [N, D, W] or None
            W: Kernel size
            group: Process group for communication
            is_first_rank: Whether this is the first rank in the sequence's processing chain
            pre_num_conv_tokens: Number of tokens from the previous rank that
                belong to the first sequence on the current rank. Must match the
                value used in the forward pass to construct initial_state.
        """
        if group is None:
            return

        D = dx.shape[-1]
        # dh0: [N, D, W] or None
        # We only care about the first sequence's initial_state gradient
        if dh0 is not None:
            # Only keep gradients for positions that had real data from the
            # previous rank. The forward fills only the last valid_len positions
            # of initial_state; gradients for the remaining (zero-padded) positions
            # must not flow back, otherwise they leak into unrelated sequences.
            valid_len = min(W - 1, pre_num_conv_tokens)
            d_initial_state = torch.zeros(W - 1, D, device=dx.device, dtype=dx.dtype)
            if valid_len > 0:
                d_initial_state[-valid_len:] = dh0[0, :, -valid_len:].T
        else:
            # dh0 is None only when this is the first rank (no initial_state needed)
            assert is_first_rank, "dh0 should not be None when is_first_rank=False"
            d_initial_state = torch.zeros(W - 1, D, device=dx.device, dtype=dx.dtype)
        if dx.shape[1] < W - 1:
            gathered_gradients, _ = all_gather_into_tensor(d_initial_state, group=group)
            CausalConv1dFunctionCP._accumulate_multi_rank_halo_gradients(
                dx,
                gathered_gradients,
                rank=dist.get_rank(group),
            )
        else:
            # Add the next rank's initial-state gradient to the local tail.
            recv_d_init = conv_cp_send_recv_bwd(d_initial_state, group)
            dx[0, -(W - 1) :, :].add_(recv_d_init)

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        chunk_indices: torch.Tensor | None,
        cp_context: FLACPContext | None,
        chunk_size: int | None,
        backend: str = 'triton',
    ):
        # Import here to avoid circular dependency
        from fla.modules.conv.triton.ops import causal_conv1d_fwd

        if cp_context is None:
            raise ValueError("cp_context must be provided for CausalConv1dFunctionCP")
        cu_seqlens = cp_context.cu_seqlens
        cu_seqlens_cpu = cp_context.cu_seqlens_cpu
        group = cp_context.group

        # Get kernel_size
        W = weight.shape[-1]  # weight: [D, W]
        # Prepare initial_state for CP
        initial_state = CausalConv1dFunctionCP._prepare_initial_state_for_cp(
            x=x,
            weight=weight,
            cu_seqlens=cu_seqlens,
            context=cp_context,
            group=group,
        )

        ctx.save_for_backward(x, weight, bias, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.chunk_size = chunk_size
        ctx.group = group
        ctx.W = W
        ctx.is_first_rank = cp_context.is_first_rank
        ctx.pre_num_conv_tokens = cp_context.pre_num_conv_tokens

        # Call original forward
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            output_final_state=False,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            BT=chunk_size,
        )

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        # Import here to avoid circular dependency
        from fla.modules.conv.triton.ops import causal_conv1d_bwd

        x, weight, bias, initial_state = ctx.saved_tensors
        group = ctx.group
        W = ctx.W

        # Call original backward
        dx, dw, db, _, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=None,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=ctx.chunk_indices,
            BT=ctx.chunk_size,
        )

        # Correct dx gradients for CP
        CausalConv1dFunctionCP._correct_dx_for_cp(
            dx=dx,
            dh0=dh0,
            W=W,
            group=group,
            is_first_rank=ctx.is_first_rank,
            pre_num_conv_tokens=ctx.pre_num_conv_tokens,
        )

        return dx, dw, db, None, None, None, None, None


def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    chunk_indices: torch.Tensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_size: int | None = None,
    backend: str = 'triton',
):
    """
    Context Parallel version of causal_conv1d.

    Automatically handles communication in CP environment:
    - Forward: get initial_state from previous rank
    - Backward: correct dx gradients

    Args:
        x: Input tensor of shape [1, T, D]
        weight: Weight tensor of shape [D, W]
        bias: Bias tensor of shape [D] or None
        activation: Activation function name or None
        cu_seqlens: Cumulative sequence lengths
        cu_seqlens_cpu: Cumulative sequence lengths on CPU
        chunk_indices: Chunk indices for variable-length sequences
        cp_context: CP context (required for CP mode)
    """
    if cp_context is None:
        raise ValueError("cp_context must be provided for causal_conv1d_cp")

    assert cp_context.conv1d_kernel_size is not None, "conv1d_kernel_size must be provided for causal_conv1d_cp"
    assert cp_context.cu_seqlens is not None, "cu_seqlens must be provided for causal_conv1d_cp"
    assert backend in ['triton'], "backend must be 'triton'"
    chunk_size = chunk_size or 64
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cp_context.cu_seqlens, chunk_size, cu_seqlens_cpu=cp_context.cu_seqlens_cpu)

    return CausalConv1dFunctionCP.apply(
        x, weight, bias, activation,
        chunk_indices, cp_context, chunk_size, backend
    )
