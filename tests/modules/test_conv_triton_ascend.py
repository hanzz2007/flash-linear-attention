# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Strict correctness gates for the dense Triton-Ascend Conv1d path."""

import pytest
import torch

from fla.modules.convolution import causal_conv1d
from fla.utils import IS_NPU, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason='Triton-Ascend kernel test')


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    """Independent FP32 causal depthwise convolution using shifted tensors."""
    B, T, D = x.shape
    W = weight.shape[1]
    out = torch.zeros(B, T, D, dtype=torch.float32, device=x.device)
    x32 = x.float()
    for tap in range(W):
        shift = W - 1 - tap
        source = torch.zeros_like(out)
        if shift == 0:
            source = x32
        else:
            if shift < T:
                source[:, shift:] = x32[:, :T - shift]
            if initial_state is not None:
                head = min(shift, T)
                source[:, :head] = initial_state[:, :, tap + 1:tap + 1 + head].transpose(1, 2).float()
        out += source * weight[:, tap].float()[None, None, :]
    if bias is not None:
        out += bias.float()[None, None, :]
    if activation in ('silu', 'swish'):
        out *= torch.sigmoid(out)
    if residual is not None:
        out += residual.float()
    return out.to(x.dtype)


def _assert_strict_close(reference: torch.Tensor, actual: torch.Tensor, ratio: float = 1e-3) -> None:
    assert torch.isfinite(reference).all().item()
    assert torch.isfinite(actual).all().item()
    reference = reference.float()
    actual = actual.float()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    error_ratio = (rms / (base + 1e-8)).item()
    assert error_ratio < ratio, f'RMS ratio {error_ratio:.6g} exceeds {ratio:.6g}'


@pytest.mark.parametrize(
    ('T', 'D', 'W', 'dtype', 'has_bias', 'activation', 'has_state', 'has_residual'),
    [
        (1, 128, 2, torch.bfloat16, False, None, False, False),
        (65, 1001, 3, torch.bfloat16, True, 'silu', True, True),
        (257, 1024, 4, torch.float16, False, None, True, False),
        pytest.param(2048, 3072, 4, torch.bfloat16, True, 'silu', True, False, id='target'),
    ],
)
def test_dense_forward(T, D, W, dtype, has_bias, activation, has_state, has_residual):
    torch.manual_seed(42)
    x = torch.randn(1, T, D, dtype=dtype, device=device) * 0.1
    weight = torch.randn(D, W, dtype=dtype, device=device) * 0.1
    bias = torch.randn(D, dtype=dtype, device=device) * 0.1 if has_bias else None
    residual = torch.randn_like(x) * 0.1 if has_residual else None
    initial_state = torch.randn(1, D, W, dtype=dtype, device=device) * 0.1 if has_state else None
    cu_seqlens_cpu = torch.tensor([0, T], dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)

    reference = _reference(x, weight, bias, residual, initial_state, activation)
    actual, _ = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    _assert_strict_close(reference, actual)


def test_dense_forward_nan_poisoning():
    from fla.modules.backends.triton_ascend.causal_conv1d import _launch_fwd_dense

    torch.manual_seed(42)
    T, D, W = 65, 1001, 4
    x = torch.randn(1, T, D, dtype=torch.bfloat16, device=device) * 0.1
    weight = torch.randn(D, W, dtype=torch.bfloat16, device=device) * 0.1
    bias = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.1
    initial_state = torch.randn(1, D, W, dtype=torch.bfloat16, device=device) * 0.1
    output = torch.full_like(x, float('nan'), memory_format=torch.contiguous_format)

    reference = _reference(x, weight, bias, None, initial_state, 'silu')
    actual = _launch_fwd_dense(x, weight, bias, None, initial_state, 'silu', output=output)
    _assert_strict_close(reference, actual)
