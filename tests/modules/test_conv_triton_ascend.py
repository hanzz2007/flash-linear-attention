# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Strict correctness gates for the Triton-Ascend Conv1d paths."""

import pytest
import torch

from fla.modules.convolution import causal_conv1d
from fla.utils import IS_NPU, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason="Triton-Ascend kernel test")

# A800 BF16 baseline is 2.77e-3 at the target shape; allow the frozen 1.10x envelope.
_DINITIAL_STATE_RATIO = 3.1e-3


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
        if shift == 0:
            source = x32
        else:
            if initial_state is not None:
                prefix = initial_state[:, :, tap + 1 :].transpose(1, 2).float()
            else:
                prefix = torch.zeros(B, shift, D, dtype=torch.float32, device=x.device)
            source = torch.cat((prefix, x32), dim=1)[:, :T]
        out = out + source * weight[:, tap].float()[None, None, :]
    if bias is not None:
        out += bias.float()[None, None, :]
    if activation in ("silu", "swish"):
        out *= torch.sigmoid(out)
    if residual is not None:
        out += residual.float()
    return out.to(x.dtype)


def _assert_strict_close(
    reference: torch.Tensor,
    actual: torch.Tensor,
    ratio: float = 1e-3,
    name: str = "tensor",
) -> None:
    assert torch.isfinite(reference).all().item()
    assert torch.isfinite(actual).all().item()
    reference = reference.float()
    actual = actual.float()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    error_ratio = (rms / (base + 1e-8)).item()
    assert error_ratio < ratio, f"{name} RMS ratio {error_ratio:.6g} exceeds {ratio:.6g}"


@pytest.mark.parametrize(
    ("T", "D", "W", "dtype", "has_bias", "activation", "has_state", "has_residual"),
    [
        (1, 128, 2, torch.bfloat16, False, None, False, False),
        (65, 1001, 3, torch.bfloat16, True, "silu", True, True),
        (257, 1024, 4, torch.float16, False, None, True, False),
        pytest.param(2048, 3072, 4, torch.bfloat16, True, "silu", True, False, id="target"),
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
    output = torch.full_like(x, float("nan"), memory_format=torch.contiguous_format)

    reference = _reference(x, weight, bias, None, initial_state, "silu")
    actual = _launch_fwd_dense(x, weight, bias, None, initial_state, "silu", output=output)
    _assert_strict_close(reference, actual)


@pytest.mark.parametrize(
    ("T", "D", "W", "dtype", "has_bias", "activation", "has_state"),
    [
        (2, 128, 4, torch.bfloat16, False, None, True),
        (65, 1001, 3, torch.bfloat16, True, "silu", True),
        (257, 1024, 2, torch.float16, False, None, False),
        pytest.param(2048, 3072, 4, torch.bfloat16, True, "silu", True, id="target"),
    ],
)
def test_dense_backward(T, D, W, dtype, has_bias, activation, has_state):
    torch.manual_seed(42)
    inputs = {
        "x": torch.randn(1, T, D, dtype=dtype, device=device) * 0.1,
        "weight": torch.randn(D, W, dtype=dtype, device=device) * 0.1,
    }
    if has_bias:
        inputs["bias"] = torch.randn(D, dtype=dtype, device=device) * 0.1
    if has_state:
        inputs["initial_state"] = torch.randn(1, D, W, dtype=dtype, device=device) * 0.1
    do = torch.randn(1, T, D, dtype=dtype, device=device) * 0.1

    reference_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    reference = _reference(
        x=reference_inputs["x"],
        weight=reference_inputs["weight"],
        bias=reference_inputs.get("bias"),
        residual=None,
        initial_state=reference_inputs.get("initial_state"),
        activation=activation,
    )
    reference.backward(do)

    actual_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    actual, _ = causal_conv1d(**actual_inputs, activation=activation)
    actual.backward(do)

    _assert_strict_close(reference, actual, name="output")
    for name in inputs:
        ratio = _DINITIAL_STATE_RATIO if name == "initial_state" else 1e-3
        _assert_strict_close(reference_inputs[name].grad, actual_inputs[name].grad, ratio=ratio, name=f"d{name}")


@pytest.mark.parametrize("W", [3, 4])
def test_dense_backward_nan_poisoning(W):
    from fla.modules.backends.triton_ascend.causal_conv1d import _launch_bwd_dense

    torch.manual_seed(42)
    T, D = 65, 1001
    x = torch.randn(1, T, D, dtype=torch.bfloat16, device=device) * 0.1
    dy = torch.randn_like(x) * 0.1
    weight = torch.randn(D, W, dtype=torch.bfloat16, device=device) * 0.1
    bias = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.1
    initial_state = torch.randn(1, D, W, dtype=torch.bfloat16, device=device) * 0.1

    outputs = _launch_bwd_dense(x, dy, weight, bias, initial_state, "silu", poison=True)
    for name, tensor in zip(("dx", "dw", "db", "dh0", "dpre"), outputs, strict=True):
        assert tensor is None or torch.isfinite(tensor).all().item(), f"{name} contains an unwritten or non-finite value"


def test_dense_precision_selector(monkeypatch):
    """The switch is validated and unpromoted shapes retain high precision."""
    from fla.modules.backends.triton_ascend.causal_conv1d import (
        _ascend_conv_precision_mode,
        _dense_backward_num_warps,
        _dense_backward_precision_mode,
    )

    x = torch.empty(1, 64, 1024, dtype=torch.bfloat16, device=device)
    weight = torch.empty(1024, 4, dtype=torch.bfloat16, device=device)

    monkeypatch.setenv("FLA_ASCEND_CONV_PRECISION", "a800")
    assert _ascend_conv_precision_mode() == "a800"
    assert _dense_backward_precision_mode(x, weight) == "high"
    assert _dense_backward_num_warps(x, weight) == 4

    monkeypatch.setenv("FLA_ASCEND_CONV_PRECISION", "invalid")
    with pytest.raises(ValueError, match="FLA_ASCEND_CONV_PRECISION"):
        _ascend_conv_precision_mode()


def test_general_grid_splits_cover_each_task_once(monkeypatch):
    import fla.modules.backends.triton_ascend.causal_conv1d as conv_backend

    monkeypatch.setattr(conv_backend, "_get_npu_max_grid", lambda: 7)
    B, NT, D, BD = 5, 4, 73, 8
    DB = (D + BD - 1) // BD
    covered = set()

    for b_start, b_count, nt_start, nt_count, d_start, d_count in conv_backend._iter_3d_grid_splits(B, NT, D, BD):
        assert b_count * nt_count * d_count <= 7
        for b in range(b_start, b_start + b_count):
            for nt in range(nt_start, nt_start + nt_count):
                for d_block in range(d_start, d_start + d_count):
                    task = b, nt, d_block
                    assert task not in covered
                    covered.add(task)

    assert covered == {(b, nt, d_block) for b in range(B) for nt in range(NT) for d_block in range(DB)}
    assert any(b_start > 0 for b_start, *_ in conv_backend._iter_3d_grid_splits(B, NT, D, BD))
    assert any(nt_start > 0 for _, _, nt_start, *_ in conv_backend._iter_3d_grid_splits(B, NT, D, BD))
    assert any(d_start > 0 for *_, d_start, _ in conv_backend._iter_3d_grid_splits(B, NT, D, BD))


def test_short_sequence_backward_respects_grid_limit(monkeypatch):
    import fla.modules.backends.triton_ascend.causal_conv1d as conv_backend

    monkeypatch.setattr(conv_backend, "_get_npu_max_grid", lambda: 2)
    assert conv_backend._use_seq_bwd(1, 16, 128, torch.bfloat16, None, None, None)
    assert not conv_backend._use_seq_bwd(2, 16, 128, torch.bfloat16, None, None, None)


def test_general_fallback_split_forward_backward(monkeypatch):
    import fla.modules.backends.triton_ascend.causal_conv1d as conv_backend

    torch.manual_seed(42)
    B, T, D, W = 2, 65, 33, 4
    x_storage = torch.randn(B, T, D * 2, dtype=torch.bfloat16, device=device) * 0.1
    x_data = x_storage[..., ::2]
    weight_data = torch.randn(D, W, dtype=torch.bfloat16, device=device) * 0.1
    bias_data = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.1
    initial_state_data = torch.randn(B, D, W, dtype=torch.bfloat16, device=device) * 0.1
    do = torch.randn(B, T, D, dtype=torch.bfloat16, device=device) * 0.1

    reference_pre = _reference(
        x=x_data,
        weight=weight_data,
        bias=bias_data,
        residual=None,
        initial_state=initial_state_data,
        activation=None,
    )
    reference = (reference_pre.float() * torch.sigmoid(reference_pre.float())).to(reference_pre.dtype)

    def run(max_grid: int):
        monkeypatch.setattr(conv_backend, "_get_npu_max_grid", lambda: max_grid)
        storage = x_storage.detach().clone()
        inputs = {
            "x": storage[..., ::2].detach().requires_grad_(True),
            "weight": weight_data.detach().clone().requires_grad_(True),
            "bias": bias_data.detach().clone().requires_grad_(True),
            "initial_state": initial_state_data.detach().clone().requires_grad_(True),
        }
        assert not inputs["x"].is_contiguous()
        output, final_state = causal_conv1d(**inputs, activation="silu", output_final_state=True)
        output.backward(do)
        return output.detach(), final_state.detach(), {name: tensor.grad.detach() for name, tensor in inputs.items()}

    baseline_output, baseline_final_state, baseline_gradients = run(conv_backend._NPU_MAX_TRITON_GRID)
    split_output, split_final_state, split_gradients = run(4)

    _assert_strict_close(reference, baseline_output, name="fallback output")
    _assert_strict_close(baseline_output, split_output, ratio=1e-7, name="split output")
    _assert_strict_close(baseline_final_state, split_final_state, ratio=1e-7, name="split final state")
    for name in baseline_gradients:
        _assert_strict_close(baseline_gradients[name], split_gradients[name], ratio=1e-7, name=f"split d{name}")


def test_incremental_update_split(monkeypatch):
    import fla.modules.backends.triton_ascend.causal_conv1d as conv_backend

    torch.manual_seed(42)
    N, D, W = 3, 65, 4
    x = torch.randn(N, D, dtype=torch.bfloat16, device=device) * 0.1
    cache = torch.randn(N, D, W, dtype=torch.bfloat16, device=device) * 0.1
    weight = torch.randn(D, W, dtype=torch.bfloat16, device=device) * 0.1
    bias = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.1
    residual = torch.randn_like(x) * 0.1

    expected_cache = torch.cat((cache[:, :, 1:], x[:, :, None]), dim=-1)

    def run(max_grid: int):
        monkeypatch.setattr(conv_backend, "_get_npu_max_grid", lambda: max_grid)
        return conv_backend.causal_conv1d_update_npu(
            x=x,
            cache=cache.clone(),
            residual=residual,
            weight=weight,
            bias=bias,
            activation="silu",
        )

    baseline_output, baseline_cache = run(conv_backend._NPU_MAX_TRITON_GRID)
    split_output, split_cache = run(4)
    _assert_strict_close(baseline_output, split_output, ratio=1e-7, name="split update output")
    _assert_strict_close(baseline_cache, split_cache, ratio=1e-7, name="split update cache")
    _assert_strict_close(expected_cache, split_cache, ratio=1e-7, name="update cache reference")
