# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Strict correctness gates for the Triton-Ascend Conv1d paths."""

import pytest
import torch
from conv_ascend_cases import CONV_DENSE_CASES, CONV_VARLEN_CASES, ConvDenseCase, ConvVarlenCase

from fla.modules.convolution import causal_conv1d
from fla.utils import IS_NPU, device

pytestmark = [
    pytest.mark.skipif(not IS_NPU, reason="Triton-Ascend kernel test"),
    pytest.mark.ascend_npu,
]

# The initial-state gradient crosses the low-precision kernel boundary.
_DINITIAL_STATE_RATIO = 3.1e-3


def _varlen_case_param(case: ConvVarlenCase):
    marks = (pytest.mark.nightly,) if case.nightly else ()
    return pytest.param(case, marks=marks, id=case.name)


def _dense_case_param(case: ConvDenseCase):
    marks = (pytest.mark.nightly,) if case.nightly else ()
    return pytest.param(case, marks=marks, id=case.name)


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
        out = torch.nn.functional.silu(out)
    if residual is not None:
        out += residual.float()
    return out.to(x.dtype)


def _reference_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    activation: str | None,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    """Apply the shifted-tensor reference independently to every packed sequence."""
    boundaries = cu_seqlens_cpu.tolist()
    W = weight.shape[1]
    weight32 = weight.float()
    bias32 = None if bias is None else bias.float()
    x32 = x.float()
    residual32 = None if residual is None else residual.float()
    initial_state32 = None if initial_state is None else initial_state.float()
    outputs = []
    for i_n, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        sequence = x32[:, bos:eos]
        length = eos - bos
        output = torch.zeros(1, length, x.shape[-1], dtype=torch.float32, device=x.device)
        for tap in range(W):
            shift = W - 1 - tap
            if shift == 0:
                source = sequence
            else:
                prefix = (
                    torch.zeros(1, shift, x.shape[-1], dtype=torch.float32, device=x.device)
                    if initial_state32 is None
                    else initial_state32[i_n : i_n + 1, :, tap + 1 :].transpose(1, 2)
                )
                source = torch.cat((prefix, sequence), dim=1)[:, :length]
            output += source * weight32[:, tap][None, None, :]
        if bias32 is not None:
            output += bias32[None, None, :]
        if activation in ("silu", "swish"):
            output = torch.nn.functional.silu(output)
        if residual32 is not None:
            output += residual32[:, bos:eos]
        outputs.append(output)
    return torch.cat(outputs, dim=1).to(x.dtype)


def _reference_final_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor,
    W: int,
) -> torch.Tensor:
    """Build each final cache by taking the last W values of history plus sequence."""
    boundaries = cu_seqlens_cpu.tolist()
    states = []
    D = x.shape[-1]
    for i_n, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        history = (
            torch.zeros(W, D, dtype=x.dtype, device=x.device) if initial_state is None else initial_state[i_n].transpose(0, 1)
        )
        states.append(torch.cat((history, x[0, bos:eos]), dim=0)[-W:].transpose(0, 1))
    return torch.stack(states)


def _assert_strict_close(
    reference: torch.Tensor,
    actual: torch.Tensor,
    ratio: float = 1e-3,
    abs_atol: float = 1e-6,
    name: str = "tensor",
) -> None:
    assert torch.isfinite(reference).all().item(), f"{name} reference is non-finite"
    assert torch.isfinite(actual).all().item(), f"{name} result is non-finite"
    reference = reference.float()
    actual = actual.float()
    max_abs = (reference - actual).abs().max().item()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    error_ratio = (rms / (base + 1e-8)).item()
    assert max_abs <= abs_atol or error_ratio < ratio, (
        f"{name}: max_abs={max_abs:.6g}, abs_limit={abs_atol:.6g}, RMS ratio={error_ratio:.6g}, ratio_limit={ratio:.6g}"
    )


@pytest.mark.parametrize("case", [_dense_case_param(case) for case in CONV_DENSE_CASES])
def test_dense_forward(case: ConvDenseCase):
    torch.manual_seed(42)
    B, T, D, W, dtype = case.B, case.T, case.D, case.W, case.dtype
    x = torch.randn(B, T, D, dtype=dtype, device=device) * 0.1
    weight = torch.randn(D, W, dtype=dtype, device=device) * 0.1
    bias = torch.randn(D, dtype=dtype, device=device) * 0.1 if case.has_bias else None
    residual = torch.randn_like(x) * 0.1 if case.has_residual else None
    initial_state = torch.randn(B, D, W, dtype=dtype, device=device) * 0.1 if case.has_state else None
    cu_seqlens_cpu = torch.tensor([0, T], dtype=torch.int32) if B == 1 else None
    cu_seqlens = None if cu_seqlens_cpu is None else cu_seqlens_cpu.to(device)

    reference = _reference(x, weight, bias, residual, initial_state, case.activation)
    actual, _ = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        activation=case.activation,
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


@pytest.mark.parametrize("case", [_dense_case_param(case) for case in CONV_DENSE_CASES])
def test_dense_backward(case: ConvDenseCase):
    torch.manual_seed(42)
    B, T, D, W, dtype = case.B, case.T, case.D, case.W, case.dtype
    inputs = {
        "x": torch.randn(B, T, D, dtype=dtype, device=device) * 0.1,
        "weight": torch.randn(D, W, dtype=dtype, device=device) * 0.1,
    }
    if case.has_bias:
        inputs["bias"] = torch.randn(D, dtype=dtype, device=device) * 0.1
    if case.has_state:
        inputs["initial_state"] = torch.randn(B, D, W, dtype=dtype, device=device) * 0.1
    if case.has_residual:
        inputs["residual"] = torch.randn(B, T, D, dtype=dtype, device=device) * 0.1
    do = torch.randn(B, T, D, dtype=dtype, device=device) * 0.1

    reference_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    reference = _reference(
        x=reference_inputs["x"],
        weight=reference_inputs["weight"],
        bias=reference_inputs.get("bias"),
        residual=reference_inputs.get("residual"),
        initial_state=reference_inputs.get("initial_state"),
        activation=case.activation,
    )
    reference.backward(do)

    actual_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}
    actual, _ = causal_conv1d(**actual_inputs, activation=case.activation)
    actual.backward(do)

    _assert_strict_close(reference, actual, name="output")
    for name in inputs:
        ratio = _DINITIAL_STATE_RATIO if name == "initial_state" else 1e-3
        _assert_strict_close(reference_inputs[name].grad, actual_inputs[name].grad, ratio=ratio, name=f"d{name}")


@pytest.mark.parametrize("case", [_varlen_case_param(case) for case in CONV_VARLEN_CASES])
def test_varlen_forward_backward_matches_independent_reference(case: ConvVarlenCase):
    """Exercise the general packed path, all public gradients, and final caches."""
    torch.manual_seed(20260715)
    T, D, W = sum(case.lengths), case.D, case.W
    N = len(case.lengths)

    if case.noncontiguous_x:
        x_storage = torch.randn(1, T, D * 2, dtype=case.dtype, device=device) * 0.05
        x_data = x_storage[..., ::2]
        assert not x_data.is_contiguous()
    else:
        x_data = torch.randn(1, T, D, dtype=case.dtype, device=device) * 0.05
    inputs = {
        "x": x_data,
        "weight": torch.randn(D, W, dtype=case.dtype, device=device) * 0.05,
    }
    if case.has_bias:
        inputs["bias"] = torch.randn(D, dtype=case.dtype, device=device) * 0.05
    if case.has_state:
        inputs["initial_state"] = torch.randn(N, D, W, dtype=case.dtype, device=device) * 0.05
    if case.has_residual:
        inputs["residual"] = torch.randn_like(x_data) * 0.05

    boundaries = [0]
    for length in case.lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens_cpu = torch.tensor(boundaries, dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)

    if case.noncontiguous_dy:
        dy_storage = torch.randn(1, T, D * 2, dtype=case.dtype, device=device) * 0.05
        dy = dy_storage[..., ::2]
        assert not dy.is_contiguous()
    else:
        dy = torch.randn(1, T, D, dtype=case.dtype, device=device) * 0.05

    reference_inputs = {name: value.detach().clone().requires_grad_(True) for name, value in inputs.items()}
    reference = _reference_varlen(
        x=reference_inputs["x"],
        weight=reference_inputs["weight"],
        bias=reference_inputs.get("bias"),
        residual=reference_inputs.get("residual"),
        initial_state=reference_inputs.get("initial_state"),
        activation=case.activation,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    reference_final_state = _reference_final_state(
        reference_inputs["x"],
        reference_inputs.get("initial_state"),
        cu_seqlens_cpu,
        W,
    )
    reference.backward(dy)

    actual_inputs = {name: value.detach().requires_grad_(True) for name, value in inputs.items()}
    actual, actual_final_state = causal_conv1d(
        **actual_inputs,
        activation=case.activation,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    actual.backward(dy)

    assert actual_final_state is not None
    _assert_strict_close(reference, actual, name="varlen output")
    _assert_strict_close(reference_final_state, actual_final_state, name="varlen final state")
    for name in inputs:
        ratio = _DINITIAL_STATE_RATIO if name == "initial_state" else 1e-3
        abs_atol = 2.5e-4 if case.dtype == torch.bfloat16 and name in ("weight", "bias") else 1e-6
        _assert_strict_close(
            reference_inputs[name].grad,
            actual_inputs[name].grad,
            ratio=ratio,
            abs_atol=abs_atol,
            name=f"varlen d{name}",
        )


def test_varlen_impulses_freeze_sequence_and_tap_order():
    """Use exactly representable impulses to expose boundary, channel, and tap reversal bugs."""
    boundaries = (0, 4, 7, 12)
    T, D = boundaries[-1], 17
    dtype = torch.bfloat16
    x = torch.zeros(1, T, D, dtype=dtype, device=device)
    for token, value in ((0, 1.0), (3, 2.0), (4, 4.0), (6, 8.0), (7, 16.0), (11, 32.0)):
        x[0, token, -1] = value
    weight = torch.tensor((1.0, 2.0, 4.0, 8.0), dtype=dtype, device=device).repeat(D, 1)
    dy = torch.zeros_like(x)
    for token in (0, 4, 7, 11):
        dy[0, token, -1] = 1.0
    cu_seqlens_cpu = torch.tensor(boundaries, dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)

    reference_x = x.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = _reference_varlen(
        reference_x,
        reference_weight,
        None,
        None,
        None,
        None,
        cu_seqlens_cpu,
    )
    reference.backward(dy)

    actual_x = x.detach().clone().requires_grad_(True)
    actual_weight = weight.detach().clone().requires_grad_(True)
    actual, _ = causal_conv1d(
        actual_x,
        actual_weight,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    actual.backward(dy)

    _assert_strict_close(reference, actual, ratio=1e-7, name="impulse output")
    _assert_strict_close(reference_x.grad, actual_x.grad, ratio=1e-7, name="impulse dx")
    _assert_strict_close(reference_weight.grad, actual_weight.grad, ratio=1e-7, name="impulse dw")


def test_varlen_general_outputs_are_fully_written_from_nan():
    import fla.modules.backends.triton_ascend.causal_conv1d as conv_backend
    from fla.ops.utils import prepare_chunk_indices

    torch.manual_seed(20260715)
    lengths = (1, 2, 3, 4)
    boundaries = (0, 1, 3, 6, 10)
    B, T, D, W = 1, boundaries[-1], 129, 4
    x = torch.randn(B, T, D, dtype=torch.bfloat16, device=device) * 0.05
    weight = torch.randn(D, W, dtype=torch.bfloat16, device=device) * 0.05
    bias = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.05
    residual = torch.randn_like(x) * 0.05
    initial_state = torch.randn(len(lengths), D, W, dtype=torch.bfloat16, device=device) * 0.05
    cu_seqlens_cpu = torch.tensor(boundaries, dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(device)
    BD, BT, num_warps = conv_backend._npu_tile_config(T, 64, D, x.dtype, initial_state)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)

    output = conv_backend._launch_fwd_core(
        x,
        weight,
        bias,
        initial_state,
        cu_seqlens,
        chunk_indices,
        B,
        T,
        D,
        W,
        BT,
        BD,
        num_warps,
        residual,
        "silu",
        poison=True,
    )
    reference = _reference_varlen(
        x,
        weight,
        bias,
        residual,
        initial_state,
        "silu",
        cu_seqlens_cpu,
    )
    _assert_strict_close(reference, output, name="poisoned general output")

    final_state = conv_backend.causal_conv1d_update_states_npu(
        x,
        W,
        initial_state,
        cu_seqlens,
        poison=True,
    )
    _assert_strict_close(
        _reference_final_state(x, initial_state, cu_seqlens_cpu, W),
        final_state,
        name="poisoned final state",
    )

    y_pre = conv_backend._launch_fwd_core(
        x,
        weight,
        bias,
        initial_state,
        cu_seqlens,
        chunk_indices,
        B,
        T,
        D,
        W,
        BT,
        BD,
        num_warps,
        output_dtype=torch.float32,
        poison=True,
    )
    dy = torch.randn_like(x) * 0.05
    dpre = conv_backend._launch_silu_bwd(y_pre, dy, poison=True)
    assert torch.isfinite(y_pre).all().item()
    assert torch.isfinite(dpre).all().item()


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


def test_dense_selector_rejects_packed_and_noncontiguous_inputs():
    from fla.modules.backends.triton_ascend.causal_conv1d import _is_dense_single_sequence

    T, D, W = 10, 129, 4
    x = torch.empty(1, T, D, dtype=torch.bfloat16, device=device)
    weight = torch.empty(D, W, dtype=torch.bfloat16, device=device)
    single_cpu = torch.tensor((0, T), dtype=torch.int32)
    packed_cpu = torch.tensor((0, 1, 3, 6, T), dtype=torch.int32)
    assert _is_dense_single_sequence(x, weight, None, None, None, single_cpu.to(device), single_cpu)
    assert not _is_dense_single_sequence(x, weight, None, None, None, packed_cpu.to(device), packed_cpu)

    storage = torch.empty(1, T, D * 2, dtype=torch.bfloat16, device=device)
    noncontiguous_x = storage[..., ::2]
    assert not _is_dense_single_sequence(noncontiguous_x, weight, None, None, None, None, None)


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

    reference = _reference(
        x=x_data,
        weight=weight_data,
        bias=bias_data,
        residual=None,
        initial_state=initial_state_data,
        activation="silu",
    )

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


@pytest.mark.parametrize("layout", ["nd", "1nd", "n1d"])
def test_incremental_update_split(monkeypatch, layout):
    import fla.modules.backends.triton_ascend.causal_conv1d as conv_backend

    torch.manual_seed(42)
    N, D, W = 3, 65, 4
    x_2d = torch.randn(N, D, dtype=torch.bfloat16, device=device) * 0.1
    if layout == "nd":
        x = x_2d
    elif layout == "1nd":
        x = x_2d.unsqueeze(0)
    else:
        x = x_2d.unsqueeze(1)
    cache = torch.randn(N, D, W, dtype=torch.bfloat16, device=device) * 0.1
    weight = torch.randn(D, W, dtype=torch.bfloat16, device=device) * 0.1
    bias = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.1
    residual = torch.randn_like(x) * 0.1

    expected_cache = torch.cat((cache[:, :, 1:], x_2d[:, :, None]), dim=-1)
    expected_pre = torch.zeros(N, D, dtype=torch.float32, device=device)
    for tap in range(W):
        expected_pre += expected_cache[:, :, tap].float() * weight[:, tap].float()[None, :]
    expected_pre += bias.float()[None, :]
    expected_pre = expected_pre.to(x.dtype).float()
    expected_output = (expected_pre * torch.sigmoid(expected_pre)).to(x.dtype)
    expected_output = (expected_output.reshape(x.shape).float() + residual.float()).to(x.dtype)

    def run(max_grid: int):
        monkeypatch.setattr(conv_backend, "_get_npu_max_grid", lambda: max_grid)
        guard = 31
        sentinel = 123.0
        storage = torch.full((guard + cache.numel() + guard,), sentinel, dtype=cache.dtype, device=device)
        cache_view = storage[guard:-guard].view_as(cache)
        cache_view.copy_(cache)
        output, updated_cache = conv_backend.causal_conv1d_update_npu(
            x=x,
            cache=cache_view,
            residual=residual,
            weight=weight,
            bias=bias,
            activation="silu",
        )
        assert torch.equal(storage[:guard], torch.full_like(storage[:guard], sentinel))
        assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], sentinel))
        return output, updated_cache

    baseline_output, baseline_cache = run(conv_backend._NPU_MAX_TRITON_GRID)
    split_output, split_cache = run(4)
    _assert_strict_close(baseline_output, split_output, ratio=1e-7, name="split update output")
    _assert_strict_close(baseline_cache, split_cache, ratio=1e-7, name="split update cache")
    _assert_strict_close(expected_output, split_output, name="update output reference")
    _assert_strict_close(expected_cache, split_cache, ratio=1e-7, name="update cache reference")
