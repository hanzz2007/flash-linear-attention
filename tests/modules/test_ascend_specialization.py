# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compile-cache contracts for shape-polymorphic Triton-Ascend kernels."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from fla.utils import IS_NPU, device, device_torch_lib


@dataclass(frozen=True)
class SpecializationContract:
    path: str
    kernel: str
    runtime: frozenset[str]
    constexpr: frozenset[str]


_ROOT = Path(__file__).resolve().parents[2]
_GDN_CP = "fla/ops/cp/backends/triton_ascend/chunk_delta_h.py"
_GDN_H = "fla/ops/common/backends/triton_ascend/chunk_delta_h.py"
_GDN_GATE = "fla/ops/gated_delta_rule/backends/triton_ascend/gate.py"
_GDN_WY = "fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py"
_GDN_KKT = "fla/ops/common/backends/triton_ascend/chunk_scaled_dot_kkt.py"
_GDN_O = "fla/ops/common/backends/triton_ascend/chunk_o.py"
_GDN_CUMSUM = "fla/ops/utils/backends/triton_ascend/cumsum.py"
_GDN_SOLVE = "fla/ops/utils/backends/triton_ascend/solve_tril.py"
_CONV = "fla/modules/backends/triton_ascend/causal_conv1d.py"
_BLOCK_POINTER_I32 = {
    _GDN_H: frozenset({"V_OFFSET"}),
    _GDN_GATE: frozenset({"NT_OFFSET"}),
    _GDN_WY: frozenset({"NT_OFFSET"}),
    _GDN_KKT: frozenset({"NT_OFFSET"}),
    _GDN_O: frozenset({"V_OFFSET", "K_OFFSET", "NT_OFFSET"}),
    _GDN_CUMSUM: frozenset({"NT_OFFSET"}),
    _GDN_SOLVE: frozenset({"NT_OFFSET"}),
}

# Start with the GDN kernels that already satisfy the runtime-value contract.
# Later optimization stages extend this manifest before changing each kernel family.
SPECIALIZATION_CONTRACTS = (
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_gate_factors_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "TASK_OFFSET"}),
        constexpr=frozenset({"BT"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_bwd_gate_factors_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "TASK_OFFSET"}),
        constexpr=frozenset({"BT"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_fwd_h_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "TASK_OFFSET"}),
        constexpr=frozenset({"K", "V", "BT", "BV"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_fwd_m_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "TASK_OFFSET"}),
        constexpr=frozenset({"K", "BT", "BM", "NM"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_bwd_dh_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "scale", "TASK_OFFSET"}),
        constexpr=frozenset({"K", "V", "BT", "BV"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_bwd_m_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "TASK_OFFSET"}),
        constexpr=frozenset({"K", "BT", "BM", "NM"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_bwd_fused_128_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "scale", "TASK_OFFSET"}),
        constexpr=frozenset({"BT", "PRECOMPUTED_GATE", "A800_PRECISION"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_merge_one_rank_kernel",
        runtime=frozenset({"SOURCE_RANK", "TASK_OFFSET"}),
        constexpr=frozenset({"K", "V", "BR", "BV"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_merge_rank_chain_kernel",
        runtime=frozenset({"SOURCE_START", "SOURCE_STEP", "TASK_OFFSET"}),
        constexpr=frozenset({"K", "V", "BV", "NUM_RANKS"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_transpose_state_kernel",
        runtime=frozenset({"TASK_OFFSET"}),
        constexpr=frozenset({"K", "V", "BK", "BV"}),
    ),
) + tuple(
    SpecializationContract(
        path=path,
        kernel=kernel,
        runtime=frozenset(runtime),
        constexpr=frozenset(constexpr),
    )
    for path, kernels, runtime, constexpr in (
        (
            _GDN_H,
            (
                "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu",
                "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu",
            ),
            {"T", "V_OFFSET", "NH_OFFSET"},
            {"BT", "BV"},
        ),
        (
            _GDN_GATE,
            ("gdn_gate_fwd_kernel_npu", "gdn_gate_bwd_kernel_npu"),
            {"T", "NT_OFFSET", "H_OFFSET"},
            {"BT"},
        ),
        (
            _GDN_GATE,
            ("gdn_gate_chunk_cumsum_scalar_kernel_npu",),
            {"T", "NT_OFFSET", "BH_OFFSET"},
            {"BT"},
        ),
        (
            _GDN_WY,
            (
                "recompute_w_u_fwd_kernel_npu",
                "prepare_wy_repr_bwd_k_npu",
                "prepare_wy_repr_bwd_v_npu",
                "prepare_wy_repr_bwd_da_mask_npu",
                "prepare_wy_repr_bwd_da_dot1_npu",
                "prepare_wy_repr_bwd_da_dot2_npu",
                "prepare_wy_repr_bwd_da_gate_npu",
                "prepare_wy_repr_bwd_finalize_k_npu",
                "prepare_wy_repr_bwd_finalize_a2_npu",
                "prepare_wy_repr_bwd_finalize_dg_npu",
            ),
            {"T", "NT_OFFSET", "BH_OFFSET"},
            {"BT"},
        ),
        (
            _GDN_KKT,
            ("chunk_scaled_dot_kkt_fwd_kernel_npu",),
            {"T", "NT_OFFSET", "BH_OFFSET"},
            {"BT", "BK"},
        ),
        (
            _GDN_O,
            (
                "chunk_fwd_kernel_o_inter_npu",
                "chunk_fwd_kernel_o_fused_hv1_npu",
                "chunk_fwd_kernel_o_intra_hv1_npu",
                "chunk_fwd_kernel_o_intra_npu",
            ),
            {"T", "V_OFFSET", "NT_OFFSET", "BH_OFFSET"},
            {"BT", "BV"},
        ),
        (
            _GDN_O,
            ("chunk_bwd_kernel_dv_local_hv1_npu", "chunk_bwd_kernel_dv_local_npu"),
            {"T", "NT_OFFSET", "BH_OFFSET"},
            {"BT", "BV"},
        ),
        (
            _GDN_O,
            ("chunk_bwd_kernel_dqkwg_npu", "chunk_bwd_kernel_dg_npu"),
            {"B", "T", "K_OFFSET", "NT_OFFSET", "BH_OFFSET"},
            {"BT", "BK"},
        ),
        (
            _GDN_CUMSUM,
            ("chunk_local_cumsum_scalar_kernel_npu", "chunk_local_cumsum_vector_kernel_npu"),
            {"T", "NT_OFFSET", "BH_OFFSET"},
            {"BT"},
        ),
        (
            _GDN_CUMSUM,
            ("chunk_global_cumsum_scalar_kernel_npu", "chunk_global_cumsum_vector_kernel_npu"),
            {"T", "BH_OFFSET"},
            {"BT"},
        ),
        (
            _GDN_SOLVE,
            (
                "solve_tril_16x16_kernel_npu",
                "merge_16x16_to_32x32_inverse_kernel_npu",
                "merge_16x16_to_64x64_inverse_kernel_npu",
            ),
            {"T", "NT_OFFSET", "BH_OFFSET"},
            {"BT"},
        ),
    )
    for kernel in kernels
) + (
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_fwd_dense_kernel",
        runtime=frozenset({"T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BT", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_dpre_dense_kernel",
        runtime=frozenset({"T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BT", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_dx_dense_kernel",
        runtime=frozenset({"T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BT", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_dh0_dense_kernel",
        runtime=frozenset({"T", "D", "B_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_fwd_kernel",
        runtime=frozenset({"T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BT", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="_silu_kernel",
        runtime=frozenset({"ELEMENT_OFFSET", "ELEMENT_END"}),
        constexpr=frozenset({"BLOCK"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="_add_kernel",
        runtime=frozenset({"ELEMENT_OFFSET", "ELEMENT_END"}),
        constexpr=frozenset({"BLOCK"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="_silu_bwd_kernel",
        runtime=frozenset({"T", "D", "ELEMENT_OFFSET", "ELEMENT_END"}),
        constexpr=frozenset({"BLOCK"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_bwd_seq_kernel",
        runtime=frozenset({"B", "TC", "D"}),
        constexpr=frozenset({"W", "BLOCK"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_bwd_kernel",
        runtime=frozenset({"T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET", "NT_TOTAL"}),
        constexpr=frozenset({"W", "BT", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_dw_reduce_kernel",
        runtime=frozenset({"N", "T", "D", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BW", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="compute_dh0_kernel",
        runtime=frozenset({"T", "D", "N_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_states_fwd_kernel",
        runtime=frozenset({"T", "D", "W", "N_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"BW", "BD"}),
    ),
    SpecializationContract(
        path=_CONV,
        kernel="causal_conv1d_update_kernel",
        runtime=frozenset({"D", "N_OFFSET", "D_BLOCK_OFFSET"}),
        constexpr=frozenset({"W", "BD"}),
    ),
)


def _jit_decorator(node: ast.FunctionDef) -> ast.Call:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "triton.jit":
            return decorator
    raise AssertionError(f"{node.name} is not decorated with triton.jit")


def _string_set(node: ast.AST) -> set[str]:
    assert isinstance(node, (ast.List, ast.Tuple))
    return {item.value for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)}


@pytest.mark.parametrize("contract", SPECIALIZATION_CONTRACTS, ids=lambda contract: contract.kernel)
def test_specialization_manifest(contract: SpecializationContract):
    tree = ast.parse((_ROOT / contract.path).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert contract.kernel in functions
    kernel = functions[contract.kernel]
    decorator = _jit_decorator(kernel)
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    actual_runtime = _string_set(keywords["do_not_specialize"])
    annotations = {
        argument.arg: ast.unparse(argument.annotation) if argument.annotation is not None else None
        for argument in kernel.args.args
    }

    assert contract.runtime <= actual_runtime
    assert contract.runtime.isdisjoint(contract.constexpr)
    for name in contract.runtime - {"scale"}:
        expected_type = "tl.int32" if name in _BLOCK_POINTER_I32.get(contract.path, ()) else "tl.int64"
        assert annotations[name] == expected_type
    for name in contract.constexpr:
        assert annotations[name] == "tl.constexpr"


def test_triton_implicit_integer_specialization_classes():
    from triton.backends.compiler import AttrsDescriptor

    assert AttrsDescriptor.get_property_key(1, align=True) == "1"
    assert AttrsDescriptor.get_property_key(16, align=True) == "D"
    assert AttrsDescriptor.get_property_key(17, align=True) == "N"
    assert AttrsDescriptor.get_property_key(16, align=False) == "N"
    assert AttrsDescriptor.get_property_key(1, align=False) == "1"


@triton.jit(do_not_specialize=["VALUE"])
def _write_runtime_i64_kernel(output, VALUE: tl.int64):
    if tl.program_id(0) == 0:
        tl.store(output, VALUE == VALUE)


@pytest.mark.skipif(not IS_NPU, reason="Triton-Ascend compile-cache test")
@pytest.mark.ascend_npu
def test_do_not_specialize_reuses_i64_kernel_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    _write_runtime_i64_kernel.cache.clear()
    output = torch.empty((), dtype=torch.int32, device=device)
    cache_counts = []
    disk_counts = []

    for value in (1, 16, 17, 2**31 + 17):
        _write_runtime_i64_kernel[(1,)](output, value, num_warps=1)
        device_torch_lib.synchronize()
        assert output.item() == 1
        cache_counts.append(sum(len(entries) for entries in _write_runtime_i64_kernel.cache.values()))
        cache_root = tmp_path / "triton-cache"
        disk_counts.append(sum(path.is_file() for path in cache_root.rglob("*")) if cache_root.exists() else 0)

    assert cache_counts == [1, 1, 1, 1]
    assert disk_counts[0] > 0
    assert disk_counts == [disk_counts[0]] * len(disk_counts)
