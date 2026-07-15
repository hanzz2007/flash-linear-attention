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

# Start with the GDN kernels that already satisfy the runtime-value contract.
# Later optimization stages extend this manifest before changing each kernel family.
SPECIALIZATION_CONTRACTS = (
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_fwd_h_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT"}),
        constexpr=frozenset({"K", "V", "BT", "BV"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_fwd_m_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT"}),
        constexpr=frozenset({"K", "BT", "BM", "NM"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_bwd_dh_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT", "scale"}),
        constexpr=frozenset({"K", "V", "BT", "BV"}),
    ),
    SpecializationContract(
        path=_GDN_CP,
        kernel="_cp_gdn_bwd_m_kernel",
        runtime=frozenset({"BOS", "SEGMENT_T", "NT"}),
        constexpr=frozenset({"K", "BT", "BM", "NM"}),
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
