# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run paired high-precision CP benchmarks from isolated worktrees."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    script: str
    prefix: str
    nproc: int
    args: tuple[str, ...]
    latency_threshold: float


def _gdn_case(nproc: int) -> BenchmarkCase:
    return BenchmarkCase(
        name=f"gdn_cp{nproc}_fwd_bwd",
        script="benchmarks/cp/benchmark_gdn_cp.py",
        prefix="GDN_CP_BENCH_RESULT=",
        nproc=nproc,
        args=("--mode", "fwd_bwd", "--total-seq-len", "16384", "--q-heads", "8", "--v-heads", "8"),
        latency_threshold=0.12,
    )


def _gdn_preprocess_case(nproc: int, direction: str) -> BenchmarkCase:
    return BenchmarkCase(
        name=f"gdn_preprocess_cp{nproc}_{direction}",
        script="benchmarks/cp/benchmark_gdn_cp_preprocess.py",
        prefix="GDN_CP_PREPROCESS_RESULT=",
        nproc=nproc,
        args=("--direction", direction, "--component", "full", "--total-seq-len", "16384"),
        latency_threshold=0.12,
    )


def _conv_case(nproc: int, dim: int, packed: bool) -> BenchmarkCase:
    args = [
        "--kind",
        "cp",
        "--mode",
        "fwd_bwd",
        "--total-seq-len",
        "16384",
        "--dim",
        str(dim),
        "--precision",
        "high",
    ]
    if packed:
        args.extend(("--lengths", "3000,4000,5000,4384"))
    layout = "packed" if packed else "single"
    return BenchmarkCase(
        name=f"conv_cp{nproc}_d{dim}_{layout}",
        script="benchmarks/cp/benchmark_conv_cp.py",
        prefix="CONV_CP_BENCH_RESULT=",
        nproc=nproc,
        args=tuple(args),
        latency_threshold=0.12,
    )


def _conv_kernel_case(dim: int) -> BenchmarkCase:
    return BenchmarkCase(
        name=f"conv_kernel_d{dim}",
        script="benchmarks/cp/benchmark_conv_cp.py",
        prefix="CONV_CP_BENCH_RESULT=",
        nproc=1,
        args=(
            "--kind",
            "kernel",
            "--mode",
            "fwd_bwd",
            "--total-seq-len",
            "2048",
            "--dim",
            str(dim),
            "--precision",
            "high",
        ),
        latency_threshold=0.10,
    )


def _conv_comm_case(nproc: int) -> BenchmarkCase:
    return BenchmarkCase(
        name=f"conv_comm_cp{nproc}",
        script="benchmarks/cp/benchmark_conv_cp.py",
        prefix="CONV_CP_BENCH_RESULT=",
        nproc=nproc,
        args=("--kind", "comm", "--mode", "fwd_bwd", "--dim", "3072", "--precision", "high"),
        latency_threshold=0.12,
    )


def _cases(suite: str) -> list[BenchmarkCase]:
    if suite == "pr":
        return [_gdn_case(8), _conv_case(8, 1024, True)]
    cases: list[BenchmarkCase] = []
    for nproc in (2, 4, 8):
        cases.append(_gdn_case(nproc))
        cases.extend(_gdn_preprocess_case(nproc, direction) for direction in ("fwd", "bwd"))
        cases.append(_conv_comm_case(nproc))
        for dim in (1024, 3072):
            cases.extend(_conv_case(nproc, dim, packed) for packed in (False, True))
    cases.extend(_conv_kernel_case(dim) for dim in (1024, 3072))
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-worktree", type=Path, required=True)
    parser.add_argument("--candidate-worktree", type=Path, required=True)
    parser.add_argument("--platform", choices=("npu", "cuda"), required=True)
    parser.add_argument("--suite", choices=("pr", "full"), default="pr")
    parser.add_argument("--profile", choices=("candidate", "final"), default="candidate")
    parser.add_argument("--case", action="append", default=[], help="Run only an exact case name; repeatable")
    parser.add_argument("--output", type=Path, default=Path("ascend_cp_regression.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp/fla-cp-regression"))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _visible_devices(nproc: int) -> str:
    if nproc == 8:
        return ",".join(str(index) for index in range(8))
    return ",".join(str(index) for index in range(2, 2 + nproc))


def _command(case: BenchmarkCase, harness_worktree: Path, warmup: int, samples: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={case.nproc}",
        str(harness_worktree / case.script),
        *case.args,
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--confirm-samples",
        "20",
    ]


def _run_case(
    case: BenchmarkCase,
    *,
    label: str,
    worktree: Path,
    harness_worktree: Path,
    platform: str,
    cache_root: Path,
    warmup: int,
    samples: int,
    timeout: int,
    dry_run: bool,
) -> dict[str, Any]:
    command = _command(case, harness_worktree, warmup, samples)
    if dry_run:
        print("DRY_RUN", json.dumps(command))
        return {"dry_run": True, "command": command}
    if not (harness_worktree / case.script).is_file():
        raise FileNotFoundError(harness_worktree / case.script)

    env = os.environ.copy()
    visible = _visible_devices(case.nproc)
    visible_name = "ASCEND_RT_VISIBLE_DEVICES" if platform == "npu" else "CUDA_VISIBLE_DEVICES"
    hidden_name = "CUDA_VISIBLE_DEVICES" if platform == "npu" else "ASCEND_RT_VISIBLE_DEVICES"
    env[visible_name] = visible
    env.pop(hidden_name, None)
    env["FLA_ASCEND_CP_GDN_PRECISION"] = "high"
    env["FLA_ASCEND_CONV_PRECISION"] = "high"
    env["FLA_BENCH_CACHE_ROOT"] = str(cache_root / label)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(worktree), env.get("PYTHONPATH"))))
    if platform == "npu":
        env.setdefault("ASCEND_UB_CAPACITY_BITS", "1572864")
        env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=worktree,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(f"{case.name} ({label}) exited with {completed.returncode}")
    payloads = [line.removeprefix(case.prefix) for line in completed.stdout.splitlines() if line.startswith(case.prefix)]
    if len(payloads) != 1:
        raise RuntimeError(f"Expected one {case.prefix} line for {case.name}, found {len(payloads)}")
    result = json.loads(payloads[0])
    result["runner_wall_seconds"] = time.perf_counter() - started
    return result


def _validate_metadata(baseline: dict[str, Any], candidate: dict[str, Any], case: BenchmarkCase) -> None:
    keys = ("backend", "device", "torch", "triton", "triton_ascend", "world_size", "dtype", "precision")
    mismatches = {key: (baseline.get(key), candidate.get(key)) for key in keys if baseline.get(key) != candidate.get(key)}
    if mismatches:
        raise ValueError(f"Refusing to compare {case.name} with mismatched metadata: {mismatches}")


def _compare_pair(case: BenchmarkCase, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    _validate_metadata(baseline, candidate, case)
    baseline_ms = baseline["latency_ms"]["median"]
    candidate_ms = candidate["latency_ms"]["median"]
    baseline_memory = baseline["peak_memory_bytes"]
    candidate_memory = candidate["peak_memory_bytes"]
    backend = baseline["backend"]
    latency_threshold = 0.05 if backend == "nccl" else case.latency_threshold
    return {
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "latency_ratio": candidate_ms / baseline_ms,
        "latency_threshold": latency_threshold,
        "latency_regressed": candidate_ms > baseline_ms * (1 + latency_threshold),
        "baseline_peak_memory_bytes": baseline_memory,
        "candidate_peak_memory_bytes": candidate_memory,
        "memory_ratio": candidate_memory / baseline_memory if baseline_memory else 1.0,
        "memory_regressed": bool(baseline_memory and candidate_memory > baseline_memory * 1.05),
    }


def main() -> None:
    args = _parse_args()
    warmup, samples = (2, 5) if args.profile == "candidate" else (3, 10)
    selected = _cases(args.suite)
    if args.case:
        requested = set(args.case)
        selected = [case for case in selected if case.name in requested]
        missing = requested - {case.name for case in selected}
        if missing:
            raise ValueError(f"Unknown or unavailable cases: {sorted(missing)}")

    worktrees = {"baseline": args.baseline_worktree.resolve(), "candidate": args.candidate_worktree.resolve()}
    harness_worktree = worktrees["candidate"]
    orders = (("baseline", "candidate"), ("candidate", "baseline"))
    report: dict[str, Any] = {
        "platform": args.platform,
        "suite": args.suite,
        "profile": args.profile,
        "warmup": warmup,
        "requested_samples": samples,
        "cases": {},
    }
    failed: list[str] = []
    for case in selected:
        pair_reports = []
        raw_runs = []
        for order_index, order in enumerate(orders):
            results = {}
            for label in order:
                result = _run_case(
                    case,
                    label=label,
                    worktree=worktrees[label],
                    harness_worktree=harness_worktree,
                    platform=args.platform,
                    cache_root=args.cache_root / f"order-{order_index}" / case.name,
                    warmup=warmup,
                    samples=samples,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                )
                results[label] = result
                raw_runs.append({"order": order_index, "label": label, "result": result})
            if not args.dry_run:
                pair_reports.append(_compare_pair(case, results["baseline"], results["candidate"]))
        latency_regressed = bool(pair_reports) and all(pair["latency_regressed"] for pair in pair_reports)
        memory_regressed = bool(pair_reports) and all(pair["memory_regressed"] for pair in pair_reports)
        regressed = latency_regressed or memory_regressed
        if regressed:
            failed.append(case.name)
        report["cases"][case.name] = {
            "definition": asdict(case),
            "pairs": pair_reports,
            "runs": raw_runs,
            "latency_regressed": latency_regressed,
            "memory_regressed": memory_regressed,
            "regressed": regressed,
        }

    report["failed_cases"] = failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    if failed:
        raise SystemExit(f"Paired regression in both orders: {', '.join(failed)}")


if __name__ == "__main__":
    main()
