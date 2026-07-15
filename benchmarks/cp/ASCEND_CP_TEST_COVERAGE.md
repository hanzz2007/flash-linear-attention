# Ascend CP industrial test coverage

This record tracks the correctness, distributed, stability, and performance gates for the Triton-Ascend FLA CP and causal Conv1d CP paths. Public signatures and numerical tolerances remain frozen while coverage is expanded.

## Frozen numerical gates

| Path | Tensor | RMS-ratio limit |
| --- | --- | ---: |
| GDN high primitive | H, M, dH, dM, merge | `1e-4` |
| GDN public | output and every gradient | `3e-3` |
| KDA/DPLR/RWKV7 smoke | output and gradients | `8e-3` |
| Conv public | output, dx, dw, db | `1e-3` |
| Conv BF16 | dh0 and multi-hop dx correction | `3.1e-3` |
| Conv CP2/4 BF16 | dw, db | `3.3e-3` |
| Conv CP8 BF16 | dw, db | `4.8e-3` |

Every numerical check also requires finite reference and actual tensors. Tests use an absolute-error escape only for near-zero references; `FLA_CI_ENV` never downgrades these gates to warnings.

## Phase results

| Phase | Scope | Environment | Result | Commit |
| --- | --- | --- | --- | --- |
| 1 | CP context validation, exact rank metadata, int32/int64 global inputs, copy semantics, coverage tags | 910B physical device 2; CP2 on devices 2–3; CANN 9.0.0, PyTorch 2.7.1, torch-npu 2.7.1.post6, Triton-Ascend 3.2.0 | 29 protocol tests and the existing CP2 Conv sequence-cut public test passed | `2f5d65c6` |
| 2 | GDN primitive dimensions, GVA, non-power-of-two/tail tiles, K=1/193/256, precomputed gates, merge ordering, canaries, forced grid splitting | 910B physical device 2; same stack; isolated Triton caches | 8 PR primitives, 5 merge/grid cases, and 9 host gates passed. Cold-cache PR primitives took 295.53 s. | `41e3ae4e` |
| 3 | Conv dense, packed varlen, FP32/layout fallback, structured impulses, final state, incremental cache, NaN poisoning, canaries, selector and pairwise coverage catalog | 910B physical device 2; same stack; isolated Triton caches | 32 PR gates passed. This includes `T=2048,D=3072,W=4` high forward/backward. The tests found and fixed repeated BF16 postprocessing casts, BF16 `dpre`, bias partial-workspace reduction, and a per-sequence reference-cast bug. | `c697d967` |
| 4 | Multi-rank short Conv halo (`Tlocal < W-1`), forward history assembly, backward owner mapping, direct FP32 `dw` reduction | 910B physical devices 2–5; same stack; isolated single-device and CP4 Triton caches | 38 host/protocol gates, 3 packed-varlen forward/backward cases, and CP4 all-gather/P2P-requested BF16 multi-hop cases passed. The tests exposed and fixed missing initial-state contributions in general `dw`; the direct reducer also removes the `O(NT*D*W)` FP32 partial workspace. Multi-hop corrections are accumulated in FP32 before one BF16 cast (`dx` RMS ratio improved from `2.86e-3` to `2.22e-3`). | `fdb0d912` |
| 5 | High-only distributed public gates, uniform markers, rank-consistent failure status, CP subgroup routing, packed CP8, non-GDN shared primitives | 910B physical devices 0–7 and A800 physical devices 2–3; designated production stacks | GDN CP2 FP16/GVA and fused-input cases, CP4 packed-varlen, and CP8 `T=16384,H=8,K=V=128` passed. Conv CP2 all-gather/P2P, CP4 multi-hop, non-contiguous subgroup, and packed CP8 `T=16384,D=1024,W=4` passed. KDA/DPLR CP4 primitive forward/backward and two A800 CUDA regression cases passed. Conv CP8 FP32 validation reduction measured `dw=3.70e-3`, `db=3.58e-3`; BF16 reduction alone measured `dw=5.21e-3` and was rejected instead of weakening the `4.8e-3` gate. | `58e0751c` |
| 6 | Path-driven PR CI, exclusive CP2/4/8 gates, scheduled nightly matrix, paired worktree performance runner, compile/latency/HBM evidence | 910B physical devices 0–7; same production stack; high precision only | Workflow YAML, benchmark imports, dry-run command matrix, and 3 runner contract tests passed. GDN, Conv, and preprocessing benchmark entry points passed real HCCL smoke; KDA/DPLR CP8 primitive forward/backward passed 2/2 in 61.55 s. | This phase commit |

Compatibility precision selectors remain available, but A800-mode numerical and performance gates are excluded from the active test matrix. High precision is the only acceptance mode.

Cold compilation is recorded separately from numerical execution. The two new GDN CP2 specializations took 545.92 s on a cold 910B cache; cached CP4 and CP8 gates took 46.64 s and 49.95 s. The packed Conv CP8 gate took 30.40 s after compilation. These wall times are test-process totals and are not kernel performance measurements.

Performance sampling is deliberately subordinate to correctness. Candidate runs use two warmups and five samples; final manual runs use three warmups and ten samples. A case expands to 20 samples only when its observed CV exceeds 10%. The paired runner executes baseline/candidate in both orders and fails only when both comparisons exceed the frozen envelope: 10% for kernel-only NPU, 12% for HCCL end-to-end, 5% for CUDA routing, and 5% for peak HBM. Nightly correctness retains the complete CP2/4/8 matrix while stochastic coverage uses eight fixed seeds and 50 continuous high-precision iterations per CP8 target.

The stage-6 one-sample smoke values (`83.98 ms` GDN CP8 and `39.92 ms` packed Conv CP8) validate the measurement path only and are not accepted as performance conclusions. Cold/load time was reported separately (`3.46 s` and `3.40 s` respectively); both outputs and all checked gradients remained finite.

For near-zero BF16 Conv parameter gradients, the single-device tests additionally freeze `max_abs <= 2.5e-4`; non-near-zero tensors must satisfy the RMS-ratio limit.

## Reproduction

```bash
source /data/Ascend/9.0.0/cann-9.0.0/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 \
ASCEND_UB_CAPACITY_BITS=1572864 \
PYTHONPATH=. \
/data/qiuhan/3.conda/swift-mmu-beta-v0.2-cann900_triton321/bin/python \
  -m pytest tests/context_parallel/test_cp_context.py -v
```
