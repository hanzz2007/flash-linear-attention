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
| 6 | Path-driven PR CI, exclusive CP2/4/8 gates, scheduled nightly matrix, paired worktree performance runner, compile/latency/HBM evidence | 910B physical devices 0–7; same production stack; high precision only | Workflow YAML, benchmark imports, dry-run command matrix, and 3 runner contract tests passed. GDN, Conv, and preprocessing benchmark entry points passed real HCCL smoke; KDA/DPLR CP8 primitive forward/backward passed 2/2 in 61.55 s. | `7853a55a` |
| 7 | Final high-only nightly execution, result consolidation, Markdown and standalone wide-screen HTML report | 910B physical devices 0–7; same production stack | Single-NPU nightly 34/34 and distributed CP2/4/8 39/39 passed. Fifty-iteration GDN and Conv CP8 stability runs remained finite with zero sustained allocation growth. | Report commit |

Compatibility precision selectors remain available, but A800-mode numerical and performance gates are excluded from the active test matrix. High precision is the only acceptance mode.

Cold compilation is recorded separately from numerical execution. The two new GDN CP2 specializations took 545.92 s on a cold 910B cache; cached CP4 and CP8 gates took 46.64 s and 49.95 s. The packed Conv CP8 gate took 30.40 s after compilation. These wall times are test-process totals and are not kernel performance measurements.

Performance sampling is deliberately subordinate to correctness. Candidate runs use two warmups and five samples; final manual runs use three warmups and ten samples. A case expands to 20 samples only when its observed CV exceeds 10%. The paired runner executes baseline/candidate in both orders and fails only when both comparisons exceed the frozen envelope: 10% for kernel-only NPU, 12% for HCCL end-to-end, 5% for CUDA routing, and 5% for peak HBM. Nightly correctness retains the complete CP2/4/8 matrix while stochastic coverage uses eight fixed seeds and 50 continuous high-precision iterations per CP8 target.

The stage-6 one-sample smoke values (`83.98 ms` GDN CP8 and `39.92 ms` packed Conv CP8) validate the measurement path only and are not accepted as performance conclusions. Cold/load time was reported separately (`3.46 s` and `3.40 s` respectively); both outputs and all checked gradients remained finite.

## Final high-only acceptance

| Gate | Scope | Result | Wall time |
| --- | --- | ---: | ---: |
| Single NPU nightly | GDN K=127/129/192/255, long/range scans; Conv D=8191/8192/8193, dense and packed varlen forward/backward | 34 passed | 242.60 s |
| Distributed nightly | GDN public CP2/4/8, state layouts, K256; KDA/DPLR primitives CP4/8; Conv CP2/4/8, short tails, subgroup, all-gather/P2P, multi-hop halo, D1024/3072 targets | 39 passed, 36 deselected | 1241.66 s |
| Runner contracts | High-only matrix, device mapping, NPU/CUDA envelopes, stack mismatch rejection | 3 passed | 0.04 s |
| GDN CP8 stability | `Tglobal=16384,H=HV=8,K=V=128,BT=64,BF16`, forward + backward, 50 iterations | finite, allocation growth 0 | measured separately |
| Conv CP8 stability | packed `[3000,4000,5000,4384]`, `D=1024,W=4,BF16`, forward + backward, 50 iterations | finite, allocation growth 0 | measured separately |

The stability measurements are runner validation and regression baselines, not a claim that the original A800 latency target has been reached:

| Workload | Compile/load | p20 | p50 | p80 | CV | Tokens/s | Peak HBM | Growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN CP8 high | 3493.92 ms | 83.497 ms | 83.745 ms | 83.903 ms | 0.408% | 195,641 | 125,327,872 B | 0 B |
| Conv CP8 high packed | 3281.27 ms | 39.021 ms | 39.129 ms | 39.378 ms | 0.858% | 418,715 | 56,663,552 B | 0 B |

## Coverage architecture

| Layer | What it catches | Representative frozen edges |
| --- | --- | --- |
| Protocol/host | Invalid metadata before collectives, rank mapping, selector and coverage drift | world size 1/2/4/8, int32/int64 global metadata, malformed cumulative lengths, subgroup ranks |
| Primitive | Addressing, tail masks, scan order, cast points, uninitialized output | K=1..256, V non-power-of-two, nonzero BOS, forced grid 4/7, NaN poisoning, canaries |
| Public distributed | Integration across autograd and HCCL | CP2/4/8, GVA, packed varlen, state layout, all-gather/P2P, complete gradients |
| Nightly/stability | Rare specializations and state pollution | D=8193, long scan, shape alternation, fixed seeds, repeated high-precision execution |
| Performance | Trend regression without weakening correctness | isolated caches/worktrees, cold compile excluded, slowest-rank wall clock, paired order |

The coverage catalog is represented by tagged dataclasses and pure-Python meta-tests. Removing a minimum, maximum, target, dtype, layout, tail, or fallback bucket therefore fails collection-level coverage even before an NPU is allocated.

## Defects exposed and retained fixes

- CP metadata validation now rejects malformed global lengths, zero-token partitions and inconsistent CPU/device metadata before HCCL.
- Conv testing exposed repeated BF16 postprocessing casts, BF16 `dpre`, partial-workspace bias reduction, and a per-sequence reference cast error.
- General Conv `dw` omitted initial-state contributions. The direct FP32 reducer fixed correctness and removed the `O(NT*D*W)` partial workspace.
- The neighbor-only halo assumption failed when `Tlocal < W-1`. Forward now assembles right-aligned history across ranks; backward maps every halo gradient to its global owner.
- CP8 Conv validation originally reduced parameter gradients in BF16 (`dw` ratio `5.21e-3`). The validation reduction was corrected to FP32, producing `dw=3.70e-3`, `db=3.58e-3` without relaxing the `4.8e-3` gate.
- Distributed workers now publish rank-consistent status before assertion, reducing false hangs caused by one rank leaving a collective early.

No correctness failure was converted to `xfail`, a warning, or a wider tolerance. Compatibility `a800` selectors remain in the implementation but are intentionally absent from device tests and benchmarks.

## CI topology

- `ascend-a2-ci.yml`: path-driven host contracts, single-NPU PR matrix, exclusive CP2/4, and representative CP8 high gates.
- `ascend-cp-nightly.yml`: complete correctness matrix, eight fixed seed smoke runs, and 50-iteration high stability.
- `ascend-cp-performance.yml`: isolated baseline/candidate worktrees and caches, both execution orders, metadata matching, and high-only paired regression.

Distributed pytest always uses `-n 0`. Single-device files may use `-n 8 --dist loadfile`. Cold compilation is executed and recorded before warmup; only synchronized slowest-rank wall-clock samples enter latency statistics.

## Commit ledger

| Commit | Purpose |
| --- | --- |
| `2f5d65c6` | Deterministic CP protocol gates |
| `41e3ae4e` | Ascend GDN primitive and merge coverage |
| `51dca9b5` | High-only acceptance matrix; retain selector compatibility |
| `c697d967` | Ascend causal Conv dense/varlen/fallback coverage |
| `fdb0d912` | Multi-rank short halo and direct gradient reducer fix |
| `58e0751c` | CP2/4/8 distributed correctness and CUDA routing regression |
| `7853a55a` | Fast/nightly CI split and paired performance runner |

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

```bash
# Full high-only single-NPU nightly matrix (8 parallel workers)
FLA_NPU_XDIST=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m pytest -n 8 --dist loadfile \
  -m "ascend_npu and nightly and not cp_distributed" \
  tests/context_parallel/test_cp_gdn_preprocess.py \
  tests/modules/test_conv_triton_ascend.py

# Exclusive distributed CP2/4/8 matrix
FLA_NPU_XDIST=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m pytest -n 0 -m cp_distributed \
  tests/context_parallel/test_cp_gdn.py \
  tests/context_parallel/test_cp_gdn_preprocess.py \
  tests/context_parallel/test_cp_conv.py

# Lightweight paired performance gate; expands only if CV > 10%
python benchmarks/cp/run_ascend_cp_regression.py \
  --baseline-worktree /path/to/baseline \
  --candidate-worktree /path/to/candidate \
  --platform npu --suite pr --profile candidate
```
