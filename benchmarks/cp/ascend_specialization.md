# Ascend Triton compile-specialization optimization report

## Executive summary

This change reduces redundant Triton-Ascend compilation for GDN context-parallel and causal Conv1d kernels without changing their mathematics, public APIs, precision policy, communication protocol, or CUDA implementation.

The accepted implementation establishes an explicit specialization contract for 55 kernels. Logical shapes, sequence positions, ranks, task counts, and grid-split offsets are typed runtime scalars covered by `do_not_specialize`. Tile sizes, convolution width, mode/layout flags, dtypes, pointer presence, and other code-generation features remain static. Pointer and stride specialization is intentionally retained because alignment and unit-stride information can affect vectorization.

On Ascend 910B, a capability kernel confirmed that runtime values `1`, `16`, `17`, and `2**31 + 17` reuse one in-memory cache entry and do not create new disk artifacts after the first compile. All final correctness gates passed, including CP2/4/8.

A clean paired run was subsequently completed on otherwise idle 8-card 910B and A800 hosts. For the CP8 packed-varlen GDN target, cold start decreased from 194.20 s to 185.39 s (-4.5%), while paired steady-state medians changed by only +0.2% on average. Conv cold start was effectively unchanged (+0.8%), and paired steady-state medians improved by 1.3% on average. These measurements confirm that despecialization preserves steady-state performance; its primary benefit remains cross-shape cache reuse rather than faster arithmetic.

## Environment and scope

| Item | Value |
| --- | --- |
| Device | Ascend 910B, single-device gates on physical device 2; distributed gates on devices 0–7 |
| Software | CANN 9.0.0, PyTorch 2.7.1, torch_npu 2.7.1.post6, Triton 3.2.0, triton-ascend package 3.2.1 |
| Precision tested | `high` only; the `high/a800` selector remains compatible |
| Changed paths | Ascend GDN CP/shared kernels, Ascend Conv1d backend, specialization tests, this report |
| Unchanged | Public APIs, CP wire shapes, HCCL ordering, mathematical recurrence, CUDA/A800 kernels |

## Specialization contract

Triton-Ascend 3.2.0 implicitly classifies ordinary scalar values in its cache key:

| Runtime value class | Default key | Risk |
| --- | --- | --- |
| Value equals one | `1` | May be promoted to a compile-time constant |
| Divisible by 16 | `D` | Creates an alignment-specialized variant |
| Other value | `N` | Creates a second alignment class |
| Different inferred integer type | `i32` / `i64` | Creates another signature |

Removing `tl.constexpr` alone is therefore insufficient. High-variation logical scalars use both an explicit type and `do_not_specialize`:

| Parameter class | Policy |
| --- | --- |
| T, D, B/N/NT, BOS, segment length, rank, source-rank step, task count | `tl.int64` plus `do_not_specialize` |
| Grid-split and element offsets | `tl.int64` plus `do_not_specialize` |
| Block-pointer offsets | `tl.int32` plus `do_not_specialize` |
| W, BT, BD, BW, block/tile shape, direction, layout, feature flags | `tl.constexpr` |
| Pointer dtype/presence and tensor strides | Intentional default specialization |

### Block-pointer ABI exception

Triton-Ascend 3.2.0 allows int64 block-pointer shapes and strides but requires int32 offsets. The first implementation attempt used int64 for every offset and was rejected after representative GDN kernels failed compilation. The accepted path narrows only the value entering `tl.make_block_ptr` to int32; all base-pointer arithmetic remains int64. The int32 value is still covered by `do_not_specialize`, so the `1/D/N` cache split remains disabled.

## Implemented changes

### GDN CP and shared GDN kernels

- Despecialized gate-factor `NT`, BOS, segment lengths, transition/merge ranks, and all host grid-split offsets.
- Applied the same contract to shared gate, WY, KKT, output, cumsum, triangular-solve, H, and dH kernels.
- Removed unused cumsum batch parameters instead of replacing them with dynamic arguments.
- Kept K/V/tile shapes, gate mode, direction, layout, and the `high/a800` choice static.
- Preserved int64 explicit pointer arithmetic and the required int32 block-pointer offsets.

### Packed-varlen Conv1d

- Despecialized T, D, B/NT counts, total chunk count, and B/NT/D split offsets in forward and backward.
- Replaced elementwise `constexpr` offsets with runtime `ELEMENT_OFFSET` and `ELEMENT_END` in SiLU, add, and SiLU backward kernels.
- Despecialized short-sequence backward dimensions, direct dW reduction dimensions, dh0, final-state, and incremental-update offsets.
- Removed unused general forward/backward batch arguments.
- Preserved W/tile/feature flags, stride specialization, fused activation/residual behavior, direct FP32 dW reduction, and the existing low-workspace algorithm.

### Dense Conv1d

The dense forward, dpre, and dx kernels now use a grid-safe 3D mapping:

```text
program_id(0) -> channel tile + D_BLOCK_OFFSET
program_id(1) -> time tile    + NT_OFFSET
program_id(2) -> batch        + B_OFFSET
```

This removes flat-task division and the static `NT`, `DB`, and `TASK_OFFSET` dimensions. Host-side slicing guarantees each grid product is at most 65535. Dense dh0 uses a smaller channel/batch grid. A forced `max_grid=2` test verifies split and unsplit forward/backward results are identical.

## Results

### Compile reuse

| Check | Result |
| --- | --- |
| Manifested kernels | 55 |
| Capability values | `1`, `16`, `17`, `2**31 + 17` |
| In-memory cache entries after each value | `1, 1, 1, 1` |
| Disk artifact count | Unchanged after the first compile |
| Static dimensions retained | dtype, pointer/None, strides, tiles, W, mode/layout/feature flags |

The capability test is the direct evidence for cross-shape reuse. A single target shape cannot expose all avoided dynamic-shape variants, but its isolated disk caches provide a useful end-to-end check:

| CP8 workload | Baseline files | Candidate files | Change | Baseline bytes | Candidate bytes | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN packed-varlen | 1,394 | 1,314 | -5.7% | 3,760,076,104 | 3,758,016,023 | -0.05% |
| Conv packed-varlen | 275 | 278 | +1.1% | 3,657,761,428 | 3,659,104,616 | +0.04% |

The byte totals are dominated by large shared compiler artifacts, so file count and the per-kernel cache-entry test are more informative than aggregate bytes. The Conv single-shape result is neutral, as expected; its benefit appears when T/D/count/offset values vary inside the same static kernel family.

### Paired 910B performance

Baseline is `d90b2cc9`, before the production-kernel despecialization. Candidate is `7b1b5d18`. Both were synchronized into separate directories, used isolated Triton caches, and were run in baseline→candidate then candidate→baseline order. Each steady-state run used 3 warmups and 10 samples; no 910B case exceeded the 10% CV expansion threshold.

| CP8 packed-varlen BF16 fwd+bwd | Baseline medians (ms) | Candidate medians (ms) | Average change | Cold start baseline/candidate | Peak memory |
| --- | --- | --- | ---: | ---: | ---: |
| GDN, T=16384, H=HV=8, K=V=128, BT=64 | 86.214 / 86.507 | 86.456 / 86.635 | +0.2% | 194.20 / 185.39 s (-4.5%) | 126,902,272 B, unchanged |
| Conv, T=16384, D=1024, W=4, SiLU | 40.316 / 41.288 | 40.533 / 40.041 | -1.3% | 54.57 / 55.02 s (+0.8%) | 56,663,552 B, unchanged |

The two run orders do not show a confirmation-level regression. GDN remains statistically equivalent; Conv order-to-order drift is larger than the candidate effect, but its paired average is slightly faster.

### Correctness evidence

| Gate | Result |
| --- | --- |
| Final specialization, coverage-catalog, and halo protocol suite | 65 passed |
| Shared GDN kernel matrix | 41 passed |
| GDN CP preprocessing matrix | 21 passed, 2 distributed cases skipped in the single-device run |
| Conv non-nightly single-NPU matrix | 30 passed |
| Distributed representatives | 6 passed: CP2 x2, CP4 x2, CP8 x2 |

Distributed high-precision error ratios:

| Case | Output | dx/dq | dk | dv | gate/bias or dW/dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN CP8, T=16384, H=8, K=V=128 | 0 | dq 0 | 5e-6 | 3e-6 | dg 1e-5, db 4e-6 |
| GDN CP4 complex varlen | 6.16e-4 | dq 7.94e-4 | 9.78e-4 | 8.44e-4 | dg 9.14e-4, db 7.06e-4 |
| GDN CP2 FP16, GVA, K=96, V=80, BT=16 | 0 | dq 0 | 3e-6 | 6e-6 | dg 0, db 5e-6 |
| Conv CP8 packed, T=16384, D=1024, W=4 | 2.29e-6 | dx 8.76e-5 | — | — | dW 3.70e-3, dB 3.58e-3 |
| Conv CP2 BF16, D=1024 | 0 | dx 1.81e-4 | — | — | dW 2.37e-3, dB 2.28e-3 |
| Conv CP4 multi-hop, Tlocal=1, D=17 | 0 | dx 2.22e-3 | — | — | dW 1.72e-3, dB 1.43e-3 |

All results were finite and within their frozen thresholds. The CP4 Conv case proves multi-rank halo propagation when `Tlocal < W - 1`.

## Efficiency and A800 gap

- Compile efficiency improved structurally: high-variation logical scalars no longer generate separate `1`, `D`, `N`, or inferred i32/i64 variants inside a fixed static kernel family.
- Steady-state kernel work, communication payload, numerical operations, and precision are unchanged by design. A large steady-state speedup is neither expected nor claimed.
- CUDA/A800 code was not modified. Alternating baseline/candidate runs reversed direction on the second order, so there is no confirmation-level CUDA regression.
- Earlier timing data remains excluded. Only the clean paired measurements below are used.

| Candidate CP8 packed-varlen BF16 fwd+bwd | 910B paired median average | A800 paired median average | 910B/A800 | Peak-memory ratio |
| --- | ---: | ---: | ---: | ---: |
| GDN target | 86.545 ms | 4.382 ms | 19.8x | 1.25x |
| Conv D=1024 | 40.287 ms | 2.149 ms | about 18.8x | 1.65x |

The GDN A800 per-run CV was 6.7%–9.2%, so its 19.8x ratio is suitable as an end-to-end production-stack gap. The much shorter Conv A800 workload was noisier (8.0%–13.5% CV); its candidate medians imply a 17.6x–20.1x range, and 18.8x is only the paired-average summary.

This is not a pure hardware-efficiency ratio: the hosts intentionally use different production stacks (910B: PyTorch 2.7.1/CANN 9.0.0/Triton 3.2.0; A800: PyTorch 2.11.0+cu129/Triton 3.6.0), different collective backends, and different kernel implementations. The remaining approximately 20x GDN gap is therefore outside the scope of compile despecialization and requires execution-path profiling and kernel algorithm/scheduling work.

## Reproduction

Single-device gates use physical device 2:

```bash
source /data/Ascend/9.0.0/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_UB_CAPACITY_BITS=1572864
python -m pytest -q tests/modules/test_ascend_specialization.py
python -m pytest -q tests/ops/test_gdn_kernels.py
python -m pytest -q tests/context_parallel/test_cp_gdn_preprocess.py -m "not nightly"
python -m pytest -q tests/modules/test_conv_triton_ascend.py -m "not nightly"
```

Representative distributed gates use all eight devices and no xdist:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_NPU_SOCKET_PORT_RANGE=auto
python -m pytest -s -q \
  tests/context_parallel/test_cp_gdn.py::test_cp8_single_sequence \
  'tests/context_parallel/test_cp_conv.py::test_cp8_target[1024]'
```

## Commits

| Phase | Commit |
| --- | --- |
| Specialization reuse gates | `d90b2cc9` |
| GDN CP/shared GDN despecialization | `f3ae596e` |
| Packed-varlen Conv1d despecialization | `3ef2354c` |
| Dense Conv1d 3D grid | `1aa26729` |
| Initial report | `7b1b5d18` |
| Clean paired performance evidence | This documentation update; exact SHA is recorded in the delivery summary |

No pull request was created or updated against the upstream repository.
