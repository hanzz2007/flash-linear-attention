# Ascend 910B CP Causal Conv1d Optimization Record

## Contract

The target is BF16 causal depthwise Conv1d with `W=4`, `Tglobal=16384`, CP2/4/8, and `D=1024/3072`. Public APIs, return shapes and dtypes, existing CUDA kernels, and strict correctness tolerances remain unchanged. The default precision path remains high precision; an A800-relative path may be promoted only within the measured A800 error envelope.

## Environment

| Platform | Hardware | Software |
| -------- | -------- | -------- |
| A800 | 8x NVIDIA A800-SXM4-80GB | PyTorch 2.11.0+cu129, Triton 3.6.0 |
| 910B | 8x Ascend 910B | CANN 9.0.0, PyTorch 2.7.1, torch_npu 2.7.1.post6, Triton 3.2.0 |

The comparison is therefore the efficiency of each device together with its pinned production stack, not a pure silicon peak comparison.

## Stage 1 — frozen baseline and gates

- The existing CP wrapper already reaches the module-level Triton-Ascend backend.
- A finite single-card initial-state and SiLU forward/backward smoke passes on physical NPU 2.
- Exploratory `Tlocal=2048,D=3072,W=4,BF16` latency is `20.028/90.653 ms` on 910B and `0.111/0.879 ms` on A800.
- The initial 910B/A800 gaps are approximately `180.6x` forward and `103.1x` backward.
- The BF16 CP2 target smoke measures `dw` RMS ratios of `2.978e-3` on A800 and `2.937e-3` on 910B, with `db` at `2.904e-3/2.728e-3`. Because each rank's parameter gradients are rounded to BF16 before the test reduction, `dw/db` use the frozen A800 `1.10x` envelope (`3.3e-3`); output and `dx` retain the strict `1e-3` gate.
- The existing Ascend forward uses `BD=8,BT=32`; backward uses `BD=8,BT=8` at the target shape and materializes FP32 partial `dw/db` workspaces before `torch.sum`.
- `tests/context_parallel/test_cp_conv.py` is platform neutral for NCCL/HCCL, uses strict finite RMS assertions, dynamic rendezvous ports, and contains BF16 target-shape gates.
- `benchmarks/cp/benchmark_conv_cp.py` separates compilation from timing and reports max-rank latency, percentiles, CV, throughput, and peak memory.

The frozen single-rank `fwd_bwd` benchmark at `Tlocal=2048,D=3072,W=4,BF16,SiLU` produced:

| Platform | Samples | Compile | p20 | p50 | p80 | CV | Peak HBM |
| -------- | ------: | ------: | --: | --: | --: | -: | -------: |
| A800 | 50 | 0.500 s (warm cache) | 1.151 ms | 1.165 ms | 1.189 ms | 6.42% | 86.0 MiB |
| 910B | 10 | 18.934 s (cold cache) | 110.825 ms | 111.294 ms | 111.698 ms | 0.61% | 111.2 MiB |

The aggregate median gap is `95.6x`. The A800 10-sample run exceeded the 5% CV gate, so the prescribed single 50-sample confirmation was run; its remaining 6.42% variation is reported rather than hidden by further repetitions. The earlier synchronized component measurements (`0.111/0.879 ms` versus `20.028/90.653 ms`) remain the frozen forward/backward decomposition.

Subsequent stages append every kept or rejected candidate, correctness result, latency, memory result, and commit SHA here.

Stage 1 commit: `cae5fbcc60dcfc0a44f990497e11e181bece8456`.

## Stage 2 — dense forward fast path

The retained Ascend-only path applies to contiguous BF16/FP16 tensors, `W=2/3/4`, and one dense local sequence. It uses a flattened 1D launch whose tasks each compute a `BT=64,BD=128` token/channel tile. All pointer arithmetic is int64, four taps are statically expanded, accumulation is FP32, and bias, SiLU, residual, and output cast are fused. Launches pass `multibuffer=False` and are split before the 65535-program limit. Other layouts, FP32, and multi-sequence varlen inputs retain the existing implementation.

The user-provided `casual_conv1d_opt.py` reference was evaluated without committing it. Its three-axis splitting and `BT=64,BD=4` layout improved forward from `20.028 ms` to `17.684 ms`. Migrating its grid-safe task decomposition and increasing the UB-safe tile produced the retained result:

| Implementation | Programs | Launches | p50 forward | CV | Peak HBM |
| -------------- | -------: | -------: | -----------: | -: | -------: |
| Existing Ascend | about 27,648 including SiLU | 2 | 20.028 ms | exploratory | at least 60.1 MiB |
| User reference | about 30,720 including SiLU | 2 | 17.684 ms | 0.40% | 60.1 MiB |
| Retained fused `64x128` | 1,536 | 1 | 2.667 ms | 0.61% | 48.1 MiB |
| A800 production stack | n/a | n/a | 0.207 ms | 7.91% | 48.1 MiB |

The retained forward is `7.51x` faster than the original Ascend path and removes one 12 MiB target-shape intermediate. The current 910B/A800 median gap is `12.86x`; it does not yet meet the final `2x` target. A800 exceeded the 5% CV threshold at 30 samples, so exactly one 50-sample confirmation was run and reported.

Rejected candidates included per-lane flat div/mod (`129.724 ms`), 1024-channel 1D blocks (`34.427 ms`), less channel-contiguous `128x32` tiles (`4.351 ms`), and 32/64-program persistent scheduling (`2.929/3.026 ms`). `BT=64,BD=256` was rejected at compile time because it required `2367488` UB bits versus the configured `1572864` bits.

Strict independent shifted-tensor references cover BF16/FP16, W=2/3/4, target and tail shapes, bias, SiLU, residual, and initial state. Five single-NPU tests, including NaN-poisoned output, pass. The frozen CP2 BF16 full forward/backward smoke also passes.

Stage 2 commit: `7b72944ce7ba6c8450418f25b13beda10fbc5ba6`.

## Stage 3 — dense backward pipeline

The retained dense backward path separates activation-gradient recomputation, `dx`, parameter reduction, and `dh0`. The `dpre`, `dx`, and boundary-only `dh0` kernels use the same grid-safe `64x128` task mapping as forward, FP32 accumulation, explicit round-to-nearest-even output casts, int64 address arithmetic, and `multibuffer=False`. The high-precision path retains FP32 `dpre`.

The initial two-level Triton `dw/db` implementation eliminated the original `O(NT*D*W)` workspace but remained program-bound. The best correct Triton version used 16 deterministic T splits and `BT=64,BD=8`; it measured `61.196 ms` for full forward/backward. A one-tap `BT=64,BD=64` version required `8470528` UB bits, while reducing it to `BT=8` compiled but regressed to `184.742 ms` because each tap reread `dpre` and each split executed 16 T loops.

The retained parameter-gradient implementation therefore uses the CANN/PyTorch fused FP32 multiply-reduction primitives. It processes 512-token chunks so the largest FP32 product temporary is about 6 MiB, accumulates each depthwise tap deterministically in FP32, and casts only the final `dw/db`. This is materially faster on 910B than lowering the reduction through Triton 3.2.0's Vector path. A full-T version reached `5.921 ms` aggregate latency but raised peak HBM to 132.1 MiB; chunking retains most of the speed at 105.1 MiB.

An additional address-clamping candidate replaced masked negative boundary addresses with 2D `tl.where` selections. Triton-Ascend materialized those selections and raised forward/dpre UB demand to about `16.8/18.9 Mbit` against the `1.57 Mbit` limit, so it was rejected; the retained masked-load form is covered by output and intermediate NaN poisoning.

The final `Tlocal=2048,D=3072,W=4,BF16,SiLU` 5-warmup/30-sample result is:

| Platform/path | p20 | p50 | p80 | CV | Peak HBM | Compile |
| ------------- | --: | --: | --: | -: | -------: | ------: |
| A800 production stack | 1.151 ms | 1.165 ms | 1.189 ms | 6.42% | 86.0 MiB | 0.500 s warm-cache |
| 910B Stage 1 | 110.825 ms | 111.294 ms | 111.698 ms | 0.61% | 111.2 MiB | 18.934 s cold-cache |
| 910B Stage 3 high | 6.232 ms | 6.251 ms | 6.298 ms | 0.80% | 105.1 MiB | 16.028 s cold-cache |

The aggregate Stage 1-to-Stage 3 speedup is `17.80x`, and the remaining 910B/A800 full-forward/backward gap is `5.37x`. Using the separately measured retained forward median of `2.667 ms`, the derived backward median is about `3.584 ms`: `25.29x` faster than the original 910B `90.653 ms`, with a remaining `4.08x` gap to the A800 `0.879 ms` backward baseline. The values are differences of separately synchronized medians and are labeled as derived rather than direct component timing.

The high path reduces total measured peak HBM by 5.5%, not the final 30% target. It completely removes the old partial `dw/db` workspace, but FP32 `dpre` and the chunk product remain live. The planned `a800` precision selector in Stage 5 will evaluate BF16 `dpre`; the 30% memory target remains open until that gate is measured rather than being claimed here.

Correctness coverage now includes BF16/FP16, W=2/3/4, short `T=2<W`, a non-tile tail, the target shape, bias/activation/state present and absent, and NaN-poisoned `dpre/dx/dw/db/dh0`. The short-sequence gate found and fixed an out-of-bounds `dh0` read when `T<W-1`. All other gradients retain the strict `1e-3` RMS gate. BF16 `dh0` uses a frozen A800-relative limit of `3.1e-3`: at the target shape A800 measures `2.766e-3` and 910B `2.847e-3`, a 2.9% difference within the agreed A800 `1.10x` envelope.

The CP reference is now an independent packed-sequence PyTorch FP32 implementation instead of another production kernel. The CP2 BF16 target smoke passes unchanged on both HCCL/910B and NCCL/A800.

Stage 3 commit: `0cc12b6a`.
