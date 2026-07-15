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

## Stage 4 — fixed halo and communication A/B

The CP wire tensor is now always `[W-1,D]`. Local chunks shorter than the halo are right-aligned and zero-padded on the left, and backward adds the received gradient to only the valid local tail. This fixes both forward construction and backward accumulation when `Tlocal<W-1`. The default all-gather route remains API-compatible.

An opt-in `FLA_CP_CONV_COMM=p2p` route uses `batch_isend_irecv` and process-group-local peers, so non-contiguous CP subgroups do not confuse global and group ranks. Send buffers are made contiguous before entering NCCL/HCCL. Invalid selector values fail before a collective. HCCL/910B gates pass CP2 BF16 with both methods and a CP4 complex packed-sequence case; NCCL/A800 gates pass both CP2 methods. Both platforms also pass forward/backward exchange on the non-contiguous global-rank subgroup `[0,2]`.

Candidate end-to-end `Tglobal=16384,D=3072,W=4,BF16,SiLU` results use three warmups and ten samples:

| CP | All-gather p50 | P2P p50 | P2P change | All-gather CV | P2P CV |
| --: | -------------: | -------: | ---------: | ------------: | -------: |
| 2 | 24.219 ms | 23.215 ms | 4.15% faster | 2.77% | 0.17% |
| 4 | 12.311 ms | 12.215 ms | 0.78% faster | 0.78% | 0.50% |
| 8 | 6.687 ms | 6.761 ms | 1.12% slower | 1.32% | 3.89% |

At fixed global tokens, the corresponding CP2-to-CP8 strong-scaling efficiencies are 90.5% for all-gather and 85.8% for P2P. These are candidate-stage measurements; Stage 5 performs the final five-warmup/30-sample comparison.

Communication-only measurements include one forward and one backward halo exchange:

| CP | All-gather p50 | P2P p50 | All-gather peak | P2P peak |
| --: | -------------: | -------: | --------------: | -------: |
| 2 | 0.443 ms | 0.533 ms | 112 KiB | 94 KiB |
| 4 | 0.476 ms | 0.631 ms | 148 KiB | 94 KiB |
| 8 | 0.564 ms | 0.661 ms | 220 KiB | 94 KiB |

The initial CP8 P2P communication run had an outlier and 37.5% CV. Per the frozen timing protocol, exactly one 50-sample confirmation was run; it measured p20/p50/p80 `0.644/0.661/0.675 ms` with 4.05% CV. P2P therefore does not meet the promotion rule of at least 5% CP8 end-to-end improvement with no greater than 2% CP2/4 regression. All-gather remains the default, while P2P is retained behind the explicit switch for reproducible A/B testing. No communication-overlap candidate was merged because the synchronous P2P primitive did not first establish a benefit.

Stage 4 commit: `1c0d674a`.

## Stage 5 — precision gate, fixed schedule, and final matrix

`FLA_ASCEND_CONV_PRECISION=high|a800` is now validated by the Ascend backend. `high` remains the default. The selector maintains an explicit promotion bucket list, but that list is intentionally empty after both reduced-precision candidates failed at least one immutable gate. An `a800` request therefore reports requested `a800`, effective `high`, and safely executes the general FP32-`dpre` path. Unsupported or unpromoted shapes never enter dead-reckoned low precision.

The target `Tlocal=2048,D=3072,W=4,BF16` numerical study used the same independent FP32 reference and inputs on both platforms:

| Path | output | dx | dw | db | dh0 |
| ---- | -----: | -: | -: | -: | --: |
| A800 production | 7.84e-6 | 7.64e-4 | 7.81e-4 | 6.07e-4 | 2.77e-3 |
| 910B BF16 `dpre` | 1.25e-5 | 2.69e-3 | 3.62e-3 | 2.56e-3 | 3.49e-3 |
| 910B FP16 `dpre` | 1.25e-5 | 9.31e-4 | 8.31e-4 | 1.62e-3 | 2.85e-3 |

BF16 `dpre` reduced the single-rank peak from 105.1 MiB to 90.1 MiB, but regressed p50 from 6.238 ms to 6.351 ms and exceeded the A800-relative error envelope. FP16 `dpre` measured 6.412 ms, retained a 105.1 MiB measured peak because mixed BF16/FP16 reductions created promoted temporaries, and failed the `db` gate. Neither candidate met the planned 30% memory reduction, correctness, or speed requirements, so neither is reachable from the public selector.

Backward scheduling was tuned independently from the frozen forward tile. Candidate results use three warmups and ten samples at D3072:

| Backward tile/schedule | fwd+bwd p50 | Decision |
| ---------------------- | -----------: | -------- |
| `64x128`, 2 warps | 6.238 ms | baseline |
| `32x256`, 2 warps | 7.304 ms | reject, 17.1% slower |
| `128x64`, 2 warps | 6.211 ms | reject, within noise |
| `64x128`, 4 warps | 6.027 ms | retain for target buckets |
| `128x64`, 4 warps | 6.201 ms | reject |

The retained four-warp schedule is restricted to contiguous BF16 `W=4,D=1024/3072,T>=64`; all other dense cases keep two warps. At D1024 its candidate p50 changed from 3.474 ms to 3.349 ms. The final five-warmup/30-sample run is reported below rather than substituting the shorter candidate timing.

### Final single-rank kernel-only results

All times are synchronized wall-clock. Backward is derived as `fwd+bwd - fwd` from separately synchronized medians.

| D | Platform | Forward p20/p50/p80 | Fwd+bwd p20/p50/p80 | CV fwd/total | Derived bwd | Peak total | 910B/A800 fwd/total/bwd |
| --: | -------- | --------------------: | ------------------------: | -----------: | ----------: | ---------: | ----------------------: |
| 1024 | A800 | 0.202/0.208/0.215 ms | 0.933/0.947/0.963 ms | 7.42%/11.24% | 0.739 ms | 28.7 MiB | 1.00x/1.00x/1.00x |
| 1024 | 910B | 1.430/1.445/1.465 ms | 3.435/3.463/3.490 ms | 1.27%/0.81% | 2.019 ms | 36.1 MiB | 6.96x/3.66x/2.73x |
| 3072 | A800 | 0.202/0.205/0.219 ms | 0.933/0.948/0.959 ms | 7.15%/12.18% | 0.743 ms | 86.0 MiB | 1.00x/1.00x/1.00x |
| 3072 | 910B | 2.590/2.601/2.615 ms | 5.987/6.001/6.023 ms | 0.54%/0.38% | 3.399 ms | 105.1 MiB | 12.71x/6.33x/4.57x |

Every A800 kernel-only 30-sample run exceeded the 5% CV trigger, so exactly one 50-sample confirmation is shown. Its remaining long tails are retained in the CV instead of being filtered. The current A800 D3072 forward is 1.1% faster than the frozen Stage 2 value and fwd+bwd is 18.6% faster than the Stage 1 value, so the unchanged CUDA kernel has no measured regression greater than 2%.

The required kernel-only `910B/A800 <=2x` target is not met. Compared with the original exploratory 910B component baselines, D3072 forward improves from 20.028 ms to 2.601 ms (`7.70x`) and derived backward from 90.653 ms to 3.399 ms (`26.67x`), but the final gaps remain dominated by the four-tap vector forward and FP32 activation/reduction work.

### Final CP2/4/8 end-to-end results

All results use all-gather, five warmups, and 30 samples. A point whose CV exceeded 5% was replaced by exactly one 50-sample confirmation. The p50 comparison is:

| D | CP | 910B p50 | A800 p50 | 910B/A800 | 910B global tokens/s |
| --: | --: | --------: | --------: | ---------: | -------------------: |
| 1024 | 2 | 12.551 ms | 1.912 ms | 6.56x | 1.31M |
| 1024 | 4 | 7.160 ms | 2.035 ms | 3.52x | 2.29M |
| 1024 | 8 | 4.510 ms | 2.055 ms | 2.19x | 3.63M |
| 3072 | 2 | 23.315 ms | 1.987 ms | 11.73x | 0.70M |
| 3072 | 4 | 12.377 ms | 2.088 ms | 5.93x | 1.32M |
| 3072 | 8 | 6.700 ms | 2.225 ms | 3.01x | 2.45M |

The 910B CP2-to-CP4, CP4-to-CP8, and overall CP2-to-CP8 strong-scaling efficiencies are respectively `87.6%/79.4%/69.6%` for D1024 and `94.2%/92.4%/87.0%` for D3072. Thus D3072 passes the 80% scaling target while D1024 does not. CP8 D1024 passes the end-to-end `<=2.5x` A800 gap target; D3072 remains at `3.01x` and does not pass. A800 itself has only about 22%-23% CP2-to-CP8 efficiency at this fixed global length because its sub-millisecond local kernel is dominated by fixed launch/collective synchronization overhead; this is why the absolute CP8 gap contracts even though 910B does not reach the kernel-only target.

CP8 correctness passes on both HCCL/910B and NCCL/A800 for D1024 and D3072. The A800 CP8 BF16 `dw` RMS baseline is `4.335e-3` and 910B is `4.342e-3`; the CP8 parameter-gradient gate is therefore frozen at their A800 `1.10x` envelope (`4.8e-3`). CP2 keeps its `3.3e-3` gate. Output and `dx` remain below `1e-3` for every world size.

The first cold distributed specializations took about 14 s on the 910B stack and 38 s on the A800 stack. Compilation is reported independently and excluded from every warm latency sample.

Stage 5 commit: pending.
