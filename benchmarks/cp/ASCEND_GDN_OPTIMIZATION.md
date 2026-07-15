# Ascend GDN Context-Parallel Optimization Record

This record tracks the staged Ascend 910B optimization of GDN context-parallel preprocessing against the repository's A800 implementation. KDA and DPLR/RWKV7 remain correctness-only in this phase.

## Frozen contract

- Target shape: BF16, `Tglobal=16384`, `H=HV=8`, `K=V=128`, `BT=64`, CP2/CP4/CP8.
- CP8 always uses all eight physical 910B devices exclusively.
- Public GDN output and every gradient retain the strict RMS-ratio gate `<3e-3`, with finite and NaN-poisoning checks.
- The `high` H/dH/M/dM path and merge retain `<1e-4`.
- An `a800` H/dH/M/dM specialization may be promoted only when each supported shape stays within `1.10x` the independently measured A800 RMS-error envelope and passes the unchanged public gate.
- Triton compilation is completed before timing. Candidate timing uses five warmups and 20 samples; final evidence uses five warmups and 30 samples.
- The HCCL payload, wire shape, collective count, and rank order are frozen.

The runtime selector is `FLA_ASCEND_CP_GDN_PRECISION=high|a800`. `high` is the default. The `a800` mode only dispatches a promoted shape specialization; unsupported shapes fall back to `high`.

## Baselines

All latency values are the slowest-rank wall-clock median in milliseconds. The A800 and 910B measurements use the same benchmark shape and exclude compilation.

| Device/path        | CP2 fwd | CP2 bwd | CP4 fwd | CP4 bwd | CP8 fwd | CP8 bwd |
| ------------------ | ------: | ------: | ------: | ------: | ------: | ------: |
| A800 CUDA          |   0.803 |   0.871 |   0.528 |   0.571 |   0.399 |   0.424 |
| 910B high, current |   1.767 |   2.306 |   1.355 |   1.533 |   1.203 |   1.093 |

The current 910B CP4-to-CP8 scaling efficiency is 56.3% forward and 70.1% backward, versus 66.2% and 67.3% on A800. Absolute latency remains outside the target and optimization continues.

## Stage results

### Stage 1: Forward transition precision study

The A800 CUDA transition and two 910B Cube candidates were compared with the same independent FP32 recurrence. The range-stress case is `K=128,V=64,BT=32,T=70,input_scale=0.2`; the CP8 case is `K=V=128,BT=64,Tlocal=2048,input_scale=0.05`.

| Transition arithmetic       | Range RMS ratio | CP8 RMS ratio | Isolated M | CP8 end-to-end | Decision |
| --------------------------- | --------------: | ------------: | ---------: | -------------: | -------- |
| A800 repository CUDA        |       1.486e-3 |     2.139e-2 |          — |              — | Numerical envelope |
| 910B high FP32              |       0.000e+0 |     3.368e-7 |   0.896 ms |       1.172 ms | Retain |
| 910B FP16 Cube for `W @ M`  |       2.646e-4 |     1.971e-2 |   0.857 ms |       1.186 ms | Reject |
| 910B BF16 Cube for `W @ M`  |       2.157e-3 |     2.448e-3 |   0.845 ms |       1.260 ms | Reject |

Both Cube variants accelerate isolated M, but they move M onto the same Cube resource used by the concurrent H scan. That contention removes the microbenchmark gain and regresses eight-card end-to-end latency by 1.2% for FP16 and 7.5% for BF16. BF16 also exceeds the A800 range-stress envelope. No forward low-precision path is promoted; the implementation remains on the high-precision path.

Both `high` and the candidate `a800` mode passed the CP8 public output/all-gradient gate before the performance decision. The rejected candidates were reverted after measurement.

### Stage 2: Backward fused dH/dM precision path

The unmodified A800 backward dM ratios are `1.476e-3` for `K=V=128,BT=32,T=70,input_scale=0.2` and `2.139e-2` for the CP8 target. The promoted candidate uses native BF16 Cube for both dM contractions inside the fused dH/dM kernel. dH arithmetic is unchanged.

| Candidate                         | Range dM ratio | CP8 dM ratio | CP8 backward result                       | Decision |
| --------------------------------- | -------------: | -----------: | ----------------------------------------- | -------- |
| High FP32                         |      0.000e+0 |    3.403e-7 | 1.129/1.127 ms in the paired runs         | Retain as default |
| BF16 first contraction only       |      2.099e-3 |    2.484e-3 | +1.7% then -4.6%; not repeatable           | Reject |
| FP16 first contraction only       |      2.608e-4 |    1.592e-2 | 1.109 ms vs 1.090 ms high                 | Reject |
| FP16 both contractions            |      3.757e-4 |    3.870e-2 | Not timed after exceeding the 2.353e-2 cap | Reject |
| BF16 both contractions            |      3.049e-3 |    3.436e-3 | 1.035/1.030 ms vs 1.129/1.127 ms high     | Promote for the CP8 target |

The BF16-both path improves paired eight-card backward medians by 8.4% and 8.6%. Its CP8 dM error is about 16% of the A800 error, and the public CP8 output/all-gradient test passes with a worst observed RMS ratio of `5e-6`. The short-sequence BF16 result is outside the A800+10% envelope, so dispatch is deliberately limited to BF16 `K=V=128,Tlocal=2048`; every other dtype, dimension, and local length uses `high` even when the environment requests `a800`.

A periodic mixed FP32/BF16 update was also rejected before timing: a runtime branch over loop-carried dM state silently produced an `8.65e4` error ratio on the CP8 case under the current compiler. The promoted kernel contains no runtime arithmetic branch; `A800_PRECISION` is a compile-time specialization.

### Stage 3: Final CP2/CP4/CP8 matrix after precision promotion

The final matrix uses five warmups and 30 samples per direction. CP2/CP4 use physical devices starting at device 2; CP8 exclusively uses all eight physical devices. Values are slowest-rank wall-clock medians in milliseconds.

| CP size | A800 fwd | 910B fwd | 910B/A800 | A800 bwd | 910B bwd | 910B/A800 |
| ------: | -------: | -------: | --------: | --------: | --------: | --------: |
|       2 |    0.793 |    1.828 |     2.306 |     0.861 |     2.293 |     2.664 |
|       4 |    0.520 |    1.363 |     2.623 |     0.559 |     1.477 |     2.645 |
|       8 |    0.394 |    1.237 |     3.139 |     0.416 |     1.052 |     2.531 |

The corresponding 910B throughput is 43.4%/37.5% of A800 at CP2, 38.1%/37.8% at CP4, and 31.9%/39.5% at CP8 for forward/backward. The final A800 medians are within 2% of or faster than the frozen baseline in every case, so the CUDA regression gate passes.

| Gate | Result | Status |
| ---- | ------ | ------ |
| Public GDN output/all gradients `<3e-3` | CP8 `a800` worst observed ratio `5e-6`; CP4 fallback and high primitives pass | pass |
| 910B latency `<=1.10x` A800 | Ratios are 2.306-3.139 forward and 2.531-2.664 backward | fail |
| 910B throughput `>=90%` A800 | 31.9%-43.4% | fail |
| A800 CUDA regression `<=2%` | All six final medians are no slower than the frozen baselines | pass |
| CP4-to-CP8 efficiency within 10 percentage points of A800 | Forward 55.1% vs 65.9% (10.8 pp gap); backward 70.2% vs 67.2% | forward fail; backward pass |

Median variability was low except for isolated high-tail samples in A800 CP4 backward and 910B CP4 forward/CP8 backward. Their medians, p10, and p90 remain clustered; the full logs retain CV values rather than discarding the outliers. The performance objective is not complete after this stage, and further work must target the H/dH scan and fixed distributed overhead rather than further relaxing transition precision.

### Stage 4: H/dH envelope and critical-path decomposition

The repository A800 H/dH kernels and the 910B `high` path were compared with the same independent FP32 recurrence. The short case is `K=V=128,BT=32,T=70,input_scale=0.2`; the CP8 case is `K=V=128,BT=64,Tlocal=2048,input_scale=0.05`.

| Local state | A800 short | 910B high short | A800 CP8 | 910B high CP8 | A800+10% CP8 cap |
| ----------- | ---------: | --------------: | --------: | -------------: | ----------------: |
| H           |  7.051e-8 |        2.443e-8 | 5.128e-4 |       3.086e-4 |          5.640e-4 |
| dH          |  1.880e-3 |        1.586e-4 | 1.358e-3 |       4.919e-4 |          1.494e-3 |

The current 910B serial path is more accurate than A800 on all four measurements, so the user-approved A800-parity mode has numerical room without changing the public `<3e-3` gate. The previously rejected four-way associative scan remains outside that room: its CP8 H/dH ratios are `2.47e-3/2.44e-3`, or 4.4x/1.6x the respective A800+10% caps. It is not revived. A two-way scan is the next bounded candidate because it changes fewer recurrence boundaries and can be rejected before timing if either cap fails.

Production-equivalent single-device diagnostics include gate precomputation and the actual concurrent-stream/fused scheduling. Values below are three warmups and ten samples, with compilation excluded.

| Local T | fwd gate | gate+H | gate+M | concurrent local fwd | bwd gate | fused local bwd |
| ------: | -------: | -----: | -----: | -------------------: | -------: | --------------: |
|    2048 | 0.272 ms | 0.621 ms | 0.718 ms | 0.849 ms | 0.238 ms | 0.742 ms |
|    4096 |        — |        — |        — | 1.120 ms |        — | 1.263 ms |
|    8192 | 0.380 ms | 1.174 ms | 1.438 ms | 1.627 ms | 0.323 ms | 2.173 ms |

The CP8 row uses the promoted BF16 dM specialization; CP2/CP4-sized backward rows automatically use `high`. Precomputed H/M timings include the shared gate launch, so they diagnose the production schedule but are not additive. The old standalone H/M modes recompute gates inside each scan and are deliberately excluded from this table.

An exclusive eight-card decomposition of the unchanged FP32 summary protocol measured a `0.353 ms` all-gather median, `0.284/0.286 ms` forward/backward merge medians, and `0.482/0.495 ms` combined communication-plus-merge medians. These are slowest-rank values; stage maxima are not additive because the boundary rank that has the longest merge skips the local summary. Nevertheless, the collective alone consumes 81% of the `0.433 ms` CP8 forward budget and 77% of the `0.457 ms` backward budget implied by the 1.10x A800 target. Local scans and the fixed distributed path must both improve; an H/dH-only change cannot meet the absolute target. The HCCL dtype, shape, collective count, and rank order remain unchanged.
