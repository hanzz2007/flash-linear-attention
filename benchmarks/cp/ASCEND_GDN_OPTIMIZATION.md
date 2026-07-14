# Ascend GDN Context-Parallel Optimization Record

This record tracks the staged Ascend 910B optimization of GDN context-parallel preprocessing against the repository's A800 implementation. KDA and DPLR/RWKV7 remain correctness-only in this phase.

## Frozen contract

- Target shape: BF16, `Tglobal=16384`, `H=HV=8`, `K=V=128`, `BT=64`, CP2/CP4/CP8.
- CP8 always uses all eight physical 910B devices exclusively.
- Public GDN output and every gradient retain the strict RMS-ratio gate `<3e-3`, with finite and NaN-poisoning checks.
- Internal H, dH, and merge gates retain `<1e-4`.
- The high-precision M/dM path retains `<1e-4` and must remain available.
- An A800-parity M/dM path may be promoted only when each supported shape stays within the independently measured A800 error envelope and passes the unchanged public gate.
- Triton compilation is completed before timing. Candidate timing uses five warmups and 20 samples; final evidence uses five warmups and 30 samples.
- The HCCL payload, wire shape, collective count, and rank order are frozen.

The planned runtime selector is `FLA_ASCEND_CP_GDN_PRECISION=high|a800`. `high` is the default. The `a800` mode will only dispatch a promoted shape specialization; unsupported shapes fall back to `high`. The selector is not added until at least one A800-parity path improves the end-to-end target.

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
