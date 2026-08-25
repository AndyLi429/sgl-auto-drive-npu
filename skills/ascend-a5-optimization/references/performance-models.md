# A5 performance and roofline models

## Contents

- Units and notation
- Multi-roof model
- A5 balance reference points
- GEMM and fused epilogues
- Decode attention / GQA / MQA
- Communication and overlap
- Measurement discipline

## Units and notation

- Count one fused multiply-add as **2 FLOPs**.
- Published `TFLOPS`, `GB/s`, and `TB/s` are decimal SI (`10^12`, `10^9`, `10^12`).
- Use B for bytes; label binary tensor sizes as MiB/GiB.
- `F_c`, `F_v`: executed Cube and Vector FLOPs/ops.
- `Q_x`: bytes crossing path or memory level `x` after reuse, padding, retries, and temporary traffic.
- `P_c`, `P_v`: sustainable Cube/Vector compute roofs.
- `B_x`: sustainable bandwidth roof for path `x`.

## Multi-roof model

For each independently limiting resource:

```text
AI_x       = relevant_ops / Q_x                    [ops/B]
balance_x  = P_relevant / B_x                     [ops/B]
T_compute  = F / P
T_x        = Q_x / B_x
T_lower    = max(T_cube, T_vector, T_HBM, T_L2, T_CV, T_MTE..., T_comm)
```

`T_lower` assumes perfect legal overlap. Add serial dependency segments when the implementation cannot overlap them. If `T_measured >> T_lower`, inspect launch, scalar issue, synchronization, imbalance, tail tiles, bank/port conflicts, frequency throttling, cache misses, and inaccurate traffic assumptions.

Use the bundled calculator for one compute roof and one memory roof:

```bash
python scripts/roofline.py \
  --flops 8.589934592e9 --bytes 8.589934592e9 \
  --peak-tflops 425 --peak-gbs 1400
```

## A5 balance reference points

These are **derived from official theoretical peaks**, not sustained measurements.

| Product/bin | HBM TB/s | Total BF16 balance FLOP/B | Cube BF16 balance FLOP/B | Vector BF16 balance FLOP/B |
|---|---:|---:|---:|---:|
| 950PR full (486 total, 432 Cube, 54 Vector) | 1.6 | 303.75 | 270.00 | 33.75 |
| Atlas 350 / 950PR bin (425, 378, 47) | 1.4 | 303.57 | 270.00 | 33.57 |
| 950DT 36C/72V (547, 486, 60) | 4.0 | 136.75 | 121.50 | 15.00 |
| 950DT 32C/64V (486, 432, 54) | 4.0 | 121.50 | 108.00 | 13.50 |
| 950DT 28C/56V (425, 378, 47) | 4.0 | 106.25 | 94.50 | 11.75 |

Choose the row matching the deployed SKU. Use Cube balance for dense matrix work and Vector balance for vector-only work. For a mixed kernel, model both; combined peak does not prove both can be reached simultaneously.

## GEMM and fused epilogues

For `C[M,N] = A[M,K] @ B[K,N]`:

```text
F_cube ≈ 2*M*N*K
Q_min  = eA*M*K + eB*K*N + eC*M*N          # beta=0, one ideal read/write
AI_min = F_cube / Q_min
```

Actual `Q_HBM` must include repeated weight/input loads not served by L2/local reuse, partial-result traffic, padding, transposes, and temporary tensors. For a fused bias/activation/quantization epilogue, model `F_vector`, L0C→UB/CV traffic, conversion traffic, and output bytes separately.

Local tile feasibility:

```text
S_live = buffers * (A_tile + B_tile + accum_tile + vector_temp + conversion_temp)
S_live <= usable local capacity for every participating memory
```

Do not use the nominal 512 KB UB/L1 or 256 KB L0C as fully usable without accounting for static allocations, queues, SIMT Data Cache, compiler workspace, double buffering, and alignment.

## Decode attention / GQA / MQA

For one generated token per sequence:

- `Hq`: query heads
- `Hkv`: KV heads
- `S`: cached sequence length
- `D`: head dimension
- `e`: KV element bytes

Ignoring small Q/output traffic and softmax scalar work:

```text
F_QK+PV    ≈ 4 * Hq * S * D
Q_KV       ≈ 2 * Hkv * S * D * e
AI_KV      ≈ 2 * Hq / (Hkv * e)
```

For BF16 (`e=2`), `AI_KV ≈ Hq/Hkv`: MHA is about 1 FLOP/B; GQA with 8 query heads per KV head is about 8 FLOP/B if the implementation reads each KV head once and reuses it across its query group. Batch size multiplies both FLOPs and bytes and does not change ideal arithmetic intensity.

For `B=32, Hq=Hkv=64, S=8192, D=128, BF16`:

```text
per output token: 268,435,456 FLOPs; 268,435,456 B KV traffic; AI = 1 FLOP/B
per batch step:    8,589,934,592 FLOPs; 8,589,934,592 B; AI = 1 FLOP/B
```

That point is far below every published A5 BF16/HBM balance in the table, so ideal MHA decode is HBM-bound on all listed A5 bins. This is a derived theoretical classification; confirm sustained bandwidth, paging/layout, duplicate KV reads, cache behavior, and fusion with profiling.

Non-fused score/probability materialization adds approximately `2*Hq*S*e_score` bytes per token for one write plus one read, plus any extra passes. Flash/decode fusion should eliminate this traffic.

## Communication and overlap

For collective or remote-memory work, add:

```text
T_comm_lower = message_bytes / effective_link_bandwidth + latency_terms
```

Use topology-effective bandwidth, not the 2 TB/s chip aggregate, when only a subset of ports/links is active. PCIe, UBoE, UB/URMA, and HBM are separate roofs. For MoE, model dispatch and combine traffic separately and include imbalance/padding.

For a producer/consumer CV pipeline:

```text
T_steady_tile >= max(T_cube, T_cv_transfer, T_vector)
T_total ≈ startup + n_tiles*T_steady_tile + drain
```

If barriers or buffer reuse serialize stages, replace the `max` with the actual critical-path sum. A faster Cube tile can worsen performance when it increases CV/Vector pressure or reduces reusable local capacity.

## Measurement discipline

1. Calibrate separate microbenchmarks for Cube, Vector, HBM read/write, GM↔UB, GM↔L1, L1↔L0, L0C/fixpipe, direct CV, and communication.
2. Match dtype, layout, alignment, tile size, concurrency, cache state, frequency, ECC, and power mode to the target workload.
3. Sweep shapes and tail cases; report median plus a tail statistic after warmup.
4. Compare fused vs decomposed traffic and time. Validate numerical accuracy before accepting reduced precision or in-flight quantization.
5. Report `observed / published_peak`, `observed / calibrated_roof`, and `T_measured / T_lower`; they answer different questions.

