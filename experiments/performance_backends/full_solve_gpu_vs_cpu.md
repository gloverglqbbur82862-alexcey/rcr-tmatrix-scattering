# Full Solve CPU vs GPU Benchmark Report
- **Metadata & Provenance**:
  - **Git Commit**: `7e1494b0d2e71e1d60c1c40acd1be74e39acf995`
  - **CPU Model**: AMD EPYC 9754 128-Core Processor
  - **GPU Model**: NVIDIA GeForce RTX 3090
  - **CPU Parallel Threads**: 64

## Summary Table

| N | Backend | Status | Solve Time (s) | Speedup vs CPU Serial | Speedup vs CPU Parallel | Qext |
|---|---------|--------|----------------|------------------------|--------------------------|------|
| 16 | `direct_cpu_serial` | Success | 23.836 | 1.00x | 1.00x | 1.15876289 |
| 16 | `direct_cpu_parallel` | Success | 23.825 | 1.00x | 1.00x | 1.15876289 |
| 16 | `direct_gpu` | Success | 9.386 | 2.54x | 2.54x | 1.15876289 |
| 16 | `direct_gpu_dense` | Success | 7.926 | 3.01x | 3.01x | 1.15876289 |
| 32 | `direct_cpu_serial` | Success | 37.945 | 1.00x | 0.71x | 1.50608726 |
| 32 | `direct_cpu_parallel` | Success | 26.967 | 1.41x | 1.00x | 1.50608726 |
| 32 | `direct_gpu` | Success | 9.766 | 3.89x | 2.76x | 1.50608726 |
| 32 | `direct_gpu_dense` | Success | 9.110 | 4.17x | 2.96x | 1.50608726 |
| 64 | `direct_cpu_serial` | Success | 89.595 | 1.00x | 0.32x | 2.01079368 |
| 64 | `direct_cpu_parallel` | Success | 28.253 | 3.17x | 1.00x | 2.01079368 |
| 64 | `direct_gpu` | Success | 10.236 | 8.75x | 2.76x | 2.01079368 |
| 64 | `direct_gpu_dense` | Success | 8.980 | 9.98x | 3.15x | 2.01079368 |
| 128 | `direct_cpu_serial` | Skipped (Time limit) | N/A | N/A | N/A | N/A |
| 128 | `direct_cpu_parallel` | Success | 81.447 | N/A | 1.00x | 2.49093092 |
| 128 | `direct_gpu` | Success | 14.830 | N/A | 5.49x | 2.49093092 |
| 128 | `direct_gpu_dense` | Success | 13.625 | N/A | 5.98x | 2.49093092 |
| 256 | `direct_cpu_serial` | Skipped (Time limit) | N/A | N/A | N/A | N/A |
| 256 | `direct_cpu_parallel` | Success | 449.796 | N/A | 1.00x | 3.15555374 |
| 256 | `direct_gpu` | Success | 46.401 | N/A | 9.69x | 3.15555374 |
| 256 | `direct_gpu_dense` | Success | 31.341 | N/A | 14.35x | 3.15555374 |

## Detailed Phase Breakdown

| N | Backend | Total Time (s) | Assembly (s) | H2D (s) | RHS Build (s) | Solve/Krylov (s) | Merge Total (s) | Merge Precompute (s) | Merge GEMM (ms) | Orientation (s) | Iters (SP/DP/Ref) | Final Res | Fallback? | VRAM (MB) |
|---|---------|----------------|--------------|---------|---------------|------------------|-----------------|----------------------|----------------|-----------------|-------------------|-----------|-----------|-----------|
| 16 | `direct_cpu_serial` | 23.836 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 14.0767 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 16 | `direct_cpu_parallel` | 23.825 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 13.9211 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 16 | `direct_gpu` | 9.386 | 0.0000 | 0.0000 | 5.4686 | 0.287 | 3.4014 | 3.3991 | 2.122 | 0.0269 | 0 / 10 / 0 | 0.00e+00 | No | 0.0 |
| 16 | `direct_gpu_dense` | 7.926 | 0.0031 | 0.0015 | 3.8852 | 0.121 | 3.3999 | 3.3974 | 2.260 | 0.0289 | 14 / 1 / 1 | 1.05e-10 | No | 3.5 |
| 32 | `direct_cpu_serial` | 37.945 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 14.2237 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 32 | `direct_cpu_parallel` | 26.967 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 14.5312 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 32 | `direct_gpu` | 9.766 | 0.0000 | 0.0000 | 5.3493 | 0.476 | 3.4206 | 3.4162 | 4.232 | 0.0555 | 0 / 9 / 0 | 0.00e+00 | No | 0.0 |
| 32 | `direct_gpu_dense` | 9.110 | 0.0091 | 0.0035 | 4.0929 | 0.598 | 3.5943 | 3.5890 | 5.107 | 0.0482 | 157 / 1 / 1 | 1.15e-10 | No | 14.1 |
| 64 | `direct_cpu_serial` | 89.595 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 14.0834 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 64 | `direct_cpu_parallel` | 28.253 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 13.9204 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 64 | `direct_gpu` | 10.236 | 0.0000 | 0.0000 | 5.3567 | 0.861 | 3.4009 | 3.3947 | 5.878 | 0.0538 | 0 / 10 / 0 | 0.00e+00 | No | 0.0 |
| 64 | `direct_gpu_dense` | 8.980 | 0.0249 | 0.0112 | 3.9192 | 0.705 | 3.4041 | 3.3978 | 6.050 | 0.0566 | 157 / 1 / 1 | 1.55e-10 | No | 56.2 |
| 128 | `direct_cpu_parallel` | 81.447 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 0.1839 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 128 | `direct_gpu` | 14.830 | 0.0000 | 0.0000 | 5.9156 | 2.597 | 3.4679 | 3.4421 | 25.429 | 0.1584 | 0 / 10 / 0 | 0.00e+00 | No | 0.0 |
| 128 | `direct_gpu_dense` | 13.625 | 0.0901 | 0.0384 | 4.4675 | 2.267 | 3.6350 | 3.6093 | 25.225 | 0.1621 | 158 / 1 / 1 | 2.89e-11 | No | 225.0 |
| 256 | `direct_cpu_parallel` | 449.796 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.000 | 0.3627 | 0 / 0 / 0 | N/A | N/A | 0.0 |
| 256 | `direct_gpu` | 46.401 | 0.0000 | 0.0000 | 8.7467 | 25.798 | 3.6554 | 3.5674 | 87.452 | 0.3517 | 0 / 11 / 0 | 0.00e+00 | No | 0.0 |
| 256 | `direct_gpu_dense` | 31.341 | 0.3307 | 0.1448 | 7.5173 | 10.786 | 3.8576 | 3.7713 | 85.692 | 0.3485 | 159 / 1 / 1 | 1.10e-09 | No | 900.0 |