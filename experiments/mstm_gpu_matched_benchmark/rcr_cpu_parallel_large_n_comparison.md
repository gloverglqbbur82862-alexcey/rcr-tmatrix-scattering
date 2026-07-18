# RCR CPU-Parallel Large-N Matched Timing Comparison

This note compares the RCR `direct_cpu_parallel` backend with the existing
same-workstation MSTM MPI-8 and RCR GPU matrix-free timing artifacts.  All
entries use the same frozen geometry for the same `N`, the same optical
parameters, random-orientation output, and the MSTM-estimated aggregate order
used in the corresponding RCR runs.

## Sources

- `rcr_cpu_parallel_n256_matched_results.json`
- `rcr_cpu_parallel_n512_probe_results.json`
- `gpu_mstm_matched_n256_formal_results.json`
- `gpu_mstm_matched_n512_probe_results.json`

## Summary

| N | Geometry | Solver | Repeats | Compute time (s) | Process/wall time (s) | Qext | Relative to MSTM MPI-8 |
|---:|---|---|---:|---:|---:|---:|---:|
| 256 | Two-cluster stress geometry | MSTM MPI-8 | 2 | 258.300 | 258.300 | 3.13760 | 1.00x |
| 256 | Two-cluster stress geometry | RCR CPU parallel, 64 threads | 2 | 436.625 | 474.054 | 3.137598521 | 0.59x |
| 256 | Two-cluster stress geometry | RCR GPU matrix-free | 2 | 62.044 | 96.092 | 3.137598521 | 4.16x |
| 512 | Two-cluster stress geometry | MSTM MPI-8 | 1 | 1924.294 | 1924.294 | 3.96940 | 1.00x |
| 512 | Two-cluster stress geometry | RCR CPU parallel, 64 threads | 1 | 2792.512 | 2829.929 | 3.969363094 | 0.69x |
| 512 | Two-cluster stress geometry | RCR GPU matrix-free | 1 | 334.594 | 368.944 | 3.969363094 | 5.75x |

`Relative to MSTM MPI-8` is computed from the compute-window time for RCR and
the MSTM wall time.  Values below 1 mean slower than MSTM MPI-8.

## Interpretation

The RCR CPU-parallel backend is a valid multicore realization of the same RCR
operator and reproduces the scalar extinction values of the MSTM and GPU
matrix-free runs.  It is not, however, the performance-leading backend in these
large-N matched checks.  At `N=256`, RCR CPU parallel is slower than MSTM MPI-8
by about `1.69x` in the compute window.  At `N=512`, the one-repeat probe is
slower than MSTM MPI-8 by about `1.45x`.

The stronger performance claim remains the accelerator-oriented matrix-free GPU
path.  The CPU-parallel backend should be described as a portable multicore
realization and fallback path, not as the basis for a faster-than-MSTM claim.
