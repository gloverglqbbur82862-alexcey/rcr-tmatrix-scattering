# Same-Workstation GPU-RCR versus MSTM Benchmark

This artifact records a same-physical-workstation comparison. MSTM is run
inside WSL2 as CPU serial and, when available, MPI-parallel executables.
RCR GPU backends are run from native Windows Python/CuPy on the same reference workstation
workstation. It is therefore a same-machine benchmark, not a same-runtime
benchmark.

## Timing Policy

- Aggregate-coordinate generation and input-deck generation are excluded from timed values.
- MSTM wall time is measured around the WSL2 command invocation after a one-command WSL warm-up; it includes executable startup, input parsing, solve, and output-file writing.
- RCR process/wall time includes native Python subprocess startup plus JIT/GPU warm-up and the physical computation.
- RCR compute-window time excludes the small dummy JIT/GPU warm-up and includes solver construction, aggregate T-matrix construction, and orientation averaging.
- For each N, all solvers use the same frozen sphere-aggregate geometry, monomer radius, wavelength, refractive index, random-orientation output, and tolerance policy recorded here.
- MSTM MPI timings use the OpenMPI rank counts recorded in the environment metadata.

## Environment

- `timestamp_utc`: 2026-06-11T19:37:47Z
- `platform`: Windows-11-10.0.26100-SP0
- `python`: 3.13.2
- `git_commit`: unavailable
- `wsl_distro`: Ubuntu-24.04
- `mstm_wsl_binary`: /root/mstm_build/mstm
- `mstm_mpi_wsl_binary`: /root/mstm_build_parallel/mstm_parallel
- `requested_mpi_procs`: [8]
- `wsl_gfortran`: GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- `wsl_mpifort`: /usr/bin/mpifort
- `wsl_mpirun`: /usr/bin/mpirun
- `nvidia_smi`: 0, NVIDIA GeForce RTX 3090, 24576 MiB

## Geometry Sources

- `N=256`: `experiments\mstm_gpu_matched_benchmark\frozen_inputs\mstm_bcca_n256_w550_seed42.inp`
  - provenance: deterministic_doubled_frozen_geometry; Two translated copies of the frozen N=128 BCCA seed-42 geometry; used only as a frozen N=256 performance stress-test geometry, not as an independently generated BCCA seed-42 realization.

## Summary

| N | Solver | Repeats | Median compute time (s) | Median process/wall time (s) | Median Qext | Speedup vs fastest MSTM |
|---:|---|---:|---:|---:|---:|---:|
| 256 | `mstm_wsl_mpi_np8` | 2 | 258.3 | 258.3 | 3.1376 | 1.00x |
| 256 | `rcr_direct_gpu_t64` | 2 | 62.0436 | 96.092 | 3.137598521 | 4.16x |
| 256 | `rcr_direct_gpu_dense_t64` | 2 | 84.4315 | 122.39 | 3.137598521 | 3.06x |

## Manuscript Use

Allowed: use successful rows as a same-workstation GPU-RCR versus MSTM
CPU/MPI context, with the WSL2/native-Windows distinction stated explicitly.
Use the RCR process/wall time column when making conservative statements;
use the RCR compute-window column only when also stating that JIT/library
warm-up is excluded.

Not allowed: claim general large-N superiority beyond the tested geometry,
truncation, repeat count, and MPI rank count.
