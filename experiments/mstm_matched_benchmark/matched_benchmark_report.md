# Matched MSTM/RCR Timing Benchmark

This artifact records a same-geometry local timing comparison. It is not a
claim that the present RCR implementation is generally faster than MSTM.

## Timing Policy

- Input-deck generation and aggregate-coordinate generation are excluded from timed values.
- MSTM wall time includes executable startup, input parsing, solve, and output-file writing.
- RCR timed_window_s is measured inside a Python subprocess after a small JIT/library warm-up; it includes solver construction, aggregate T-matrix construction, and orientation averaging.
- RCR process_wall_time_s is retained in raw trials but is not used for the summary table.
- GPU RCR backends are only included when the local Python environment provides CuPy/CUDA.
- Timings are local-machine measurements and must not be mixed with the separate remote GPU benchmark as a same-hardware comparison.

## Environment

- `timestamp_utc`: 2026-06-11T17:36:06Z
- `platform`: macOS-26.2-arm64-arm-64bit-Mach-O
- `python`: 3.13.3
- `git_commit`: 668d16d84331f98d20271219f9f4c73ccbf386b6
- `gfortran`: GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0
- `mpifort`: GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0
- `mpirun`: mpirun (Open MPI) 5.0.7
- `mstm_mpi_build`: run `experiments/mstm_matched_benchmark/build_mstm_parallel.sh` before reproducing MPI timings.

## Summary

| N | Solver | Repeats | Median time (s) | Min time (s) | Median Qext | Time / MSTM serial |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `mstm_serial` | 2 | 0.0395148 | 0.0390507 | 0.63154 | 1.000x |
| 4 | `mstm_mpi_np4` | 2 | 0.108646 | 0.105941 | 0.63154 | 2.750x |
| 4 | `rcr_direct_cpu_parallel_t8` | 2 | 2.18408 | 2.17373 | 0.6315419685 | 55.272x |
| 8 | `mstm_serial` | 2 | 0.150602 | 0.150435 | 0.86602 | 1.000x |
| 8 | `mstm_mpi_np4` | 2 | 0.146191 | 0.145309 | 0.86602 | 0.971x |
| 8 | `rcr_direct_cpu_parallel_t8` | 2 | 2.29834 | 2.27909 | 0.866019854 | 15.261x |
| 16 | `mstm_serial` | 2 | 0.852349 | 0.846224 | 1.1605 | 1.000x |
| 16 | `mstm_mpi_np4` | 2 | 0.374507 | 0.373835 | 1.1605 | 0.439x |
| 16 | `rcr_direct_cpu_parallel_t8` | 2 | 2.55835 | 2.54859 | 1.160510995 | 3.002x |
| 32 | `mstm_serial` | 2 | 5.93041 | 5.85086 | 1.485 | 1.000x |
| 32 | `mstm_mpi_np4` | 2 | 1.93867 | 1.92204 | 1.485 | 0.327x |
| 32 | `rcr_direct_cpu_parallel_t8` | 2 | 4.54551 | 4.54254 | 1.484992439 | 0.766x |

## Manuscript Use

Allowed: cite this as a local matched timing artifact with the timing
policy above, and use it to explain that MSTM has serial and MPI
realizations while the current RCR advantage is structural and
accelerator-oriented.

Not allowed: claim broad speed superiority over MSTM from this artifact
alone, or mix these local CPU/MPI timings with the separate remote GPU
benchmark as if they were collected on the same hardware.
