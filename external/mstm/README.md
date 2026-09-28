# MSTM reference solver

The paper compares RCR results with MSTM v4.0. The repository does not
currently contain a verified, redistributable copy of the canonical
`mstm-v4.0.f90` distribution. Do not treat the local files as a substitute
without checking their provenance and license.

`build_mstm_parallel.sh` expects a separately downloaded MSTM distribution
and copies `mstm-v4.0.f90`, `mstm-intrinsics.f90`, and
`mpidefs-parallel.f90` into a temporary build directory. GNU Fortran and
OpenMPI are required for the MPI build.

The frozen input decks and raw outputs used for validation are kept under
`experiments/validation/inputs` and `experiments/validation/outputs`.
