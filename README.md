# RCR T-Matrix Scattering

Reproducible code and data for the matrix-free rotation-coaxial-rotation
(RCR) T-matrix method for light scattering by sphere aggregates.

The implementation supports BCCA and BPCA aggregate geometries, direct RCR
translations, batched aggregate T-matrix construction, BiCGSTAB solution,
analytical random-orientation averaging, Mueller-matrix observables, and
CPU/GPU execution paths.

## Repository layout

```text
src/rcr_scattering/      Core geometry, RCR, math, and solver code
experiments/application/  BCCA/BPCA application cases and result readers
experiments/validation/   Frozen MSTM input/output validation cases
experiments/performance/  Backend and hardware benchmarks
data/                     Frozen results, tables, and MSTM artifacts
figures/                  Manuscript and generated figures
configs/                  Reproducible experiment configurations
scripts/                  End-to-end convenience entry points
tests/                    Numerical and regression tests
external/mstm/            MSTM build instructions and local source artifacts
```

The formal ten-realization ensemble is in
`experiments/application/results/seed0-9/`. Files in the parent `results/`
directory are demonstration or newly generated outputs and should not be
mixed into the formal ensemble without an explicit audit.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```powershell
python -m pip install -e ".[dev]"
```

The optional `gpu` extra installs CuPy for CUDA 12. The correct CuPy package
must match the installed CUDA runtime and GPU driver.

## Reproducing results

Use the existing frozen seed-0--9 JSON files to regenerate tables and figures:

```powershell
python scripts/reproduce_figures.py
```

Run application cases (requires the numerical dependencies and sufficient
CPU/GPU resources):

```powershell
python scripts/reproduce_ensembles.py --tier A --seeds 0-9
python scripts/reproduce_ensembles.py --tier B --seeds 0-9
```

Run the frozen validation wrapper:

```powershell
python scripts/reproduce_validation.py
```

Executing MSTM itself requires a separately obtained MSTM v4.0 distribution;
see `external/mstm/README.md`. The repository retains the validation input
decks and raw outputs used for the reported comparison.

## Tests

```powershell
python -m pytest -q
```

GPU-specific tests require a working CUDA/CuPy installation and are skipped
or unavailable on CPU-only systems.

## Reproducibility parameters

The main application cases use 25 nm monomers, refractive index
`1.95 + 0.79i`, monomer order `p=3`, aggregate order `P=6`, and 181 angular
samples. Ensemble statistics use independent seeds 0 through 9. Numerical
precision is double precision unless a backend explicitly documents
otherwise.

## Scope and citation

This repository contains research software and frozen evidence for the
associated manuscript. MSTM is an external reference solver and remains
subject to its own license and citation requirements. Please cite the paper
and this repository when using the implementation.
