# JQSRT / Optics Express Application Verification Sweeps

This folder contains deterministic configuration parameters and runners to perform multi-sphere soot scattering simulation sweeps using the RCR engine.

## Layout

- `cases.json`: Database defining case parameters for Tier A (wavelength sensitivity) and Tier B (size scaling).
- `run_case.py`: CLI script to run a single case from `cases.json` and output JSON results.
- `run_batch.py`: Python orchestrator using subprocess case execution with controlled thread budgets.
- `plot_figures.py`: Manuscript figure generation script using generated JSON results.
- `export_tables.py`: Script to extract CSV summary sheets and markdown benchmark descriptors.

## Case Tiers

- **Tier A**:
  - Targets BCCA and BPCA clusters of $N=16$ monomers.
  - Sweeps wavelengths: 450, 550, 650 nm.
  - Refractive index is locked at $m = 1.95 + 0.79i$.
  - Truncation orders are set to monomer $p=3$ and aggregate $p_{\text{agg}}=6$.
- **Tier B**:
  - Size scaling sweeps for $N=8, 16, 32, 64$ at wavelength 550 nm.
- **Tier C (Optional)**:
  - Wavelength spectral sweep (300-900 nm) to produce extinction spectra. Not run by default.

## Execution

To execute Tier A:
```bash
./run_tier_a.sh
```

To execute Tier B:
```bash
./run_tier_b.sh
```
