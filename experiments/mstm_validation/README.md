# Frozen MSTM Validation Artifacts

Created: 2026-06-07  
Purpose: preserve raw MSTM v4.0 input/output evidence for the JCP manuscript
validation claim that the RCR solver matches MSTM random-orientation extinction
for BCCA soot aggregates with `N = 4, 8, 16, 32, 64, 128`.

## Contents

| Path | Role |
|---|---|
| `run_mstm_validation.py` | Generates MSTM inputs, optionally runs MSTM, and parses Qext/Qsca/order |
| `inputs/*.inp` | Frozen MSTM input decks |
| `outputs/*.out` | Frozen raw MSTM output files |
| `mstm_validation_manifest.json` | Machine-readable parsed validation manifest |
| `temp_pos.dat`, `tmattemp.dat` | MSTM-generated transient files retained for provenance |

## Command Used

```bash
./.venv/bin/python3 experiments/mstm_validation/run_mstm_validation.py \
  --run \
  --mstm-bin /path/to/mstm
```

## Physical Setup

| Quantity | Value |
|---|---|
| Aggregate family | BCCA |
| Random seed | `42` |
| Monomer radius | `25.0 nm` |
| Wavelength | `550.0 nm` |
| Radius size parameter | `0.28559933214452665` |
| Refractive index | `1.95 + 0.79i` |
| MSTM solution epsilon | `1.d-12` |
| MSTM translation epsilon | `1.d-12` |
| Random orientation | `true` |
| Scattering matrix calculation | `true` |

## Parsed Parity

| N | MSTM Qext | RCR Qext from `state.yaml` | Difference in `state.yaml` | Parsed vs state MSTM |
|---:|---:|---:|---:|---:|
| 4 | 0.63154 | 0.6315419685 | +0.0003% | 0.0% |
| 8 | 0.86602 | 0.8660198540 | -0.00002% | 0.0% |
| 16 | 1.16050 | 1.1605109947 | +0.0009% | 0.0% |
| 32 | 1.48500 | 1.4849924375 | -0.0005% | 0.0% |
| 64 | 1.91560 | 1.9155928712 | -0.0004% | 0.0% |
| 128 | 2.47330 | 2.4732545401 | -0.0018% | 0.0% |

## Manuscript Use

Allowed claim:

> For BCCA soot aggregates with `N = 4..128`, the RCR solver reproduces MSTM
> v4.0 random-orientation extinction efficiencies within `0.002%`.

Do not use these artifacts to claim speed superiority over MSTM. The MSTM run
times here are serial local reference timings and are not a fair performance
comparison against the remote GPU benchmark.

