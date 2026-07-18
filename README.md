# Fast RCR T-matrix Scattering

This repository contains Python source code, tests, benchmark scripts, and
small validation data for fast rotation-coaxial-rotation (RCR) T-matrix
calculations of light scattering by sphere aggregates.

The code supports direct RCR aggregate construction, matrix-free operator
application, analytical random-orientation averaging, and example BCCA/BPCA
application cases. The hierarchical/FMM modules are included as experimental
research code and are not used for the main optical observables reported in the
manuscript.

## Repository Contents

- `rcr_engine/`: Wigner rotation and coaxial translation operators.
- `math_core/`: numerical special-function support.
- `geometry/`: BCCA/BPCA aggregate geometry generation and utilities.
- `solver/`: direct and matrix-free multi-sphere RCR solvers.
- `fmm/`: experimental hierarchical/FMM-related modules.
- `apps/rcr_workbench/`: Streamlit workbench for interactive cases.
- `examples/`: example case configuration files.
- `benchmark/`: benchmark and comparison scripts.
- `experiments/`: validation and application scripts with compact result data.
- `demo_data/`: small demonstration result files for the workbench.
- `tests/`: pytest validation and regression tests.

## Installation

Use Python 3.10-3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

## Tests

Run the core test suite from the repository root:

```bash
python -m pytest -q
```

Some GPU, remote-workstation, or external-reference tests require optional
dependencies and local environment configuration.

## MSTM Validation

MSTM is used as an external validation reference and is not redistributed in
this repository. To regenerate MSTM comparisons, obtain and build MSTM v4.0
from the original public distribution, then pass the executable path explicitly:

```bash
python experiments/mstm_validation/run_mstm_validation.py --run --mstm-bin /path/to/mstm
```

Matched benchmark scripts similarly require `--mstm-bin` or the corresponding
MPI executable path. Existing compact input/output files are included only to
document the validation cases used during development.

## Remote GPU Workbench

Remote execution in `apps/rcr_workbench` is optional. It is configured only
through environment variables:

```bash
export RCR_REMOTE_HOST=your-host
export RCR_REMOTE_REPO=/path/to/remote/checkout
export RCR_REMOTE_PYTHON=python
```

If these variables are not set, remote solver modes fail with a configuration
error instead of using any machine-specific default.

## Data Availability

The included JSON/YAML files provide compact examples, validation records, and
figure-source-style data. Larger archival data packages, if used for a paper
submission, should be deposited in a public repository such as Zenodo, Figshare,
OSF, or an institutional archive.

## Citation

If you use this code, cite the associated manuscript when available. MSTM
comparisons should also cite the original MSTM publication and distribution.

## License

This project is distributed under the MIT License. See `LICENSE`.
