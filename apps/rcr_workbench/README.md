# RCR Scattering Workbench

The **RCR Scattering Workbench** is a local Streamlit application for electromagnetic scattering calculations of multi-sphere clusters. It provides a scientific GUI for configuring cluster morphologies (BCCA, BPCA, or custom coordinate uploads), launching CPU/GPU solver realizations, analyzing analytical random-orientation results, and exporting data tables and high-resolution figures.

## Prerequisites & Installation

To run the workbench, ensure you are using the virtual environment:

1. **Activate the environment and verify dependencies**:
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

2. **Verify CUDA/CuPy (Optional)**:
   GPU solvers are automatically enabled if CUDA and `cupy` are successfully imported on the system. If CuPy is not available, the app defaults to CPU realizations or Demo Mode.

## Launching the App

Start the Streamlit application from the repository root:

```bash
.venv/bin/streamlit run apps/rcr_workbench/app.py
```

This will automatically open the workbench in your default web browser (typically at `http://localhost:8501`).

---

## Workbench Features & Usage Guide

### 1. Home / Overview
- Displays active case configuration summary.
- If a result is loaded, key optical observables ($Q_{\text{ext}}, Q_{\text{sca}}, Q_{\text{abs}}, SSA, g$) are displayed as KPI metric cards.
- Offers **Quick Actions** to instantly load precomputed verified cases (e.g. $N=32$ or $N=64$ soot aggregates at $\lambda=550$ nm).

### 2. Geometry & Physical Setup (`1_Geometry.py`)
- **Morphology Selection**: Choose **BCCA** (Cluster-Cluster), **BPCA** (Particle-Cluster), or **Upload Coordinates**.
  - For BCCA, $N$ must be a power of 2.
  - For custom upload, you can upload CSV/XYZ files containing $X, Y, Z$ coordinate rows.
- **Physical Input**: Set wavelength ($\lambda$), monomer radius, complex refractive index ($n + ik$), random seed, and sphere truncation order ($p$).
- **VRAM / Footprint Estimation**: Dynamically calculates size parameters, matrix dimensions ($D$), and estimated dense matrix memory footprint, warning you if memory requirements exceed safe limits.
- **3D Visualization**: Render the aggregate structure interactively with Plotly.

### 3. Solver Execution (`2_Solver.py`)
- Choose between **CPU-Parallel**, **CPU-Serial**, **GPU Matrix-Free** (low VRAM), **GPU Dense Preassembled** (fastest for suitable matrix sizes), or **Demo Mode** solver realizations.
- Displays warning for heavy calculations on CPU before running.
- Executes the simulation and displays progress logs and completion metrics.

### 4. Results Viewer (`3_Results.py`)
- Summarizes calculated properties.
- Shows time breakdowns for geometry building, solving, and phase matrix evaluation.
- Renders interactive Plotly charts for the 6 independent orientation-averaged phase matrix elements ($F_{11}, F_{12}/F_{11}, F_{22}/F_{11}, F_{33}/F_{11}, F_{34}/F_{11}, F_{44}/F_{11}$).
- **Verification Overlay**: Toggle on "Overlay Reference (Demo) Data" to compare your live calculations against verified precomputed benchmark results.

### 5. Export Center (`4_Export.py`)
- **Numerical Exports**: Download `case.yaml`, `results.json`, `global_observables.csv`, and `phase_matrix.csv`.
- **High-Resolution Figures**: Renders 300 DPI publication-grade figures (PNG and vector PDF formats) for both the 3D aggregate geometry and the 6-panel phase matrix elements.
- **Markdown Report**: Download a formatted `summary_report.md` summarizing configurations, integrated properties, and timing profiles.

---

## Unit Testing

Run the automated test suite to verify the workbench modules:

```bash
.venv/bin/pytest tests/test_workbench_models.py tests/test_workbench_demo_loader.py tests/test_workbench_exports.py tests/test_workbench_runner_smoke.py
```
