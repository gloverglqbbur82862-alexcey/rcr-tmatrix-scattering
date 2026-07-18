#!/usr/bin/env python3
"""Plot manuscript figures for the Optics Express paper from JSON result files."""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_png_and_pdf(fig, out_dir, stem):
    """Save a Matplotlib figure as high-resolution PNG and vector PDF."""
    png_path = Path(out_dir) / f"{stem}.png"
    pdf_path = Path(out_dir) / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def load_result_safe(directory, filename):
    filepath = Path(directory) / filename
    if not filepath.exists():
        print(f"Warning: {filename} not found in {directory}. Figure will skip this case.", file=sys.stderr)
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Error loading {filename}: {exc}", file=sys.stderr)
        return None


def plot_fig1_parity(out_dir):
    """Figure 1: MSTM vs RCR Qext Parity and Difference."""
    # Verified data from detailed_work_report.md
    N_vals = np.array([4, 8, 16, 32, 64, 128])
    mstm_q = np.array([0.63154, 0.86602, 1.16050, 1.48500, 1.91560, 2.47330])
    rcr_q = np.array([0.6315419685, 0.8660198540, 1.1605109947, 1.4849924375, 1.9155928712, 2.4732545401])
    
    diff_pct = (rcr_q - mstm_q) / mstm_q * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Parity scatter plot
    ax1.plot([0.5, 2.6], [0.5, 2.6], 'k--', alpha=0.5, label='1:1 Parity Line')
    ax1.scatter(mstm_q, rcr_q, color='blue', edgecolor='darkblue', s=70, zorder=5, label='BCCA Aggregates')
    ax1.set_xlabel('MSTM $Q_{\\mathrm{ext}}$ (Reference)', fontsize=11)
    ax1.set_ylabel('Python RCR $Q_{\\mathrm{ext}}$ (Calculated)', fontsize=11)
    ax1.set_title('RCR vs MSTM $Q_{\\mathrm{ext}}$ Parity', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    # Right: Difference vs N
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.plot(N_vals, diff_pct, 'o-', color='purple', linewidth=2, markersize=8, label='Relative Error (%)')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(N_vals)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel('Aggregate Size $N$ (Monomers)', fontsize=11)
    ax2.set_ylabel('Difference $(Q_{\\mathrm{rcr}} - Q_{\\mathrm{mstm}})/Q_{\\mathrm{mstm}} \\times 100\\%$', fontsize=11)
    ax2.set_title('Relative Discrepancy vs. System Size', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_png_and_pdf(fig, out_dir, "fig1_mstm_parity_summary")
    plt.close()


def plot_fig2_mueller(results_dir, out_dir):
    """Figure 2: Mueller matrix angular comparisons for BCCA vs BPCA (N=16)."""
    wavelengths = [450, 550, 650]
    colors = {450: "blue", 550: "green", 650: "red"}
    
    # Load cases
    bcca_cases = {}
    bpca_cases = {}
    
    for wl in wavelengths:
        bcca_cases[wl] = load_result_safe(results_dir, f"bcca_n16_w{wl}_seed42.json")
        bpca_cases[wl] = load_result_safe(results_dir, f"bpca_n16_w{wl}_seed42.json")
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Check if any case loaded
    has_data = any(c is not None for c in (list(bcca_cases.values()) + list(bpca_cases.values())))
    if not has_data:
        print("Skip Figure 2: No N=16 cases found in results directory.", file=sys.stderr)
        plt.close()
        return

    # Subplots setup
    sub_configs = [
        # (row, col, key, ylabel, title, log_y)
        (0, 0, "F11", "Phase function $P_{11}(\\theta)$", "Unpolarized phase function $P_{11}$", True),
        (0, 1, "F12_over_F11", "Polarization ratio $P_{12}/P_{11}$", "Polarization ratio $P_{12}/P_{11}$", False),
        (1, 0, "F33_over_F11", "Ratio $P_{33}/P_{11}$", "Ratio $P_{33}/P_{11}$", False),
        (1, 1, "F34_over_F11", "Ratio $P_{34}/P_{11}$", "Ratio $P_{34}/P_{11}$", False)
    ]

    for row, col, key, ylabel, title, log_y in sub_configs:
        ax = axs[row, col]
        
        # Plot BCCA
        for wl in wavelengths:
            c = bcca_cases[wl]
            if c is not None:
                theta = c["theta_deg"]
                val = c["mueller"][key]
                ax.plot(theta, val, color=colors[wl], linestyle='-', linewidth=2, label=f"BCCA ({wl} nm)")
                
        # Plot BPCA
        for wl in wavelengths:
            c = bpca_cases[wl]
            if c is not None:
                theta = c["theta_deg"]
                val = c["mueller"][key]
                ax.plot(theta, val, color=colors[wl], linestyle='--', linewidth=1.5, label=f"BPCA ({wl} nm)")

        ax.set_xlabel("Scattering Angle $\\theta$ (deg)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.grid(True, linestyle=':', alpha=0.5)
        if log_y:
            ax.set_yscale('log')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_png_and_pdf(fig, out_dir, "fig2_bcca_bpca_mueller_wavelengths")
    plt.close()


def plot_fig3_scaling(results_dir, out_dir):
    """Figure 3: Size scaling metrics Qext, Qsca, SSA, g vs N."""
    N_vals = [8, 16, 32, 64]
    
    bcca_obs = {"N": [], "Qext": [], "Qsca": [], "SSA": [], "g": []}
    bpca_obs = {"N": [], "Qext": [], "Qsca": [], "SSA": [], "g": []}

    # Load BCCA Tier B
    for n in N_vals:
        c = load_result_safe(results_dir, f"bcca_n{n}_w550_seed42.json")
        if c is not None:
            bcca_obs["N"].append(n)
            bcca_obs["Qext"].append(c["observables"]["Qext"])
            bcca_obs["Qsca"].append(c["observables"]["Qsca"])
            bcca_obs["SSA"].append(c["observables"]["SSA"])
            bcca_obs["g"].append(c["observables"]["g"])

    # Load BPCA Tier B
    for n in N_vals:
        c = load_result_safe(results_dir, f"bpca_n{n}_w550_seed42.json")
        if c is not None:
            bpca_obs["N"].append(n)
            bpca_obs["Qext"].append(c["observables"]["Qext"])
            bpca_obs["Qsca"].append(c["observables"]["Qsca"])
            bpca_obs["SSA"].append(c["observables"]["SSA"])
            bpca_obs["g"].append(c["observables"]["g"])

    if not bcca_obs["N"] and not bpca_obs["N"]:
        print("Skip Figure 3: No scaling cases found in results directory.", file=sys.stderr)
        return

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    
    metrics = [
        # (row, col, key, ylabel, title)
        (0, 0, "Qext", "Extinction Efficiency $Q_{\\mathrm{ext}}$", "Extinction $Q_{\\mathrm{ext}}$ vs. size $N$"),
        (0, 1, "Qsca", "Scattering Efficiency $Q_{\\mathrm{sca}}$", "Scattering $Q_{\\mathrm{sca}}$ vs. size $N$"),
        (1, 0, "SSA", "Single Scattering Albedo (SSA)", "SSA vs. size $N$"),
        (1, 1, "g", "Asymmetry Parameter $g$", "Asymmetry Parameter $g$ vs. size $N$")
    ]

    for row, col, key, ylabel, title in metrics:
        ax = axs[row, col]
        
        # Plot BCCA
        if bcca_obs["N"]:
            ax.plot(bcca_obs["N"], bcca_obs[key], 'o-', color='blue', linewidth=2, markersize=7, label='BCCA')
            
        # Plot BPCA
        if bpca_obs["N"]:
            ax.plot(bpca_obs["N"], bpca_obs[key], 's--', color='red', linewidth=2, markersize=7, label='BPCA')

        ax.set_xscale('log', base=2)
        ax.set_xticks(N_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Aggregate Size $N$ (Monomers)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right' if key in ["g", "SSA"] else 'upper left')

    plt.tight_layout()
    save_png_and_pdf(fig, out_dir, "fig3_size_scaling_metrics")
    plt.close()


def write_figure_manifest(out_dir):
    manifest_path = Path(out_dir) / "figure_manifest.md"
    content = """# Figure Manifest for Optics Express Validation Package

This folder contains manuscript-ready figures generated from the RCR simulation results.

## Main Manuscript Figures Generated by This Script

### Wavelength-dependent Mueller observables
- **Files**: `fig2_bcca_bpca_mueller_wavelengths.png`, `fig2_bcca_bpca_mueller_wavelengths.pdf`
- **Caption**: Comparison of the orientation-averaged phase-matrix components $P_{11}(\\theta)$, $P_{12}(\\theta)/P_{11}(\\theta)$, $P_{33}(\\theta)/P_{11}(\\theta)$, and $P_{34}(\\theta)/P_{11}(\\theta)$ over $0$--$180^\\circ$ scattering angles for representative BCCA (solid lines) and BPCA (dashed lines) $N=16$ aggregates at wavelengths $\\lambda = 450$, 550, and 650 nm.

### Size scaling
- **Files**: `fig3_size_scaling_metrics.png`, `fig3_size_scaling_metrics.pdf`
- **Caption**: Global optical observables $Q_{\\text{ext}}$, $Q_{\\text{sca}}$, single-scattering albedo, and asymmetry parameter $g$ as functions of monomer count $N \\in \\{8, 16, 32, 64\\}$ at $\\lambda = 550$ nm for representative BCCA and BPCA aggregates.

## Auxiliary Validation Asset

### MSTM parity summary
- **Files**: `fig1_mstm_parity_summary.png`, `fig1_mstm_parity_summary.pdf`
- **Use**: Auxiliary validation material only. The Optics Express manuscript now reports this evidence in Table 1 rather than as a main-text figure.
"""
    with open(manifest_path, "w") as f:
        f.write(content)
    print(f"Written figure manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot manuscript figures from simulation results")
    parser.add_argument("--results", default="experiments/jqsrt_application/results", help="Directory containing result JSON files")
    parser.add_argument("--out", default="figures/jqsrt_application", help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating manuscript figures...")
    plot_fig1_parity(out_dir)
    plot_fig2_mueller(results_dir, out_dir)
    plot_fig3_scaling(results_dir, out_dir)
    write_figure_manifest(out_dir)
    print("All figures completed successfully.")


if __name__ == "__main__":
    main()
