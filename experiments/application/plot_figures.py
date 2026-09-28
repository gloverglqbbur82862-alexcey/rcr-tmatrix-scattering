#!/usr/bin/env python3
"""Plot manuscript figures for the Optics Express paper from ensemble JSON result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ensemble_summary import group_results, load_results, parse_seed_spec, mueller_mean_std


def save_png_and_pdf(fig, out_dir, stem):
    png_path = Path(out_dir) / f"{stem}.png"
    pdf_path = Path(out_dir) / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def find_group(groups, family, n_monomers, wavelength_nm, tier=None):
    for key, group in groups.items():
        if key[0] != family:
            continue
        if key[1] != n_monomers:
            continue
        if key[3] != wavelength_nm:
            continue
        if tier is not None and key[8] != tier:
            continue
        return group
    return None


def maybe_add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        return handles, labels
    return None, None


def filter_results_by_seed_spec(results_list, seed_spec: str):
    seed_filter = set(parse_seed_spec(seed_spec))
    return [r for r in results_list if r["input_parameters"]["seed"] in seed_filter]


def plot_fig4_mueller_ensemble(results_dir, out_dir):
    results = filter_results_by_seed_spec(load_results(results_dir), "0-9")
    groups = group_results([r for r in results if r["input_parameters"]["n_monomers"] == 16 and r["input_parameters"]["wavelength_nm"] in (450.0, 550.0, 650.0)])
    if not groups:
        print("Skip Figure 4: No N=16 ensemble cases found.", file=sys.stderr)
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    configs = [
        (0, 0, "F11", "P11", True),
        (0, 1, "F12_over_F11", "P12/P11", False),
        (1, 0, "F33_over_F11", "P33/P11", False),
        (1, 1, "F34_over_F11", "P34/P11", False),
    ]
    titles = {
        "F11": "Unpolarized phase function $P_{11}$",
        "F12_over_F11": "Polarization ratio $P_{12}/P_{11}$",
        "F33_over_F11": "Ratio $P_{33}/P_{11}$",
        "F34_over_F11": "Ratio $P_{34}/P_{11}$",
    }
    colors = {450.0: "tab:blue", 550.0: "tab:green", 650.0: "tab:red"}
    linestyles = {"bcca": "-", "bpca": "--"}

    for row, col, key, ylabel, log_y in configs:
        ax = axs[row, col]
        for family in ("bcca", "bpca"):
            for wl in (450.0, 550.0, 650.0):
                group = find_group(groups, family, 16, wl, tier="A")
                if group is None:
                    continue
                theta, mean_curve, std_curve = mueller_mean_std(group)[key]
                ax.plot(theta, mean_curve, color=colors[wl], linestyle=linestyles[family], linewidth=1.8, label=f"{family.upper()} {int(wl)} nm")
                ax.fill_between(theta, mean_curve - std_curve, mean_curve + std_curve, color=colors[wl], alpha=0.15)
        ax.set_xlabel("Scattering angle $\\theta$ (deg)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(titles[key], fontsize=11, fontweight="bold")
        if log_y:
            ax.set_yscale("log")

    handles, labels = maybe_add_legend(axs[0, 0])
    if handles is None:
        handles, labels = maybe_add_legend(axs[0, 1])
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_png_and_pdf(fig, out_dir, "fig4_bcca_bpca_mueller_ensemble")
    plt.close(fig)


def plot_fig5_scaling_ensemble(results_dir, out_dir):
    results = filter_results_by_seed_spec(load_results(results_dir), "0-9")
    groups = group_results([r for r in results if r["input_parameters"]["wavelength_nm"] == 550.0 and r["input_parameters"]["n_monomers"] in (8, 16, 32, 64)])
    if not groups:
        print("Skip Figure 5: No scaling ensemble cases found.", file=sys.stderr)
        return

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    metrics = [
        (0, 0, "Qext", "Extinction efficiency $Q_{\\rm ext}$"),
        (0, 1, "Qsca", "Scattering efficiency $Q_{\\rm sca}$"),
        (1, 0, "SSA", "Single-scattering albedo"),
        (1, 1, "g", "Asymmetry parameter $g$"),
    ]
    titles = {
        "Qext": "Extinction efficiency $Q_{\\rm ext}$",
        "Qsca": "Scattering efficiency $Q_{\\rm sca}$",
        "SSA": "Single-scattering albedo",
        "g": "Asymmetry parameter $g$",
    }
    colors = {"bcca": "tab:blue", "bpca": "tab:red"}

    for row, col, key, ylabel in metrics:
        ax = axs[row, col]
        for family, marker in (("bcca", "o"), ("bpca", "s")):
            xs = []
            means = []
            stds = []
            for n in (8, 16, 32, 64):
                group = find_group(groups, family, n, 550.0)
                if group is None:
                    continue
                xs.append(n)
                vals = np.array([item["observables"][key] for item in group], dtype=float)
                means.append(vals.mean())
                stds.append(vals.std(ddof=1) if vals.size > 1 else 0.0)
            if xs:
                ax.errorbar(xs, means, yerr=stds, fmt=marker + "-", color=colors[family], capsize=3, linewidth=1.8, markersize=6, label=family.upper())
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Aggregate size $N$ (monomers)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(titles[key], fontsize=11, fontweight="bold")

    handles, labels = maybe_add_legend(axs[0, 0])
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_png_and_pdf(fig, out_dir, "fig5_size_scaling_ensemble")
    plt.close(fig)


def write_figure_manifest(out_dir):
    manifest_path = Path(out_dir) / "figure_manifest.md"
    content = """# Figure Manifest for Optics Express Validation Package

This folder contains manuscript-ready figures generated from the RCR simulation results.

## Main Manuscript Figures Generated by This Script

### Ensemble Mueller observables
- **Files**: `fig4_bcca_bpca_mueller_ensemble.png`, `fig4_bcca_bpca_mueller_ensemble.pdf`
- **Caption**: Ensemble mean orientation-averaged phase-matrix components with one-standard-deviation shading for BCCA and BPCA aggregates at wavelengths 450, 550, and 650 nm, using seeds 0-9.

### Ensemble size scaling
- **Files**: `fig5_size_scaling_ensemble.png`, `fig5_size_scaling_ensemble.pdf`
- **Caption**: Ensemble mean global observables $Q_{\\text{ext}}$, $Q_{\\text{sca}}$, SSA, and $g$ with one-standard-deviation error bars for BCCA and BPCA aggregates at $\\lambda = 550$ nm, using seeds 0-9.
"""
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"Written figure manifest to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot manuscript figures from simulation results")
    default_results = Path(__file__).parent / "results" / "seed0-9"
    parser.add_argument("--results", default=str(default_results), help="Directory containing result JSON files")
    parser.add_argument("--out", default="figures/generated", help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating manuscript figures...")
    plot_fig4_mueller_ensemble(results_dir, out_dir)
    plot_fig5_scaling_ensemble(results_dir, out_dir)
    write_figure_manifest(out_dir)
    print("All figures completed successfully.")


if __name__ == "__main__":
    main()
