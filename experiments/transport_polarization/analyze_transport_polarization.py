#!/usr/bin/env python3
"""Analyze transport and polarization characteristics of single scattering.

This post-processing experiment uses the existing application JSON outputs.
It does not rerun the RCR solver.  The derived quantities are intended to add
physical interpretation beyond Qext/Qsca/SSA/g and the raw Mueller curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("bcca", "bpca")
COLORS = {"bcca": "tab:blue", "bpca": "tab:red"}
MARKERS = {"bcca": "o", "bpca": "s"}
LINESTYLES = {450.0: ":", 550.0: "-", 650.0: "--"}
WAVELENGTH_LABELS = {450.0: "450 nm", 550.0: "550 nm", 650.0: "650 nm"}


def parse_seed_spec(seed_spec: str) -> Set[int]:
    seeds: Set[int] = set()
    for token in seed_spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = token.split("-", 1)
            seeds.update(range(int(start), int(stop) + 1))
        else:
            seeds.add(int(token))
    return seeds


def load_results(results_dir: Path, seed_spec: str) -> List[Dict[str, Any]]:
    seed_filter = parse_seed_spec(seed_spec)
    items: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        seed = int(result["input_parameters"]["seed"])
        if seed in seed_filter:
            items.append(result)
    return items


def result_key(result: Dict[str, Any]) -> Tuple[Any, ...]:
    ip = result["input_parameters"]
    return (
        ip["family"],
        int(ip["n_monomers"]),
        float(ip["wavelength_nm"]),
        float(ip["radius_nm"]),
        float(ip["m_real"]),
        float(ip["m_imag"]),
        int(ip["monomer_order"]),
        int(ip["aggregate_order"]),
        ip.get("tier", ""),
    )


def group_results(results: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result_key(result)].append(result)
    for group in groups.values():
        group.sort(key=lambda item: int(item["input_parameters"]["seed"]))
    return groups


def angle_index(theta: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(theta - target)))


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def half_power_width_deg(theta: np.ndarray, p11: np.ndarray) -> float:
    """Approximate angular width where P11 remains above half its maximum."""
    if p11.size == 0:
        return float("nan")
    threshold = 0.5 * float(np.max(p11))
    active = theta[p11 >= threshold]
    if active.size == 0:
        return 0.0
    return float(active[-1] - active[0])


def derive_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    ip = result["input_parameters"]
    obs = result["observables"]
    theta = np.asarray(result["theta_deg"], dtype=float)
    theta_rad = np.deg2rad(theta)
    mueller = result["mueller"]
    p11 = np.asarray(mueller["F11"], dtype=float)
    f12_over_f11 = np.asarray(mueller["F12_over_F11"], dtype=float)
    f34_over_f11 = np.asarray(mueller["F34_over_F11"], dtype=float)

    # Optical convention: positive linear polarization is reported as -P12/P11.
    linear_pol = -f12_over_f11
    circular_conversion = f34_over_f11

    max_linear_idx = int(np.argmax(np.abs(linear_pol)))
    max_circular_idx = int(np.argmax(np.abs(circular_conversion)))
    idx_0 = angle_index(theta, 0.0)
    idx_90 = angle_index(theta, 90.0)
    idx_180 = angle_index(theta, 180.0)

    denominator = integrate_trapezoid(p11 * np.sin(theta_rad), theta_rad)
    numerator = integrate_trapezoid(p11 * np.cos(theta_rad) * np.sin(theta_rad), theta_rad)
    g_from_p11 = numerator / denominator if abs(denominator) > 1.0e-300 else float("nan")

    qsca = float(obs["Qsca"])
    g = float(obs["g"])
    qtr = qsca * (1.0 - g)

    return {
        "case_id": result["case_id"],
        "family": ip["family"],
        "n_monomers": int(ip["n_monomers"]),
        "radius_nm": float(ip["radius_nm"]),
        "wavelength_nm": float(ip["wavelength_nm"]),
        "seed": int(ip["seed"]),
        "monomer_order": int(ip["monomer_order"]),
        "aggregate_order": int(ip["aggregate_order"]),
        "tier": ip.get("tier", ""),
        "qsca": qsca,
        "g": g,
        "qtr": qtr,
        "max_abs_linear_pol": float(abs(linear_pol[max_linear_idx])),
        "theta_at_max_abs_linear_pol_deg": float(theta[max_linear_idx]),
        "linear_pol_at_90deg": float(linear_pol[idx_90]),
        "linear_pol_at_180deg": float(linear_pol[idx_180]),
        "max_abs_circular_conversion": float(abs(circular_conversion[max_circular_idx])),
        "theta_at_max_abs_circular_conversion_deg": float(theta[max_circular_idx]),
        "circular_conversion_at_90deg": float(circular_conversion[idx_90]),
        "circular_conversion_at_180deg": float(circular_conversion[idx_180]),
        "forward_backward_p11_ratio": float(p11[idx_0] / p11[idx_180]) if abs(p11[idx_180]) > 1.0e-300 else float("nan"),
        "p11_half_power_width_deg": half_power_width_deg(theta, p11),
        "g_from_p11": float(g_from_p11),
        "abs_g_consistency_error": float(abs(g_from_p11 - g)),
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["family"],
            row["n_monomers"],
            row["wavelength_nm"],
            row["radius_nm"],
            row["monomer_order"],
            row["aggregate_order"],
            row["tier"],
        )
        groups[key].append(row)

    metrics = [
        "qsca",
        "g",
        "qtr",
        "max_abs_linear_pol",
        "theta_at_max_abs_linear_pol_deg",
        "linear_pol_at_90deg",
        "linear_pol_at_180deg",
        "max_abs_circular_conversion",
        "theta_at_max_abs_circular_conversion_deg",
        "circular_conversion_at_90deg",
        "circular_conversion_at_180deg",
        "forward_backward_p11_ratio",
        "p11_half_power_width_deg",
        "g_from_p11",
        "abs_g_consistency_error",
    ]
    summaries: List[Dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda item: (item[0], item[1], item[2])):
        group = groups[key]
        family, n_monomers, wavelength_nm, radius_nm, monomer_order, aggregate_order, tier = key
        item: Dict[str, Any] = {
            "family": family,
            "n_monomers": n_monomers,
            "wavelength_nm": wavelength_nm,
            "radius_nm": radius_nm,
            "monomer_order": monomer_order,
            "aggregate_order": aggregate_order,
            "tier": tier,
            "seed_count": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda r: r["seed"])),
        }
        for metric in metrics:
            m, s = mean_std([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = m
            item[f"{metric}_std"] = s
        summaries.append(item)
    return summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def save_png_and_pdf(fig, out_dir: Path, stem: str) -> None:
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def find_summary(
    summaries: List[Dict[str, Any]],
    *,
    family: str,
    n_monomers: int,
    wavelength_nm: float,
) -> Optional[Dict[str, Any]]:
    for item in summaries:
        if item["family"] == family and item["n_monomers"] == n_monomers and item["wavelength_nm"] == wavelength_nm:
            return item
    return None


def plot_transport_scaling(summaries: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (0, 0, "qtr", "Transport scattering efficiency $Q_{\\rm tr}=Q_{\\rm sca}(1-g)$"),
        (0, 1, "max_abs_linear_pol", "Maximum linear polarization $\\max|-P_{12}/P_{11}|$"),
        (1, 0, "max_abs_circular_conversion", "Maximum circular conversion $\\max|P_{34}/P_{11}|$"),
        (1, 1, "forward_backward_p11_ratio", "Forward/backward phase-function ratio"),
    ]
    sizes = [8, 16, 32, 64]
    for row, col, metric, ylabel in panels:
        ax = axs[row, col]
        for family in FAMILIES:
            xs = []
            means = []
            stds = []
            for n in sizes:
                item = find_summary(summaries, family=family, n_monomers=n, wavelength_nm=550.0)
                if item is None:
                    continue
                xs.append(n)
                means.append(item[f"{metric}_mean"])
                stds.append(item[f"{metric}_std"])
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    fmt=MARKERS[family] + "-",
                    color=COLORS[family],
                    capsize=3,
                    linewidth=1.8,
                    markersize=6,
                    label=family.upper(),
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Aggregate size $N$ (monomers)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_png_and_pdf(fig, out_dir, "fig_transport_polarization_scaling")
    plt.close(fig)


def group_raw_by_key(results: List[Dict[str, Any]]) -> Dict[Tuple[str, int, float], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
    for item in results:
        ip = item["input_parameters"]
        groups[(ip["family"], int(ip["n_monomers"]), float(ip["wavelength_nm"]))].append(item)
    for group in groups.values():
        group.sort(key=lambda item: int(item["input_parameters"]["seed"]))
    return groups


def mueller_curve_mean_std(group: List[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(group[0]["theta_deg"], dtype=float)
    stack = []
    for item in group:
        theta_i = np.asarray(item["theta_deg"], dtype=float)
        if theta_i.shape != theta.shape or not np.allclose(theta_i, theta, rtol=0.0, atol=1e-12):
            raise ValueError("Inconsistent theta grids within a group")
        stack.append(np.asarray(item["mueller"][key], dtype=float))
    arr = np.stack(stack, axis=0)
    if arr.shape[0] == 1:
        std = np.zeros_like(arr[0])
    else:
        std = np.std(arr, axis=0, ddof=1)
    return theta, np.mean(arr, axis=0), std


def plot_polarization_angle_curves(results: List[Dict[str, Any]], out_dir: Path) -> None:
    groups = group_raw_by_key(results)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axs[0], "F12_over_F11", "Linear polarization $-P_{12}/P_{11}$", -1.0),
        (axs[1], "F34_over_F11", "Circular conversion $P_{34}/P_{11}$", 1.0),
    ]
    for ax, key, ylabel, sign in panels:
        for family in FAMILIES:
            group = groups.get((family, 16, 550.0))
            if not group:
                continue
            theta, mean_curve, std_curve = mueller_curve_mean_std(group, key)
            mean_curve = sign * mean_curve
            std_curve = np.abs(sign) * std_curve
            ax.plot(
                theta,
                mean_curve,
                color=COLORS[family],
                linestyle="-",
                linewidth=1.8,
                label=family.upper(),
            )
            ax.fill_between(
                theta,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=COLORS[family],
                alpha=0.15,
            )
        ax.set_xlabel("Scattering angle $\\theta$ (deg)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    save_png_and_pdf(fig, out_dir, "fig_polarization_angle_characteristics")
    plt.close(fig)


def write_report(
    path: Path,
    results: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    seed_spec: str,
) -> None:
    available = defaultdict(set)
    for item in results:
        ip = item["input_parameters"]
        available[(ip["family"], ip["n_monomers"], ip["wavelength_nm"])].add(ip["seed"])
    lines = [
        "# Transport and Polarization Characteristics of Single Scattering",
        "",
        "This post-processing experiment derives transport and polarization metrics from existing RCR JSON outputs.",
        "",
        "## Definitions",
        "",
        "- Transport scattering efficiency: `Qtr = Qsca * (1 - g)`.",
        "- Linear polarization: `PL(theta) = -P12(theta)/P11(theta)`.",
        "- Circular conversion: `PC(theta) = P34(theta)/P11(theta)`.",
        "- `g_from_P11` is computed by the angular moment of `P11` and used as a consistency check.",
        "",
        "## Inputs",
        "",
        f"- Seed filter: `{seed_spec}`",
        f"- Result files used: `{len(results)}`",
        f"- Per-seed metric rows: `{len(rows)}`",
        f"- Ensemble summary rows: `{len(summaries)}`",
        "",
        "## Output Files",
        "",
        "- `transport_polarization_by_seed.csv`",
        "- `transport_polarization_ensemble.csv`",
        "- `fig_transport_polarization_scaling.png/.pdf`",
        "- `fig_polarization_angle_characteristics.png/.pdf`",
        "",
        "## Available Case Coverage",
        "",
    ]
    for key in sorted(available.keys(), key=lambda item: (item[0], item[1], item[2])):
        seeds = ",".join(str(seed) for seed in sorted(available[key]))
        lines.append(f"- `{key[0]}`, `N={key[1]}`, `{key[2]:.0f} nm`: seeds `{seeds}`")
    lines.append("")
    if summaries:
        max_err = max(abs(float(item["abs_g_consistency_error_mean"])) for item in summaries)
        lines.append(f"Maximum mean `|g_from_P11 - g|`: `{max_err:.3e}`")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def warn_missing_coverage(results: List[Dict[str, Any]], seed_spec: str) -> None:
    requested = parse_seed_spec(seed_spec)
    available = defaultdict(set)
    for item in results:
        ip = item["input_parameters"]
        available[(ip["family"], int(ip["n_monomers"]), float(ip["wavelength_nm"]))].add(int(ip["seed"]))
    expected_scaling = [(family, n, 550.0) for family in FAMILIES for n in (8, 16, 32, 64)]
    expected_angle = [(family, 16, 550.0) for family in FAMILIES]
    missing_messages = []
    for key in expected_scaling + expected_angle:
        seeds = available.get(key, set())
        missing = sorted(requested - seeds)
        if missing:
            missing_messages.append(f"{key}: missing seeds {missing}")
    if missing_messages:
        print("WARNING: incomplete requested seed coverage:", file=sys.stderr)
        for message in missing_messages:
            print(f"  {message}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive transport and polarization metrics from RCR single-scattering JSON outputs."
    )
    parser.add_argument("--results", default="experiments/application/results")
    parser.add_argument(
        "--out",
        default="experiments/transport_and_polarization_characteristics_of_single_scattering/output",
    )
    parser.add_argument("--seeds", default="0-9")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir, args.seeds)
    if not results:
        raise SystemExit(
            f"No result JSON files in {results_dir} matched seed filter {args.seeds!r}."
        )
    warn_missing_coverage(results, args.seeds)

    rows = [derive_metrics(item) for item in results]
    summaries = summarize_rows(rows)

    write_csv(out_dir / "transport_polarization_by_seed.csv", rows)
    write_csv(out_dir / "transport_polarization_ensemble.csv", summaries)
    plot_transport_scaling(summaries, out_dir)
    plot_polarization_angle_curves(results, out_dir)
    write_report(
        out_dir / "transport_polarization_report.md",
        results,
        rows,
        summaries,
        args.seeds,
    )
    print("Transport and polarization analysis completed.")


if __name__ == "__main__":
    main()
