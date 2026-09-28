from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CASE_KEY_FIELDS = (
    "family",
    "n_monomers",
    "radius_nm",
    "wavelength_nm",
    "m_real",
    "m_imag",
    "monomer_order",
    "aggregate_order",
    "tier",
)


def parse_seed_spec(seed_spec: str | None) -> list[int]:
    if not seed_spec:
        return []
    seeds: set[int] = set()
    for token in seed_spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid seed range '{token}'")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(token))
    return sorted(seeds)


def with_seed_suffix(case_id: str, seed: int) -> str:
    return re.sub(r"_seed\d+$", f"_seed{seed}", case_id) if re.search(r"_seed\d+$", case_id) else f"{case_id}_seed{seed}"


def case_key(result: dict[str, Any]) -> tuple[Any, ...]:
    ip = result["input_parameters"]
    return tuple(ip[field] for field in CASE_KEY_FIELDS)


def load_results(results_dir: Path | str) -> list[dict[str, Any]]:
    results_path = Path(results_dir)
    items: list[dict[str, Any]] = []
    for path in sorted(results_path.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            items.append(json.load(handle))
    return items


def group_results(results: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[case_key(result)].append(result)
    for group in grouped.values():
        group.sort(key=lambda item: item["input_parameters"]["seed"])
    return grouped


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if values.shape[0] > 1:
        std = values.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


def scalar_observable_summary(group: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {key: np.array([item["observables"][key] for item in group], dtype=float) for key in ("Qext", "Qsca", "Qabs", "SSA", "g")}
    summary: dict[str, float] = {"seed_count": float(len(group))}
    for key, values in metrics.items():
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[f"{key}_median"] = float(np.median(values))
    times = np.array([item["timing_metadata"]["time_total_s"] for item in group], dtype=float)
    summary["time_total_s_mean"] = float(times.mean())
    summary["time_total_s_std"] = float(times.std(ddof=1)) if times.size > 1 else 0.0
    summary["time_total_s_median"] = float(np.median(times))
    return summary


def aligned_mueller_stack(group: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(group[0]["theta_deg"], dtype=float)
    stack = []
    for item in group:
        theta_i = np.asarray(item["theta_deg"], dtype=float)
        if theta_i.shape != theta.shape or not np.allclose(theta_i, theta, rtol=0.0, atol=1e-12):
            raise ValueError("Inconsistent theta grids within ensemble group")
        stack.append(np.asarray(item["mueller"][key], dtype=float))
    return theta, np.stack(stack, axis=0)


def mueller_mean_std(group: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    theta = np.asarray(group[0]["theta_deg"], dtype=float)
    metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key in ("F11", "F12_over_F11", "F22_over_F11", "F33_over_F11", "F34_over_F11", "F44_over_F11"):
        theta_i, stack = aligned_mueller_stack(group, key)
        if not np.allclose(theta_i, theta, rtol=0.0, atol=1e-12):
            raise ValueError("Inconsistent theta grids within ensemble group")
        mean, std = mean_std(stack)
        metrics[key] = (theta, mean, std)
    return metrics
