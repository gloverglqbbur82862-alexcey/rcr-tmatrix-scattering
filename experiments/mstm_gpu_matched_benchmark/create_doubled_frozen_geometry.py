#!/usr/bin/env python3
"""Create a deterministic doubled stress geometry from a frozen MSTM input.

This helper is for a performance stress test when direct BCCA generation at
larger N is not robust for the seed-42 realization. The generated geometry is
a two-copy binary merge of a frozen aggregate, not an independent BCCA seed-42
realization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_remote_gpu_mstm_benchmark import R_DIMLESS, read_positions_from_mstm_input, write_mstm_input


def doubled_positions(source_positions: np.ndarray, gap_factor: float) -> np.ndarray:
    """Return two non-overlapping copies separated along x and recentered."""
    positions = np.asarray(source_positions, dtype=np.float64)
    min_x = float(np.min(positions[:, 0]))
    max_x = float(np.max(positions[:, 0]))
    shift = (max_x - min_x) + 2.0 * R_DIMLESS * float(gap_factor)
    second = positions.copy()
    second[:, 0] += shift
    merged = np.vstack([positions, second])
    merged -= np.mean(merged, axis=0, keepdims=True)
    return merged


def min_pair_distance(positions: np.ndarray) -> float:
    best = float("inf")
    for idx in range(len(positions)):
        deltas = positions[idx + 1 :] - positions[idx]
        if len(deltas) == 0:
            continue
        distances = np.linalg.norm(deltas, axis=1)
        best = min(best, float(np.min(distances)))
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-input",
        type=Path,
        default=Path("experiments/mstm_validation/inputs/mstm_bcca_n128_w550_seed42.inp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/mstm_gpu_matched_benchmark/frozen_inputs"),
    )
    parser.add_argument("--output-name", default="mstm_bcca_n256_w550_seed42.inp")
    parser.add_argument("--gap-factor", type=float, default=1.05)
    args = parser.parse_args()

    source = args.source_input
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    source_positions = read_positions_from_mstm_input(source)
    merged = doubled_positions(source_positions, gap_factor=args.gap_factor)
    if len(merged) != 2 * len(source_positions):
        raise RuntimeError("Unexpected merged geometry size")

    min_dist = min_pair_distance(merged)
    min_gap = min_dist - 2.0 * R_DIMLESS
    if min_gap < -1.0e-9:
        raise RuntimeError(f"Generated geometry overlaps: min_gap={min_gap}")

    write_mstm_input(
        out_path,
        f"outputs/generated_n{len(merged)}_geometry_placeholder.out",
        merged,
    )

    provenance = {
        "source_type": "deterministic_doubled_frozen_geometry",
        "description": (
            f"Two translated copies of the frozen N={len(source_positions)} input geometry; "
            f"used only as a frozen N={len(merged)} performance stress-test geometry, "
            "not as an independently generated BCCA seed-42 realization."
        ),
        "source_input": str(source),
        "output_input": str(out_path),
        "source_n": int(len(source_positions)),
        "output_n": int(len(merged)),
        "radius_dimless": float(R_DIMLESS),
        "gap_factor": float(args.gap_factor),
        "min_pair_distance_dimless": min_dist,
        "min_surface_gap_dimless": min_gap,
    }
    out_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
