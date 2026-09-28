#!/usr/bin/env python3
"""CP282A-001 geometry case provider and aggregate cache writer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_GEOMETRY_PATH = ROOT / "experiments" / "exp19" / "code" / "cp13e_soot_geometry.py"
CACHE_ROOT = ROOT / "experiments" / "exp25" / "cache" / "aggregates"
RESULT_DIR = ROOT / "experiments" / "exp25" / "results"
JSON_PATH = RESULT_DIR / "cp282a_001_geometry_case_provider.json"
MD_PATH = RESULT_DIR / "cp282a_001_geometry_case_provider.md"

STORY_ID = "CP282A-001"
PROVIDER_VERSION = "cp282a.geometry_provider.v1"
CASE_SCHEMA_VERSION = "cp282a.geometry_case.v1"
CASE_KEY_SCHEMA_VERSION = "cp282a.geometry_case_key.v1"
DEFAULT_AGGREGATE_KIND = "bcca"
DEFAULT_N = 64
DEFAULT_X_G_TARGET = 2.20
DEFAULT_SEED = 0
DEFAULT_MONOMER_RADIUS_NM = 25.0
DEFAULT_WAVELENGTH_NM = 550.0
DEFAULT_M_COMPLEX = 1.95 + 0.79j
DEFAULT_FRACTAL_D_F = 1.78
DEFAULT_FRACTAL_K_F = 1.3
DEFAULT_MEDIUM_REFRACTIVE_INDEX = 1.0
DEFAULT_OVERLAP = 0.0
DEFAULT_X_G_TOLERANCE = 0.05
DEFAULT_MAX_ATTEMPTS = 100000
DEFAULT_MAX_SEED_ATTEMPTS = 64


# =============================================================================
# Geometry generation copied from experiments/exp19/code/cp13e_soot_geometry.py.
# Do not import exp19 here; CP282A requires an exp25-local copy.
# =============================================================================

def random_sphere(rng=None):
    """Return a random point on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    x = np.cos(theta) * np.sqrt(1.0 - u * u)
    y = np.sin(theta) * np.sqrt(1.0 - u * u)
    z = u
    return np.array([x, y, z])


def ray_sphere_intersect(ray_origin, ray_dir, sphere_centers, sphere_radius):
    """Find the closest intersection of a ray with equal-radius spheres."""
    oc = ray_origin - sphere_centers
    b = 2.0 * (oc * ray_dir).sum(axis=1)
    c = (oc * oc).sum(axis=1) - sphere_radius**2
    disc = b * b - 4.0 * c

    hits = disc >= 0.0
    if not hits.any():
        return None, None

    t = np.full(len(sphere_centers), np.inf)
    t[hits] = (-b[hits] - np.sqrt(disc[hits])) / 2.0
    t[t < 1e-9] = np.inf

    if np.isinf(t).all():
        return None, None

    idx = int(np.argmin(t))
    hit_point = ray_origin + t[idx] * ray_dir
    return idx, hit_point


def check_overlap(new_pos, positions, radius, tol=1e-12):
    """Return True if new_pos does not overlap existing equal-radius spheres."""
    if len(positions) == 0:
        return True
    dists = np.linalg.norm(positions - new_pos, axis=1)
    return bool(np.all(dists >= 2.0 * radius - tol))


def validate_non_overlapping_spheres(positions, radii, tol=1e-12):
    """Raise ValueError if any two spheres overlap."""
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if radii.ndim != 1 or radii.shape[0] != positions.shape[0]:
        raise ValueError("radii must have shape (n,)")
    if len(radii) < 2:
        return

    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    required = radii[:, None] + radii[None, :] - tol
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(required, -np.inf)
    if np.any(dists < required):
        i, j = np.argwhere(dists < required)[0]
        raise ValueError(
            f"overlap detected between monomers {int(i)} and {int(j)}: "
            f"distance={dists[i, j]:.6g}, required>={required[i, j]:.6g}"
        )


def _validate_common_inputs(n_monomers, radius, overlap):
    if n_monomers < 1:
        raise ValueError("n_monomers must be at least 1")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")


def _merge_clusters(pos1, pos2, rad1, rad2, radius, rng, max_attempts):
    for _attempt in range(max_attempts):
        direction = random_sphere(rng)
        best_t = np.inf
        best_hit = None
        best_src = None
        best_tgt = None

        for j, p2 in enumerate(pos2):
            idx, hit = ray_sphere_intersect(p2, direction, pos1, radius)
            if hit is None or idx is None:
                continue
            t = np.linalg.norm(hit - p2)
            if t < best_t:
                best_t = t
                best_hit = hit
                best_src = j
                best_tgt = idx

        if best_hit is None or best_src is None or best_tgt is None:
            continue

        contact_vec = best_hit + (best_hit - pos1[best_tgt]) - pos2[best_src]
        candidate_pos2 = pos2 + contact_vec

        merged_pos = np.vstack([pos1, candidate_pos2])
        merged_rad = np.concatenate([rad1, rad2])
        try:
            validate_non_overlapping_spheres(merged_pos, merged_rad)
        except ValueError:
            continue
        return merged_pos, merged_rad

    raise RuntimeError("BCCA cluster merge failed before max_attempts")


def build_bpca(n_monomers=8, radius=0.5, overlap=0.0, seed=None, max_attempts=100000):
    """Build a Ballistic Particle-Cluster Aggregation geometry."""
    _validate_common_inputs(n_monomers, radius, overlap)
    rng = np.random.default_rng(seed)

    positions = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([radius])

    for _ in range(n_monomers - 1):
        placed = False
        for _attempt in range(max_attempts):
            farthest = (
                float(np.max(np.linalg.norm(positions, axis=1)))
                if len(positions) > 1
                else radius
            )
            scale = max(farthest * 2.0, radius * 4.0)

            origin = random_sphere(rng) * scale
            target = random_sphere(rng) * scale
            direction = target - origin
            direction /= np.linalg.norm(direction)

            idx, hit = ray_sphere_intersect(origin, direction, positions, radius)
            if hit is None or idx is None:
                continue

            new_pos = hit + (hit - positions[idx])

            if not check_overlap(new_pos, positions, radius):
                continue

            if overlap > 0.0:
                new_pos = hit + (hit - positions[idx]) * (1.0 - overlap)

            positions = np.vstack([positions, new_pos])
            radii = np.append(radii, radius)
            placed = True
            break
        if not placed:
            raise RuntimeError("BPCA placement failed before max_attempts")

    return positions, radii


def build_bcca(n_monomers=8, radius=0.5, overlap=0.0, seed=None, max_attempts=100000):
    """Build a Ballistic Cluster-Cluster Aggregation geometry."""
    _validate_common_inputs(n_monomers, radius, overlap)
    if n_monomers < 2 or (n_monomers & (n_monomers - 1)) != 0:
        raise ValueError("n_monomers must be a power of 2 for BCCA")

    rng = np.random.default_rng(seed)
    for _seed_attempt in range(64):
        child_seeds = rng.integers(0, 2**31, size=n_monomers // 2)
        clusters = [
            build_bpca(2, radius, overlap, seed=int(child_seed), max_attempts=max_attempts)
            for child_seed in child_seeds
        ]

        try:
            while len(clusters) > 1:
                next_clusters = []
                for i in range(0, len(clusters), 2):
                    pos1, rad1 = clusters[i]
                    pos2, rad2 = clusters[i + 1]

                    pos1 = pos1 - pos1.mean(axis=0)
                    pos2 = pos2 - pos2.mean(axis=0)

                    farthest = max(
                        float(np.max(np.linalg.norm(pos1, axis=1))),
                        float(np.max(np.linalg.norm(pos2, axis=1))),
                        radius * 4.0,
                    )
                    pos2 = pos2 + random_sphere(rng) * farthest * 10.0

                    merged = _merge_clusters(pos1, pos2, rad1, rad2, radius, rng, max_attempts)
                    next_clusters.append(merged)

                clusters = next_clusters

            return clusters[0]
        except RuntimeError:
            continue

    raise RuntimeError("BCCA cluster merge failed before max_attempts")


def gyration_radius(positions):
    """Compute radius of gyration Rg = sqrt(mean |r_i - r_cm|^2)."""
    r_cm = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - r_cm) ** 2, axis=1))))


def pairwise_distances(positions):
    """Return all N*(N-1)/2 unique pairwise distances."""
    n = len(positions)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(positions[i] - positions[j])))
    return np.array(dists)


# =============================================================================
# CP282A provider, deterministic cache writer, and verification helpers.
# =============================================================================

class GeometryProviderError(RuntimeError):
    """Error carrying a CP282A blocker type."""

    def __init__(self, blocker_type: str, message: str):
        super().__init__(message)
        self.blocker_type = blocker_type


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def decimal_string(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite decimal value: {value!r}")
    return f"{float(value):.12f}"


def decimal_token(value: float) -> str:
    return decimal_string(value).replace(".", "p").replace("-", "m")


def is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


def source_geometry_sha256() -> str:
    return sha256_file(SOURCE_GEOMETRY_PATH)


def ordered_case_key_fields(
    *,
    N: int,
    x_g_target: float,
    seed: int,
    aggregate_kind: str,
    monomer_radius_nm: float,
    wavelength_nm: float,
    m_complex: complex,
    fractal_D_f: float,
    fractal_k_f: float,
    medium_refractive_index: float,
    overlap: float,
    x_g_tolerance: float,
) -> list[list[Any]]:
    return [
        ["schema_version", CASE_KEY_SCHEMA_VERSION],
        ["provider_version", PROVIDER_VERSION],
        ["aggregate_kind", aggregate_kind],
        ["N", int(N)],
        ["x_g_target_decimal", decimal_string(x_g_target)],
        ["seed", int(seed)],
        ["monomer_radius_nm_decimal", decimal_string(monomer_radius_nm)],
        ["wavelength_nm_decimal", decimal_string(wavelength_nm)],
        ["m_real_decimal", decimal_string(float(m_complex.real))],
        ["m_imag_decimal", decimal_string(float(m_complex.imag))],
        ["fractal_D_f_decimal", decimal_string(fractal_D_f)],
        ["fractal_k_f_decimal", decimal_string(fractal_k_f)],
        ["medium_refractive_index_decimal", decimal_string(medium_refractive_index)],
        ["overlap_decimal", decimal_string(overlap)],
        ["x_g_tolerance_decimal", decimal_string(x_g_tolerance)],
        ["source_geometry_sha256", source_geometry_sha256()],
    ]


def case_key_sha256(fields: list[list[Any]]) -> str:
    return sha256_bytes(json.dumps(fields, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def validate_provider_inputs(
    *,
    N: int,
    x_g_target: float,
    aggregate_kind: str,
    monomer_radius_nm: float,
    wavelength_nm: float,
    medium_refractive_index: float,
    fractal_D_f: float,
    fractal_k_f: float,
    overlap: float,
) -> None:
    if int(N) != N or int(N) < 2:
        raise GeometryProviderError("invalid_N", "N must be an integer >= 2")
    if not math.isfinite(float(x_g_target)) or float(x_g_target) <= 0.0:
        raise GeometryProviderError("invalid_x_g_target", "x_g_target must be positive")
    if aggregate_kind not in {"bcca", "bpca"}:
        raise GeometryProviderError("invalid_aggregate_kind", "aggregate_kind must be bcca or bpca")
    if float(monomer_radius_nm) <= 0.0 or not math.isfinite(float(monomer_radius_nm)):
        raise GeometryProviderError("invalid_monomer_radius", "monomer_radius_nm must be positive")
    if float(wavelength_nm) <= 0.0 or not math.isfinite(float(wavelength_nm)):
        raise GeometryProviderError("invalid_wavelength", "wavelength_nm must be positive")
    if float(medium_refractive_index) <= 0.0 or not math.isfinite(float(medium_refractive_index)):
        raise GeometryProviderError("invalid_medium_refractive_index", "medium_refractive_index must be positive")
    if float(fractal_D_f) <= 0.0 or not math.isfinite(float(fractal_D_f)):
        raise GeometryProviderError("invalid_fractal_D_f", "fractal_D_f must be positive")
    if float(fractal_k_f) <= 0.0 or not math.isfinite(float(fractal_k_f)):
        raise GeometryProviderError("invalid_fractal_k_f", "fractal_k_f must be positive")
    if float(overlap) < 0.0 or float(overlap) >= 1.0 or not math.isfinite(float(overlap)):
        raise GeometryProviderError("invalid_overlap", "overlap must satisfy 0 <= overlap < 1")
    if aggregate_kind == "bcca" and not is_power_of_two(int(N)):
        raise GeometryProviderError(
            "invalid_bcca_N",
            "copied CP-13E build_bcca supports only power-of-two N via balanced pairwise merges",
        )


def build_candidate(
    *,
    N: int,
    aggregate_kind: str,
    seed: int,
    monomer_radius_nm: float,
    overlap: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    if aggregate_kind == "bcca":
        return build_bcca(
            n_monomers=int(N),
            radius=float(monomer_radius_nm),
            overlap=float(overlap),
            seed=int(seed),
            max_attempts=int(max_attempts),
        )
    return build_bpca(
        n_monomers=int(N),
        radius=float(monomer_radius_nm),
        overlap=float(overlap),
        seed=int(seed),
        max_attempts=int(max_attempts),
    )


def case_for_target(
    N: int,
    x_g_target: float,
    seed: int,
    *,
    aggregate_kind: str = DEFAULT_AGGREGATE_KIND,
    monomer_radius_nm: float = DEFAULT_MONOMER_RADIUS_NM,
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    m_complex: complex = DEFAULT_M_COMPLEX,
    fractal_D_f: float = DEFAULT_FRACTAL_D_F,
    fractal_k_f: float = DEFAULT_FRACTAL_K_F,
    medium_refractive_index: float = DEFAULT_MEDIUM_REFRACTIVE_INDEX,
    overlap: float = DEFAULT_OVERLAP,
    x_g_tolerance: float = DEFAULT_X_G_TOLERANCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_seed_attempts: int = DEFAULT_MAX_SEED_ATTEMPTS,
) -> dict[str, Any]:
    """Create a deterministic aggregate case matching x_g_target within tolerance."""
    aggregate_kind = str(aggregate_kind).lower()
    N = int(N)
    seed = int(seed)
    m_complex = complex(m_complex)
    validate_provider_inputs(
        N=N,
        x_g_target=x_g_target,
        aggregate_kind=aggregate_kind,
        monomer_radius_nm=monomer_radius_nm,
        wavelength_nm=wavelength_nm,
        medium_refractive_index=medium_refractive_index,
        fractal_D_f=fractal_D_f,
        fractal_k_f=fractal_k_f,
        overlap=overlap,
    )
    if int(max_attempts) < 1:
        raise GeometryProviderError("invalid_max_attempts", "max_attempts must be >= 1")
    if int(max_seed_attempts) < 1:
        raise GeometryProviderError("invalid_max_seed_attempts", "max_seed_attempts must be >= 1")

    fields = ordered_case_key_fields(
        N=N,
        x_g_target=float(x_g_target),
        seed=seed,
        aggregate_kind=aggregate_kind,
        monomer_radius_nm=float(monomer_radius_nm),
        wavelength_nm=float(wavelength_nm),
        m_complex=m_complex,
        fractal_D_f=float(fractal_D_f),
        fractal_k_f=float(fractal_k_f),
        medium_refractive_index=float(medium_refractive_index),
        overlap=float(overlap),
        x_g_tolerance=float(x_g_tolerance),
    )
    case_key = case_key_sha256(fields)
    attempts: list[dict[str, Any]] = []

    for offset in range(int(max_seed_attempts)):
        accepted_seed = seed + offset
        try:
            positions_nm, radii_nm = build_candidate(
                N=N,
                aggregate_kind=aggregate_kind,
                seed=accepted_seed,
                monomer_radius_nm=float(monomer_radius_nm),
                overlap=float(overlap),
                max_attempts=int(max_attempts),
            )
        except RuntimeError as exc:
            attempts.append({"seed": accepted_seed, "status": "generation_failed", "error": str(exc)})
            continue

        positions_nm = np.ascontiguousarray(positions_nm, dtype=np.float64)
        radii_nm = np.ascontiguousarray(radii_nm, dtype=np.float64)
        try:
            validate_non_overlapping_spheres(positions_nm, radii_nm, tol=1.0e-9)
        except ValueError as exc:
            attempts.append({"seed": accepted_seed, "status": "overlap_failed", "error": str(exc)})
            continue

        R_g_nm = gyration_radius(positions_nm)
        x_g_actual = 2.0 * math.pi * float(medium_refractive_index) * R_g_nm / float(wavelength_nm)
        x_g_abs_error = abs(x_g_actual - float(x_g_target))
        dists = pairwise_distances(positions_nm)
        min_pair = float(dists.min()) if len(dists) else math.inf
        min_gap = min_pair - 2.0 * float(monomer_radius_nm)
        attempts.append(
            {
                "seed": accepted_seed,
                "status": "candidate",
                "R_g_nm": R_g_nm,
                "x_g_actual": x_g_actual,
                "x_g_abs_error": x_g_abs_error,
                "min_gap_nm": min_gap,
            }
        )
        if x_g_abs_error < float(x_g_tolerance):
            return {
                "schema_version": CASE_SCHEMA_VERSION,
                "case_key": case_key,
                "ordered_case_key_fields": fields,
                "aggregate_kind": aggregate_kind,
                "positions_nm": positions_nm,
                "radii_nm": radii_nm,
                "monomer_radius_nm": float(monomer_radius_nm),
                "wavelength_nm": float(wavelength_nm),
                "m_complex": m_complex,
                "m_real": float(m_complex.real),
                "m_imag": float(m_complex.imag),
                "medium_refractive_index": float(medium_refractive_index),
                "R_g_nm": R_g_nm,
                "x_g_target": float(x_g_target),
                "x_g_actual": x_g_actual,
                "x_g_abs_error": x_g_abs_error,
                "x_p": 2.0 * math.pi * float(medium_refractive_index) * float(monomer_radius_nm) / float(wavelength_nm),
                "N": N,
                "seed": seed,
                "accepted_seed": accepted_seed,
                "fractal_D_f": float(fractal_D_f),
                "fractal_k_f": float(fractal_k_f),
                "overlap": float(overlap),
                "min_pair_distance_nm": min_pair,
                "min_gap_nm": min_gap,
                "non_overlapping": bool(min_gap >= -1.0e-9),
                "x_g_tolerance": float(x_g_tolerance),
                "source_geometry_file": "experiments/exp19/code/cp13e_soot_geometry.py",
                "source_geometry_sha256": source_geometry_sha256(),
                "provider_version": PROVIDER_VERSION,
                "seed_attempts": attempts,
            }

    raise GeometryProviderError(
        "x_g_target_not_reached",
        f"no {aggregate_kind} case reached x_g_target={x_g_target} within {x_g_tolerance} "
        f"after {max_seed_attempts} deterministic seed attempts; attempts={attempts}",
    )


def metadata_for_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"positions_nm", "radii_nm", "m_complex"}
    } | {
        "m_complex": {"real": case["m_real"], "imag": case["m_imag"]},
    }


def cache_filename(N: int, x_g_target: float, seed: int) -> str:
    return f"case__N{int(N)}__xg{decimal_token(float(x_g_target))}__seed{int(seed)}.npz"


def npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(handle, np.asarray(array), allow_pickle=False)
    return handle.getvalue()


def deterministic_npz_bytes(arrays: list[tuple[str, np.ndarray]]) -> bytes:
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays:
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(array))
    return handle.getvalue()


def cache_arrays_for_case(case: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    metadata_json = canonical_json(metadata_for_case(case))
    return [
        ("positions_nm", np.ascontiguousarray(case["positions_nm"], dtype=np.float64)),
        ("radii_nm", np.ascontiguousarray(case["radii_nm"], dtype=np.float64)),
        ("metadata_json", np.asarray(metadata_json)),
    ]


def write_bytes_atomic(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / "staging" / f".tmp__{path.stem}__{os.getpid()}__{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    tmp_path = staging / path.name
    tmp_path.write_bytes(payload)
    fsync_file(tmp_path)
    os.replace(tmp_path, path)
    try:
        staging.rmdir()
        staging.parent.rmdir()
    except OSError:
        pass
    return path


def load_cached_case(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        positions = np.ascontiguousarray(source["positions_nm"], dtype=np.float64)
        radii = np.ascontiguousarray(source["radii_nm"], dtype=np.float64)
        metadata_raw = source["metadata_json"]
        metadata_text = str(metadata_raw.item() if metadata_raw.shape == () else metadata_raw)
    metadata = json.loads(metadata_text)
    metadata["positions_nm"] = positions
    metadata["radii_nm"] = radii
    metadata["m_complex"] = complex(metadata["m_complex"]["real"], metadata["m_complex"]["imag"])
    return metadata


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(case["positions_nm"], dtype=np.float64)
    radii = np.asarray(case["radii_nm"], dtype=np.float64)
    validate_non_overlapping_spheres(positions, radii, tol=1.0e-9)
    recomputed_R_g = gyration_radius(positions)
    recomputed_x_g = (
        2.0
        * math.pi
        * float(case["medium_refractive_index"])
        * recomputed_R_g
        / float(case["wavelength_nm"])
    )
    dists = pairwise_distances(positions)
    min_pair = float(dists.min()) if len(dists) else math.inf
    min_gap = min_pair - 2.0 * float(case["monomer_radius_nm"])
    checks = {
        "gyration_radius_matches": abs(recomputed_R_g - float(case["R_g_nm"])) < 1.0e-9,
        "x_g_recompute_matches": abs(recomputed_x_g - float(case["x_g_actual"])) < 1.0e-9,
        "overlap_check_passed": bool(np.all(dists >= 2.0 * float(case["monomer_radius_nm"]) - 1.0e-9)),
        "target_tolerance_passed": abs(float(case["x_g_actual"]) - float(case["x_g_target"])) < float(case["x_g_tolerance"]),
        "recomputed_R_g_nm": recomputed_R_g,
        "recomputed_x_g": recomputed_x_g,
        "recomputed_min_pair_distance_nm": min_pair,
        "recomputed_min_gap_nm": min_gap,
    }
    checks["all_passed"] = all(
        bool(checks[key])
        for key in [
            "gyration_radius_matches",
            "x_g_recompute_matches",
            "overlap_check_passed",
            "target_tolerance_passed",
        ]
    )
    return checks


def write_readme() -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    readme = CACHE_ROOT / "README.md"
    payload = """# CP282A Aggregate Geometry Cache

This directory stores deterministic aggregate geometry cases produced by
`experiments/exp25/code/cp282a_geometry_provider.py`.

Admission rule:
- file names use the input tuple `(N, x_g_target, seed)`;
- `x_g_actual` is measured output metadata;
- `.npz` payloads are written through staging and atomic rename;
- BCCA currently accepts only power-of-two `N` because the copied CP-13E
  implementation uses balanced pairwise cluster merges.

CP282A does not read or write the CP282 compressed-M2L SVD cache.
"""
    readme.write_text(payload, encoding="utf-8")
    return readme


def write_case_cache(case: dict[str, Any], cache_root: Path = CACHE_ROOT) -> dict[str, Any]:
    path = cache_root / cache_filename(case["N"], case["x_g_target"], case["seed"])
    payload = deterministic_npz_bytes(cache_arrays_for_case(case))
    write_bytes_atomic(path, payload)
    cached = load_cached_case(path)
    validation = validate_case(cached)
    if not validation["all_passed"]:
        raise GeometryProviderError("aggregate_cache_reload_validation_failed", canonical_json(validation))
    payload_sha = sha256_file(path)
    deterministic_sha = sha256_bytes(deterministic_npz_bytes(cache_arrays_for_case(case)))
    return {
        "cache_path": str(path.relative_to(ROOT)),
        "cache_filename": path.name,
        "payload_file_sha256": payload_sha,
        "deterministic_rewrite_sha256": deterministic_sha,
        "byte_identical_rewrite": payload_sha == deterministic_sha,
        "payload_bytes": path.stat().st_size,
        "reload_validation": validation,
    }


def run_default_story() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_readme()
    case = case_for_target(DEFAULT_N, DEFAULT_X_G_TARGET, DEFAULT_SEED)
    cache_info = write_case_cache(case)
    result = {
        "schema_version": "cp282a.geometry_case_provider.result.v1",
        "story_id": STORY_ID,
        "run_timestamp_utc": utc_now(),
        "git_commit_at_run": git_commit(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "scope": {
            "implementation": "geometry_provider_only",
            "svd_cache_touched": False,
            "native_code_touched": False,
            "production_runner_touched": False,
            "dr_x_run": False,
            "physical_parameter_lock": False,
        },
        "source": {
            "copied_from": "experiments/exp19/code/cp13e_soot_geometry.py",
            "source_geometry_sha256": source_geometry_sha256(),
            "copy_not_import": True,
        },
        "case": metadata_for_case(case),
        "cache": cache_info,
        "verification": {
            "cache_reproducibility_byte_identical": cache_info["byte_identical_rewrite"],
            "gyration_radius_consistency": cache_info["reload_validation"]["gyration_radius_matches"],
            "x_g_recompute_consistency": cache_info["reload_validation"]["x_g_recompute_matches"],
            "overlap_check": cache_info["reload_validation"]["overlap_check_passed"],
            "x_g_target_tolerance": cache_info["reload_validation"]["target_tolerance_passed"],
            "all_passed": bool(
                cache_info["byte_identical_rewrite"] and cache_info["reload_validation"]["all_passed"]
            ),
        },
        "blockers": [],
    }
    JSON_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_result(result)
    return result


def verify_existing() -> dict[str, Any]:
    path = CACHE_ROOT / cache_filename(DEFAULT_N, DEFAULT_X_G_TARGET, DEFAULT_SEED)
    if not path.exists():
        raise FileNotFoundError(path)
    case = load_cached_case(path)
    validation = validate_case(case)
    deterministic_sha = sha256_bytes(deterministic_npz_bytes(cache_arrays_for_case(case)))
    file_sha = sha256_file(path)
    result = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_sha,
        "deterministic_rewrite_sha256": deterministic_sha,
        "byte_identical_rewrite": file_sha == deterministic_sha,
        "reload_validation": validation,
        "all_passed": bool(file_sha == deterministic_sha and validation["all_passed"]),
    }
    if not result["all_passed"]:
        raise RuntimeError(canonical_json(result))
    return result


def write_markdown_result(result: dict[str, Any]) -> None:
    case = result["case"]
    cache = result["cache"]
    verification = result["verification"]
    lines = [
        "# CP282A-001 Geometry Case Provider",
        "",
        f"**Story**: `{STORY_ID}`",
        f"**Run timestamp UTC**: `{result['run_timestamp_utc']}`",
        f"**Overall**: `{'PASS' if verification['all_passed'] else 'FAIL'}`",
        "",
        "## Scope",
        "",
        "- Geometry provider only.",
        "- Copied CP-13E geometry logic into exp25; no exp19 import.",
        "- No CP282 SVD cache, native code, production runner, manuscript figures, or DR-X run.",
        "- Physical parameter lock remains deferred to CP282B.",
        "",
        "## Generated Case",
        "",
        f"- aggregate_kind: `{case['aggregate_kind']}`",
        f"- N: `{case['N']}`",
        f"- requested seed: `{case['seed']}`",
        f"- accepted_seed: `{case['accepted_seed']}`",
        f"- x_g_target: `{case['x_g_target']:.12f}`",
        f"- x_g_actual: `{case['x_g_actual']:.12f}`",
        f"- x_g_abs_error: `{case['x_g_abs_error']:.12e}`",
        f"- R_g_nm: `{case['R_g_nm']:.12f}`",
        f"- x_p: `{case['x_p']:.12f}`",
        f"- monomer_radius_nm: `{case['monomer_radius_nm']:.12f}`",
        f"- wavelength_nm: `{case['wavelength_nm']:.12f}`",
        f"- medium_refractive_index: `{case['medium_refractive_index']:.12f}`",
        f"- m_complex: `{case['m_real']}+{case['m_imag']}i`",
        f"- fractal_D_f: `{case['fractal_D_f']:.12f}`",
        f"- fractal_k_f: `{case['fractal_k_f']:.12f}`",
        "",
        "## Cache",
        "",
        f"- path: `{cache['cache_path']}`",
        f"- payload_file_sha256: `{cache['payload_file_sha256']}`",
        f"- payload_bytes: `{cache['payload_bytes']}`",
        f"- byte_identical_rewrite: `{cache['byte_identical_rewrite']}`",
        "",
        "## Verification",
        "",
        f"- cache reproducibility byte-identical: `{verification['cache_reproducibility_byte_identical']}`",
        f"- gyration radius consistency: `{verification['gyration_radius_consistency']}`",
        f"- x_g recompute consistency: `{verification['x_g_recompute_consistency']}`",
        f"- overlap check: `{verification['overlap_check']}`",
        f"- x_g target tolerance: `{verification['x_g_target_tolerance']}`",
        "",
        "## Seed Attempts",
        "",
    ]
    for attempt in case["seed_attempts"]:
        if attempt["status"] == "candidate":
            lines.append(
                f"- seed `{attempt['seed']}`: x_g=`{attempt['x_g_actual']:.12f}`, "
                f"abs_error=`{attempt['x_g_abs_error']:.12e}`, min_gap_nm=`{attempt['min_gap_nm']:.12e}`"
            )
        else:
            lines.append(f"- seed `{attempt['seed']}`: `{attempt['status']}` ({attempt.get('error', '')})")
    lines.append("")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-story", action="store_true", help="Generate the CP282A default example cache and results")
    parser.add_argument("--verify-existing", action="store_true", help="Verify the admitted default example cache")
    args = parser.parse_args(argv)

    if args.verify_existing:
        print(json.dumps(verify_existing(), indent=2, sort_keys=True))
        return 0
    result = run_default_story()
    print(json.dumps({"overall": result["verification"]["all_passed"], "cache": result["cache"]}, indent=2))
    return 0 if result["verification"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
