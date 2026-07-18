#!/usr/bin/env python3
"""Run a single electromagnetic scattering simulation case from cases.json and save results."""

import os
import sys
import json
import time
import socket
import platform
import argparse
import numpy as np
import scipy
from scipy.special import spherical_jn, spherical_yn
import numba

# Add project root to python path to allow regular imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geometry.geometry import build_bcca, build_bpca
from solver.bicgstab_solver import MultisphereRCRSystem


def mie_coefficients_manual(m_complex: complex, x: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Mie coefficients an and bn for a sphere."""
    m = m_complex
    an_list = []
    bn_list = []
    
    def psi(nn: int, z: complex) -> complex:
        return z * spherical_jn(nn, z)
        
    def dpsi(nn: int, z: complex) -> complex:
        return spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)
        
    def xi(nn: int, z: complex) -> complex:
        return z * (spherical_jn(nn, z) + 1j * spherical_yn(nn, z))
        
    def dxi(nn: int, z: complex) -> complex:
        h = spherical_jn(nn, z) + 1j * spherical_yn(nn, z)
        dh = (spherical_jn(nn, z, derivative=True) + 1j * spherical_yn(nn, z, derivative=True))
        return h + z * dh
        
    for n in range(1, n_max + 1):
        psi_n_x = psi(n, x)
        dpsi_n_x = dpsi(n, x)
        psi_n_mx = psi(n, m * x)
        dpsi_n_mx = dpsi(n, m * x)
        xi_n_x = xi(n, x)
        dxi_n_x = dxi(n, x)
        
        a_n = (m * psi_n_mx * dpsi_n_x - psi_n_x * dpsi_n_mx) / (m * psi_n_mx * dxi_n_x - xi_n_x * dpsi_n_mx)
        b_n = (psi_n_mx * dpsi_n_x - m * psi_n_x * dpsi_n_mx) / (psi_n_mx * dxi_n_x - m * xi_n_x * dpsi_n_mx)
        
        an_list.append(complex(a_n))
        bn_list.append(complex(b_n))
        
    return np.array(an_list), np.array(bn_list)


def main():
    parser = argparse.ArgumentParser(description="Run a single scattering case from cases.json")
    parser.add_argument("--cases", required=True, help="Path to cases.json")
    parser.add_argument("--case-id", required=True, help="ID of the case to run")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    t_start = time.time()

    # 1. Load cases and find the requested one
    if not os.path.exists(args.cases):
        print(f"Error: {args.cases} not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(args.cases, "r") as f:
        cases_payload = json.load(f)

    # Keep the current compact top-level list format, while accepting a future
    # {"cases": [...]} wrapper if the experiment database grows metadata.
    cases = cases_payload.get("cases", cases_payload) if isinstance(cases_payload, dict) else cases_payload
        
    case = None
    for c in cases:
        if c["case_id"] == args.case_id:
            case = c
            break
            
    if case is None:
        print(f"Error: case-id '{args.case_id}' not found in cases.json.", file=sys.stderr)
        sys.exit(1)

    print(f"Running case: {case['case_id']}")
    print(f"Parameters: N={case['n_monomers']}, family={case['family']}, wavelength={case['wavelength_nm']}nm, m={case['m_real']}+{case['m_imag']}j")

    # 2. Build geometry
    n_monomers = case["n_monomers"]
    radius_nm = case["radius_nm"]
    seed = case["seed"]
    family = case["family"].lower()
    
    t_geom_start = time.time()
    if family == "bcca":
        pos_raw, radii_raw = build_bcca(n_monomers=n_monomers, radius=radius_nm, overlap=0.0, seed=seed)
    elif family == "bpca":
        pos_raw, radii_raw = build_bpca(n_monomers=n_monomers, radius=radius_nm, overlap=0.0, seed=seed)
    else:
        print(f"Error: Unknown family '{family}'", file=sys.stderr)
        sys.exit(1)
    t_geom_end = time.time()

    # Calculate size parameters and dimensionless wave vectors
    wavelength_nm = case["wavelength_nm"]
    k = 2.0 * np.pi / wavelength_nm
    x_param = radius_nm * k
    m_complex = case["m_real"] + 1j * case["m_imag"]
    
    pos_d = pos_raw * k
    monomer_order = case["monomer_order"]
    aggregate_order = case["aggregate_order"]

    # Calculate single sphere Mie coefficients
    a, b = mie_coefficients_manual(m_complex, x_param, monomer_order)
    an_rcr = np.zeros((2, 2, monomer_order), dtype=np.complex128)
    diag = -0.5 * (a + b)
    off = -0.5 * (a - b)
    an_rcr[0, 0, :] = diag
    an_rcr[1, 1, :] = diag
    an_rcr[0, 1, :] = off
    an_rcr[1, 0, :] = off

    # Formulate system
    sys_rcr = MultisphereRCRSystem(pos_d, [monomer_order]*n_monomers, [an_rcr]*n_monomers, k_medium=1.0)
    
    t_solve_start = time.time()
    T_agg = sys_rcr.calculate_aggregate_tmatrix(aggregate_order=aggregate_order, tol=1e-12)
    t_solve_end = time.time()

    # Calculate global observables and phase matrix
    t_phase_start = time.time()
    avgs = sys_rcr.calculate_orientation_averages(T_agg, [x_param]*n_monomers)
    
    # Generate 181 scattering angles (0 to 180 degrees)
    n_angles = case["angles_deg"]
    theta_deg = np.linspace(0.0, 180.0, n_angles)
    
    sm_coef = sys_rcr.calculate_orientation_averaged_phase_matrix_coefficients(T_agg)
    pm = sys_rcr.calculate_phase_matrix_from_coefficients(sm_coef, theta_deg, normalize=True)
    t_phase_end = time.time()

    # Extract observables
    Qext = avgs["Qext"]
    Qsca = avgs["Qsca"]
    Qabs = avgs["Qabs"]
    SSA = Qsca / Qext if Qext > 0.0 else 0.0
    g = avgs["g"]

    # Extract normalized Mueller matrix ratios
    F11 = pm[:, 0, 0].tolist()
    F12_over_F11 = (pm[:, 0, 1] / pm[:, 0, 0]).tolist()
    F22_over_F11 = (pm[:, 1, 1] / pm[:, 0, 0]).tolist()
    F33_over_F11 = (pm[:, 2, 2] / pm[:, 0, 0]).tolist()
    F34_over_F11 = (pm[:, 2, 3] / pm[:, 0, 0]).tolist()
    F44_over_F11 = (pm[:, 3, 3] / pm[:, 0, 0]).tolist()

    # Prepare results JSON structure
    results = {
        "case_id": case["case_id"],
        "input_parameters": {
            "family": case["family"],
            "n_monomers": case["n_monomers"],
            "radius_nm": case["radius_nm"],
            "wavelength_nm": case["wavelength_nm"],
            "m_real": case["m_real"],
            "m_imag": case["m_imag"],
            "seed": case["seed"],
            "monomer_order": case["monomer_order"],
            "aggregate_order": case["aggregate_order"],
            "angles_deg": case["angles_deg"],
            "tier": case["tier"]
        },
        "environment_metadata": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "numba_version": numba.__version__,
            "hostname": socket.gethostname(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "os": platform.system()
        },
        "timing_metadata": {
            "time_geom_s": t_geom_end - t_geom_start,
            "time_solve_s": t_solve_end - t_solve_start,
            "time_phase_s": t_phase_end - t_phase_start,
            "time_total_s": time.time() - t_start
        },
        "observables": {
            "Qext": Qext,
            "Qsca": Qsca,
            "Qabs": Qabs,
            "SSA": SSA,
            "g": g,
            "x_param": x_param,
            "xgeff": avgs["xgeff"]
        },
        "theta_deg": theta_deg.tolist(),
        "mueller": {
            "F11": F11,
            "F12_over_F11": F12_over_F11,
            "F22_over_F11": F22_over_F11,
            "F33_over_F11": F33_over_F11,
            "F34_over_F11": F34_over_F11,
            "F44_over_F11": F44_over_F11
        }
    }

    # Write output directory if not exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Successfully finished case {case['case_id']}. Results written to {args.out}")


if __name__ == "__main__":
    main()
