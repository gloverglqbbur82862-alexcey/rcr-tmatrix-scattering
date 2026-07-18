#!/usr/bin/env python3
"""Precision verification for all 6 independent Mueller matrix components (F11, F12, F22, F33, F34, F44) over 0-180 degrees against MSTM reference."""

import os
import numpy as np
import pytest
from geometry.geometry import build_bcca
from solver.bicgstab_solver import MultisphereRCRSystem
from tests.test_orientation_averaging import mie_coefficients_manual

# Resolve reference path to mstm output file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MSTM_OUT_PATH = os.path.join(TEST_DIR, "..", "experiments", "mstm_temp_n16.out")


def parse_mstm_scattering_matrix(filepath):
    """Parse Mueller matrix rows from MSTM .out file."""
    if not os.path.exists(filepath):
        pytest.skip(f"MSTM output file {filepath} not found.")
        
    angles = []
    f11_list = []
    f12_ratio = []
    f22_ratio = []
    f33_ratio = []
    f34_ratio = []
    f44_ratio = []
    
    start_parsing = False
    with open(filepath, "r") as f:
        for line in f:
            if "theta" in line and "11" in line and "12" in line and "22" in line:
                start_parsing = True
                continue
            if start_parsing:
                parts = line.strip().split()
                if len(parts) == 0:
                    # End of block
                    break
                try:
                    theta = float(parts[0])
                    f11 = float(parts[1])
                    f12 = float(parts[2])
                    f22 = float(parts[3])
                    f33 = float(parts[4])
                    f34 = float(parts[5])
                    f44 = float(parts[6])
                    
                    angles.append(theta)
                    f11_list.append(f11)
                    f12_ratio.append(f12)
                    f22_ratio.append(f22)
                    f33_ratio.append(f33)
                    f34_ratio.append(f34)
                    f44_ratio.append(f44)
                except ValueError:
                    # Non-data line or end of table
                    break
                    
    return (np.array(angles), 
            np.array(f11_list), 
            np.array(f12_ratio), 
            np.array(f22_ratio), 
            np.array(f33_ratio), 
            np.array(f34_ratio), 
            np.array(f44_ratio))


def test_full_mueller_matrix_accuracy():
    """Verify that all 6 independent normalized Mueller matrix elements match MSTM for N=16 BCCA across 181 angles (0-180 deg)."""
    # 1. Parse reference values
    ref_angles, ref_f11, ref_f12, ref_f22, ref_f33, ref_f34, ref_f44 = parse_mstm_scattering_matrix(MSTM_OUT_PATH)
    
    assert len(ref_angles) == 181
    
    # 2. Setup BCCA N=16 physical system
    r_nm = 25.0
    wl_nm = 550.0
    k = 2.0 * np.pi / wl_nm
    x_param = r_nm * k
    m_ri = 1.95 + 0.79j
    N = 16
    
    pos_raw, _ = build_bcca(n_monomers=N, radius=r_nm, overlap=0.0, seed=42)
    pos_d = pos_raw * k
    p = 3
    
    a, b = mie_coefficients_manual(m_ri, x_param, p)
    an_rcr = np.zeros((2, 2, p), dtype=np.complex128)
    diag = -0.5 * (a + b)
    off = -0.5 * (a - b)
    an_rcr[0, 0, :] = diag
    an_rcr[1, 1, :] = diag
    an_rcr[0, 1, :] = off
    an_rcr[1, 0, :] = off
    
    sys16 = MultisphereRCRSystem(pos_d, [p]*N, [an_rcr]*N, k_medium=1.0)
    agg_order = 6
    T_agg = sys16.calculate_aggregate_tmatrix(aggregate_order=agg_order, tol=1e-12)
    
    # 3. Compute phase matrix at all angles
    sm_coef = sys16.calculate_orientation_averaged_phase_matrix_coefficients(T_agg)
    pm = sys16.calculate_phase_matrix_from_coefficients(sm_coef, ref_angles, normalize=True)
    
    # Extract calculations
    calc_f11 = pm[:, 0, 0]
    calc_f12 = pm[:, 0, 1] / pm[:, 0, 0]
    calc_f22 = pm[:, 1, 1] / pm[:, 0, 0]
    calc_f33 = pm[:, 2, 2] / pm[:, 0, 0]
    calc_f34 = pm[:, 2, 3] / pm[:, 0, 0]
    calc_f44 = pm[:, 3, 3] / pm[:, 0, 0]
    
    # 4. Compare all 181 angles to MSTM
    # Tolerance is set to 1e-4 as per MSTM's output formatting precision (five decimals)
    max_tol = 1.2e-4
    
    for i in range(181):
        angle = ref_angles[i]
        
        # F11 (unpolarized phase function)
        assert abs(calc_f11[i] - ref_f11[i]) < max_tol, f"F11 mismatch at {angle} deg: calc={calc_f11[i]}, ref={ref_f11[i]}"
        
        # F12 / F11 (polarization ratio)
        assert abs(calc_f12[i] - ref_f12[i]) < max_tol, f"F12/F11 mismatch at {angle} deg: calc={calc_f12[i]}, ref={ref_f12[i]}"
        
        # F22 / F11 (depolarization ratio)
        assert abs(calc_f22[i] - ref_f22[i]) < max_tol, f"F22/F11 mismatch at {angle} deg: calc={calc_f22[i]}, ref={ref_f22[i]}"
        
        # F33 / F11
        assert abs(calc_f33[i] - ref_f33[i]) < max_tol, f"F33/F11 mismatch at {angle} deg: calc={calc_f33[i]}, ref={ref_f33[i]}"
        
        # F34 / F11
        assert abs(calc_f34[i] - ref_f34[i]) < max_tol, f"F34/F11 mismatch at {angle} deg: calc={calc_f34[i]}, ref={ref_f34[i]}"
        
        # F44 / F11
        assert abs(calc_f44[i] - ref_f44[i]) < max_tol, f"F44/F11 mismatch at {angle} deg: calc={calc_f44[i]}, ref={ref_f44[i]}"

    print("\n  Mueller Matrix analytical orientation averages verified over all angles 0..180 deg!")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
