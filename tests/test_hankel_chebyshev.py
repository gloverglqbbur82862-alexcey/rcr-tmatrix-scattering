#!/usr/bin/env python3
"""Tests for RCR Math Core: stable Hankel evaluation and Chebyshev interpolation."""

import numpy as np
import pytest
from math_core.chebyshev_hankel import (
    evaluate_exact_hankel,
    evaluate_recurrence_hankel,
    ChebyshevHankelInterpolator
)


def test_recurrence_vs_exact():
    """Verify that the stable forward recurrence matches SciPy exact values to machine precision."""
    L_max = 26
    # Test across multiple radial distances, including close near-field and far-field
    rhos = [0.57, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0]
    
    for rho in rhos:
        exact = evaluate_exact_hankel(L_max, rho)
        recur = evaluate_recurrence_hankel(L_max, rho)
        
        for L in range(L_max + 1):
            rel_err = abs(exact[L] - recur[L]) / abs(exact[L])
            # Tolerance is set to 1.5e-14 to account for floating-point differences in recursion vs special library calls
            assert rel_err < 1.5e-14, f"Recurrence failed at L={L}, rho={rho} with rel_err={rel_err:.2e}"


def test_chebyshev_interpolation():
    """Verify that the Chebyshev interpolator achieves extreme precision inside a small interval."""
    L_max = 26
    # In FMM, grid intervals are small (e.g. spacing ~0.03 for sixteenth subdivision)
    rho_min = 0.57
    rho_max = 0.60
    
    # Using degree 12 on a small interval achieves machine precision
    interp = ChebyshevHankelInterpolator(L_max, rho_min, rho_max, deg=12)
    
    # Check intermediate test points
    test_rhos = [0.58, 0.59]
    
    for rho in test_rhos:
        exact = evaluate_exact_hankel(L_max, rho)
        approx = interp.evaluate(rho)
        
        for L in range(L_max + 1):
            rel_err = abs(exact[L] - approx[L]) / abs(exact[L])
            # High-order Chebyshev on narrow interval is perfect
            assert rel_err < 1e-14, f"Chebyshev interpolation failed at L={L}, rho={rho} with rel_err={rel_err:.2e}"


def test_array_evaluation():
    """Verify that both recurrence and Chebyshev methods support numpy arrays correctly."""
    L_max = 13
    rhos = np.array([1.01, 1.05, 1.09])
    
    exact_arr = evaluate_exact_hankel(L_max, rhos)
    recur_arr = evaluate_recurrence_hankel(L_max, rhos)
    
    assert exact_arr.shape == (L_max + 1, 3)
    assert recur_arr.shape == (L_max + 1, 3)
    
    # Verify values match
    diff = np.abs(exact_arr - recur_arr) / np.abs(exact_arr)
    assert np.all(diff < 1e-14)
    
    # Verify Chebyshev array evaluation on a small interval (width 0.1)
    interp = ChebyshevHankelInterpolator(L_max, 1.0, 1.1, deg=12)
    interp_arr = interp.evaluate(rhos)
    assert interp_arr.shape == (L_max + 1, 3)
    
    diff_interp = np.abs(exact_arr - interp_arr) / np.abs(exact_arr)
    assert np.all(diff_interp < 2e-14)

