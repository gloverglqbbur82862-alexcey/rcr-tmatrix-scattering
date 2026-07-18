#!/usr/bin/env python3
"""Numerical stability stress tests for high-order Wigner rotations and coaxial translation recurrences under extreme limits."""

import numpy as np
import pytest
from rcr_engine.wigner_rotation import wigner_d_small
from rcr_engine.coaxial_translation import CoaxialTranslationProvider


@pytest.mark.parametrize("p_order", [20, 30, 40])
def test_high_order_wigner_stability(p_order):
    """Stress test Wigner small-d matrix calculation up to degree p_order at extreme boundary angles (0, pi) and intermediate angles."""
    # Test angles: exactly 0, exactly pi, and very close to them, plus standard pi/4
    test_angles = [0.0, np.pi, 1e-15, np.pi - 1e-15, np.pi / 4.0]
    
    for theta in test_angles:
        for degree in range(1, p_order + 1):
            d_mat = wigner_d_small(degree, theta)
            
            # Assert no NaNs or Infinities are generated
            assert not np.any(np.isnan(d_mat)), f"NaN detected in Wigner d-matrix at p={degree}, theta={theta}"
            assert not np.any(np.isinf(d_mat)), f"Infinity detected in Wigner d-matrix at p={degree}, theta={theta}"
            
            # Check analytical boundary properties at theta = 0 (should be Identity matrix)
            if theta == 0.0:
                identity = np.eye(2 * degree + 1)
                assert np.allclose(d_mat, identity, atol=1e-14), f"Boundary identity verification failed at degree {degree}"
                
            # Check analytical properties at theta = pi (should be anti-diagonal with signs)
            if theta == np.pi:
                for i in range(2 * degree + 1):
                    for j in range(2 * degree + 1):
                        if i + j == 2 * degree:
                            expected_sign = (-1)**(degree - (i - degree)) # (-1)^(l - m)
                            assert abs(d_mat[i, j] - expected_sign) < 1e-14
                        else:
                            assert abs(d_mat[i, j]) < 1e-14


@pytest.mark.parametrize("p_order", [10, 20, 30])
def test_near_contact_translation_stability(p_order):
    """Stress test the coaxial translation operator at high order and near-contact separations."""
    # Setup translation provider
    provider = CoaxialTranslationProvider(p=p_order)
    
    # Test distances: near contact (rho -> 2 * radius where radius is 1.0, so rho = 2.0001)
    # and standard separations
    test_distances = [2.0001, 2.05, 5.0, 10.0]
    
    nblk_half = p_order * (p_order + 2)
    # Random complex coefficient vector
    rng = np.random.default_rng(42)
    x = rng.standard_normal(2 * nblk_half) + 1j * rng.standard_normal(2 * nblk_half)
    
    for dist in test_distances:
        # Regular Bessel translation
        y_reg = provider.apply_coaxial_translation(dist, x, regular=True)
        assert not np.any(np.isnan(y_reg)), f"NaN in regular translation at p={p_order}, dist={dist}"
        assert not np.any(np.isinf(y_reg)), f"Infinity in regular translation at p={p_order}, dist={dist}"
        
        # Singular Hankel translation (outgoing)
        y_sing = provider.apply_coaxial_translation(dist, x, regular=False)
        assert not np.any(np.isnan(y_sing)), f"NaN in singular translation at p={p_order}, dist={dist}"
        assert not np.any(np.isinf(y_sing)), f"Infinity in singular translation at p={p_order}, dist={dist}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
