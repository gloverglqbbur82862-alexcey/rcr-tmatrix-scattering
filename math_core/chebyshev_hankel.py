#!/usr/bin/env python3
"""RCR Math Core: Stable evaluation and Chebyshev interpolation of spherical Hankel functions."""

from __future__ import annotations
import numpy as np
import scipy.special as sp
from typing import Any


def evaluate_exact_hankel(L_max: int, rho: float | np.ndarray) -> np.ndarray:
    """Evaluate spherical Hankel functions of the first kind h_L^(1)(rho) using SciPy.
    
    This is used as the ground truth baseline.
    
    Args:
        L_max: Maximum order (L = 2p).
        rho: Distance (scalar or numpy array).
        
    Returns:
        Array of shape (L_max + 1, ...) containing complex Hankel values.
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    scalar_input = (rho_arr.ndim == 0)
    
    # Shape of output: (L_max + 1, *rho_arr.shape)
    if scalar_input:
        out = np.empty(L_max + 1, dtype=np.complex128)
        for L in range(L_max + 1):
            out[L] = sp.spherical_jn(L, float(rho_arr)) + 1j * sp.spherical_yn(L, float(rho_arr))
        return out
    else:
        out = np.empty((L_max + 1,) + rho_arr.shape, dtype=np.complex128)
        for L in range(L_max + 1):
            out[L] = sp.spherical_jn(L, rho_arr) + 1j * sp.spherical_yn(L, rho_arr)
        return out


def evaluate_recurrence_hankel(L_max: int, rho: float | np.ndarray) -> np.ndarray:
    """Evaluate spherical Hankel functions h_L^(1)(rho) using stable forward recurrence.
    
    Forward recurrence is extremely stable for Hankel functions of the first kind
    because they grow as L increases for a fixed rho.
    
    Args:
        L_max: Maximum order (L = 2p).
        rho: Distance (scalar or numpy array).
        
    Returns:
        Array of shape (L_max + 1, ...) containing complex Hankel values.
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    scalar_input = (rho_arr.ndim == 0)
    
    if scalar_input:
        r = float(rho_arr)
        h = np.empty(L_max + 1, dtype=np.complex128)
        exp_phase = np.exp(1j * r)
        
        # Base cases
        h[0] = -1j * exp_phase / r
        h[1] = (-1.0/r - 1j/(r**2)) * exp_phase
        
        # Forward recurrence: h_n = ((2n - 1) / r) * h_{n-1} - h_{n-2}
        for n in range(2, L_max + 1):
            h[n] = ((2 * n - 1) / r) * h[n - 1] - h[n - 2]
        return h
    else:
        shape = rho_arr.shape
        h = np.empty((L_max + 1,) + shape, dtype=np.complex128)
        exp_phase = np.exp(1j * rho_arr)
        
        # Base cases
        h[0] = -1j * exp_phase / rho_arr
        h[1] = (-1.0/rho_arr - 1j/(rho_arr**2)) * exp_phase
        
        for n in range(2, L_max + 1):
            h[n] = ((2 * n - 1) / rho_arr) * h[n - 1] - h[n - 2]
        return h


class ChebyshevHankelInterpolator:
    """High-precision 1D Chebyshev interpolator for spherical Hankel functions."""
    
    def __init__(self, L_max: int, rho_min: float, rho_max: float, deg: int):
        """Fit Chebyshev polynomials to Hankel functions over [rho_min, rho_max].
        
        Args:
            L_max: Maximum order L.
            rho_min: Minimum distance.
            rho_max: Maximum distance.
            deg: Chebyshev degree (number of coefficients = deg + 1).
        """
        self.L_max = L_max
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.deg = deg
        
        # Chebyshev nodes in [-1, 1]
        nodes_std = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
        # Map nodes to [rho_min, rho_max]
        self.nodes = 0.5 * (self.rho_max - self.rho_min) * nodes_std + 0.5 * (self.rho_max + self.rho_min)
        
        # Evaluate exact Hankel functions at nodes
        # Shape: (L_max + 1, deg + 1)
        node_vals = evaluate_recurrence_hankel(L_max, self.nodes)
        
        # Compute Chebyshev coefficients for each L
        # Shape of coeffs: (L_max + 1, deg + 1)
        self.coeffs = np.empty((L_max + 1, deg + 1), dtype=np.complex128)
        for L in range(L_max + 1):
            # Chebyshev transform
            for j in range(deg + 1):
                # T_j(x_k) = cos(j * acos(x_k))
                t_j = np.cos(j * np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
                self.coeffs[L, j] = (2.0 / (deg + 1)) * np.sum(node_vals[L] * t_j)
            # Adjust the 0-th coefficient
            self.coeffs[L, 0] /= 2.0

    def evaluate(self, rho: float | np.ndarray) -> np.ndarray:
        """Evaluate Hankel functions at rho using Chebyshev polynomials.
        
        Args:
            rho: Distance (scalar or numpy array).
            
        Returns:
            Array of shape (L_max + 1, ...) containing interpolated complex Hankel values.
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        
        # Check domain limits
        if np.any(rho_arr < self.rho_min) or np.any(rho_arr > self.rho_max):
            raise ValueError(f"rho outside interpolation domain [{self.rho_min}, {self.rho_max}]")
            
        # Map rho to [-1, 1]
        x = (2.0 * rho_arr - (self.rho_max + self.rho_min)) / (self.rho_max - self.rho_min)
        scalar_input = (rho_arr.ndim == 0)
        
        if scalar_input:
            out = np.zeros(self.L_max + 1, dtype=np.complex128)
            # Evaluate using Clenshaw recurrence for each L
            for L in range(self.L_max + 1):
                c = self.coeffs[L]
                d = 0.0
                dd = 0.0
                for j in range(self.deg, 0, -1):
                    temp = d
                    d = 2.0 * x * d - dd + c[j]
                    dd = temp
                out[L] = x * d - dd + c[0]
            return out
        else:
            shape = rho_arr.shape
            out = np.zeros((self.L_max + 1,) + shape, dtype=np.complex128)
            for L in range(self.L_max + 1):
                c = self.coeffs[L]
                d = np.zeros(shape, dtype=np.complex128)
                dd = np.zeros(shape, dtype=np.complex128)
                for j in range(self.deg, 0, -1):
                    temp = d.copy()
                    d = 2.0 * x * d - dd + c[j]
                    dd = temp
                out[L] = x * d - dd + c[0]
            return out
