#!/usr/bin/env python3
"""RCR Translation Engine: Wigner-D rotation of vector spherical harmonics with Numba JIT."""

from __future__ import annotations
import math
import numpy as np
from numba import njit
from typing import Any


# Precompute exact factorials as float64 up to 170 (float64 overflow limit is 170!)
FAC_TABLE = np.zeros(171, dtype=np.float64)
for i in range(171):
    FAC_TABLE[i] = float(math.factorial(i))


def sh_index(n: int, m: int) -> int:
    """Flat 0-based index for spherical harmonics mode (n, m).
    
    Index convention (model-1): index = n*(n+1) + m - 1 (for n>=1, m=-n..n)
    We offset by -1 because n starts from 1 (no n=0 monopole for EM vector waves).
    For a given order p, the flat half-block size is p*(p+2).
    """
    return n * (n + 1) + m - 1


def _fac(n: int) -> int:
    return math.factorial(int(n))


@njit(fastmath=True)
def _wigner_d_element_jit(l: int, m: int, mp: int, beta: float, fac_table: np.ndarray) -> float:
    """Compute Wigner small-d matrix element d^l_{m, mp}(beta) using exact formula."""
    if abs(m) > l or abs(mp) > l:
        return 0.0
    c = math.cos(beta / 2.0)
    s = math.sin(beta / 2.0)
    kmin = max(0, m - mp)
    kmax = min(l + m, l - mp)
    total = 0.0
    
    # pref = sqrt((l+m)! * (l-m)! * (l+mp)! * (l-mp)!)
    pref = math.sqrt(fac_table[l + m] * fac_table[l - m] * fac_table[l + mp] * fac_table[l - mp])
    
    for k in range(kmin, kmax + 1):
        den = fac_table[l + m - k] * fac_table[l - mp - k] * fac_table[k] * fac_table[mp - m + k]
        term = pref / den
        if k % 2 == 1:
            term = -term
        term *= (c ** (2 * l + m - mp - 2 * k)) * (s ** (mp - m + 2 * k))
        total += term
    return total


def wigner_d_element(l: int, m: int, mp: int, beta: float) -> float:
    """Compute Wigner small-d matrix element d^l_{m, mp}(beta) using exact JIT formula."""
    return _wigner_d_element_jit(l, m, mp, beta, FAC_TABLE)


@njit(fastmath=True)
def _wigner_d_small_jit(l: int, beta: float, fac_table: np.ndarray) -> np.ndarray:
    """Build the (2l+1) x (2l+1) Wigner small-d block matrix for degree l."""
    size = 2 * l + 1
    d = np.empty((size, size), dtype=np.float64)
    # Row index i corresponds to mp, column index j corresponds to m
    for i in range(size):
        mp = i - l
        for j in range(size):
            m = j - l
            d[i, j] = _wigner_d_element_jit(l, m, mp, beta, fac_table)
    return d


def wigner_d_small(l: int, beta: float) -> np.ndarray:
    """Build the (2l+1) x (2l+1) Wigner small-d block matrix for degree l (JIT accelerated)."""
    return _wigner_d_small_jit(l, beta, FAC_TABLE)


@njit(fastmath=True)
def _rotate_to_z_jit(p: int, nblk_half: int, theta: float, phi: float, x: np.ndarray, fac_table: np.ndarray) -> np.ndarray:
    """Rotate coefficient vector x to the z-axis frame using JIT contiguous slices."""
    y = np.zeros_like(x)
    for pol in range(2):
        offset = pol * nblk_half
        for n in range(1, p + 1):
            d_block = _wigner_d_small_jit(n, theta, fac_table)
            
            idx_start = offset + n * n - 1
            idx_end = idx_start + 2 * n + 1
            block_x = x[idx_start:idx_end]
            
            block_x_phased = np.empty(2 * n + 1, dtype=np.complex128)
            for j in range(2 * n + 1):
                m = j - n
                block_x_phased[j] = block_x[j] * np.exp(1j * m * phi)
            
            d_block_c = d_block.astype(np.complex128)
            block_y = d_block_c @ block_x_phased
            y[idx_start:idx_end] = block_y
    return y


@njit(fastmath=True)
def _rotate_from_z_jit(p: int, nblk_half: int, theta: float, phi: float, y: np.ndarray, fac_table: np.ndarray) -> np.ndarray:
    """Rotate coefficient vector y back from the z-axis frame using JIT contiguous slices."""
    x = np.zeros_like(y)
    for pol in range(2):
        offset = pol * nblk_half
        for n in range(1, p + 1):
            d_block = _wigner_d_small_jit(n, theta, fac_table)
            
            idx_start = offset + n * n - 1
            idx_end = idx_start + 2 * n + 1
            block_y = y[idx_start:idx_end]
            
            d_block_c = d_block.T.astype(np.complex128)
            block_x_temp = d_block_c @ block_y
            
            block_x = np.empty(2 * n + 1, dtype=np.complex128)
            for j in range(2 * n + 1):
                m = j - n
                block_x[j] = np.exp(-1j * m * phi) * block_x_temp[j]
                
            x[idx_start:idx_end] = block_x
    return x


class WignerRotationProvider:
    """Manages precomputed Wigner-D rotation blocks for FMM/RCR."""
    
    def __init__(self, p: int):
        """Initialize the rotation provider for order p.
        
        Args:
            p: Truncation order of VSWFs.
        """
        self.p = p
        self.nblk_half = p * (p + 2)
        self.nblk_full = 2 * self.nblk_half

    def get_rotation_blocks(self, beta: float) -> list[np.ndarray]:
        """Compute the Wigner small-d block matrices for each degree n = 1..p."""
        return [wigner_d_small(n, beta) for n in range(1, self.p + 1)]

    def rotate_to_z(self, theta: float, phi: float, x: np.ndarray) -> np.ndarray:
        """Rotate coefficient vector x to the z-axis frame (forward rotation R)."""
        return _rotate_to_z_jit(self.p, self.nblk_half, theta, phi, x, FAC_TABLE)

    def rotate_from_z(self, theta: float, phi: float, y: np.ndarray) -> np.ndarray:
        """Rotate coefficient vector y back from the z-axis frame (inverse rotation R^-1)."""
        return _rotate_from_z_jit(self.p, self.nblk_half, theta, phi, y, FAC_TABLE)


# ──────────────────────────────────────────────────────────────────────────────
# Cached Wigner-d rotation functions
#
# These store all d-matrix blocks (n=1..p_max) in a single flat array,
# avoiding recomputation on every BiCGSTAB iteration.
#
# Storage format for the flat d_blocks array:
#   For n = 1: d_blocks[0 : 9]           = d^1 flattened (3×3)
#   For n = 2: d_blocks[9 : 34]          = d^2 flattened (5×5)
#   For n = k: d_blocks[offset(k) : ...]  = d^k flattened ((2k+1)×(2k+1))
#
#   offset(n) = sum_{k=1}^{n-1} (2k+1)^2
# ──────────────────────────────────────────────────────────────────────────────

@njit(fastmath=True)
def _build_cached_d_jit(p_max: int, theta: float, fac_table: np.ndarray) -> np.ndarray:
    """Build the cached d-matrix flat array for all degrees n=1..p_max.
    
    Args:
        p_max: Maximum degree.
        theta: Polar angle (radians).
        fac_table: Precomputed factorial table.
    
    Returns:
        1D float64 array containing all d-blocks concatenated.
    """
    # Compute total size
    total_size = 0
    for n in range(1, p_max + 1):
        total_size += (2 * n + 1) ** 2
    
    d_blocks = np.empty(total_size, dtype=np.float64)
    
    offset = 0
    for n in range(1, p_max + 1):
        block_size = (2 * n + 1) ** 2
        d_block = _wigner_d_small_jit(n, theta, fac_table)
        # Flatten row-major into the cache
        for i in range(2 * n + 1):
            for j in range(2 * n + 1):
                d_blocks[offset + i * (2 * n + 1) + j] = d_block[i, j]
        offset += block_size
    
    return d_blocks


@njit(fastmath=True)
def _rotate_to_z_cached_jit(p: int, nblk_half: int, phi: float, x: np.ndarray, d_blocks: np.ndarray) -> np.ndarray:
    """Rotate coefficient vector x to the z-axis frame using pre-cached d-matrices.
    
    Same as _rotate_to_z_jit but reads d-matrices from d_blocks instead of computing them.
    
    Args:
        p: Truncation order.
        nblk_half: p*(p+2), half-block size.
        phi: Azimuthal angle (radians).
        x: Input coefficient vector of length 2*nblk_half.
        d_blocks: Pre-cached d-matrix flat array from _build_cached_d_jit.
    
    Returns:
        Rotated coefficient vector y of same length.
    """
    y = np.zeros_like(x)
    
    d_offset = 0
    for n in range(1, p + 1):
        block_dim = 2 * n + 1
        
        for pol in range(2):
            offset = pol * nblk_half
            idx_start = offset + n * n - 1
            
            # Apply phase factors: x_phased[j] = x[j] * exp(i*m*phi), m = j - n
            block_x_phased = np.empty(block_dim, dtype=np.complex128)
            for j in range(block_dim):
                m = j - n
                block_x_phased[j] = x[idx_start + j] * np.exp(1j * m * phi)
            
            # Matrix-vector multiply: y_block = d_block @ x_phased
            for i in range(block_dim):
                val = 0.0 + 0.0j
                for j in range(block_dim):
                    val += d_blocks[d_offset + i * block_dim + j] * block_x_phased[j]
                y[idx_start + i] = val
        
        d_offset += block_dim ** 2
    
    return y


@njit(fastmath=True)
def _rotate_from_z_cached_jit(p: int, nblk_half: int, phi: float, y: np.ndarray, d_blocks: np.ndarray) -> np.ndarray:
    """Rotate coefficient vector y back from the z-axis frame using pre-cached d-matrices.
    
    Same as _rotate_from_z_jit but reads d-matrices from d_blocks instead of computing them.
    Applies d^T (transpose) for the inverse rotation.
    
    Args:
        p: Truncation order.
        nblk_half: p*(p+2), half-block size.
        phi: Azimuthal angle (radians).
        y: Input coefficient vector of length 2*nblk_half.
        d_blocks: Pre-cached d-matrix flat array from _build_cached_d_jit.
    
    Returns:
        Un-rotated coefficient vector x of same length.
    """
    x = np.zeros_like(y)
    
    d_offset = 0
    for n in range(1, p + 1):
        block_dim = 2 * n + 1
        
        for pol in range(2):
            offset = pol * nblk_half
            idx_start = offset + n * n - 1
            
            # Matrix-vector multiply with TRANSPOSE: x_temp = d_block^T @ y_block
            block_x_temp = np.empty(block_dim, dtype=np.complex128)
            for j in range(block_dim):
                val = 0.0 + 0.0j
                for i in range(block_dim):
                    # d_block^T[j, i] = d_block[i, j]
                    val += d_blocks[d_offset + i * block_dim + j] * y[idx_start + i]
                block_x_temp[j] = val
            
            # Apply inverse phase: x[j] = exp(-i*m*phi) * x_temp[j]
            for j in range(block_dim):
                m = j - n
                x[idx_start + j] = np.exp(-1j * m * phi) * block_x_temp[j]
        
        d_offset += block_dim ** 2
    
    return x

