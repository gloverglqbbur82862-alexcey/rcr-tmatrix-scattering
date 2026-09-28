from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def scalar_mode_index(n: int, m: int) -> int:
    """Zero-based index within one polarization block.

    s(n,m) = n*(n+1) + m - 1
    """
    return n * (n + 1) + m - 1


@njit(cache=True)
def packed_mode_index(p: int, tau: int, n: int, m: int) -> int:
    """Zero-based packed index in the two-block RCR basis.

    tau = 1 -> Psi^(1)
    tau = 2 -> Psi^(2)
    """
    return (tau - 1) * p * (p + 2) + scalar_mode_index(n, m)


def mie_ab_to_rcr_basis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert standard Mie a_n/b_n coefficients to the transformed two-block RCR polarization basis."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    if a.shape != b.shape:
        raise ValueError(
            "a and b must have identical shapes; "
            f"received {a.shape} and {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError("a and b must be one-dimensional arrays")

    result = np.empty((2, 2, a.size), dtype=np.complex128)
    diagonal = -0.5 * (a + b)
    off_diagonal = -0.5 * (a - b)
    result[0, 0, :] = diagonal
    result[1, 1, :] = diagonal
    result[0, 1, :] = off_diagonal
    result[1, 0, :] = off_diagonal
    return result
