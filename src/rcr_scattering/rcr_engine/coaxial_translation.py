#!/usr/bin/env python3
"""RCR Translation Engine: Coaxial block-diagonal translation along the z-axis with Numba JIT."""

from __future__ import annotations
import math
import numpy as np
import scipy.special as sp
from numba import njit
from typing import Any
from math_core.chebyshev_hankel import evaluate_recurrence_hankel
from rcr_engine.wigner_rotation import sh_index


def _precompute_fnr(nmax: int) -> np.ndarray:
    """Precompute square roots for MSTM functions."""
    size = 2 * (nmax + 2) ** 2 + 20
    fnr = np.zeros(size)
    for i in range(1, size):
        fnr[i] = math.sqrt(float(i))
    return fnr


def _precompute_bcof(nmax: int, fnr: np.ndarray) -> np.ndarray:
    """Precompute binomial coefficients bcof(n, l) = sqrt((n+l)! / (n! l!))."""
    nbc = 4 * nmax + 6
    bcof = np.zeros((nbc + 1, nbc + 1))
    bcof[0, 0] = 1.0
    for n in range(nbc):
        for l in range(n + 1, nbc + 1):
            bcof[n, l] = fnr[n + l] * bcof[n, l - 1] / fnr[l]
            bcof[l, n] = bcof[n, l]
        if n + 1 <= nbc:
            bcof[n + 1, n + 1] = (fnr[2 * n + 2] * fnr[2 * n + 1] *
                                  bcof[n, n] / fnr[n + 1] / fnr[n + 1])
    return bcof


def _vcfunc(m: int, n: int, k: int, l: int, bcof: np.ndarray, fnr: np.ndarray) -> dict[int, float]:
    """Compute vector coupling / Clebsch-Gordan Gaunt coefficients C(m,n; k,l; w) stably."""
    mk = m + k
    nm = n - m
    lk = l - k
    nmc = n + m
    lkc = l + k
    wmax = n + l
    wmin = max(abs(m + k), abs(n - l))
    if wmin > wmax:
        return {}

    vcn = {}
    # Initial value at w = wmax
    vcn[wmax] = bcof[nm, lk] * bcof[nmc, lkc] / bcof[n + n, l + l]

    if wmin == wmax:
        return {wmax: vcn[wmax]}

    # Initial value at w = wmax - 1
    denom_w1 = fnr[n + l + mk] * fnr[n + l - mk]
    if abs(denom_w1) < 1e-300 or fnr[l] < 1e-300 or fnr[n] < 1e-300:
        vcn[wmax - 1] = 0.0
    else:
        vcn[wmax - 1] = (vcn[wmax] * float(l * m - k * n)
                         * fnr[2 * (l + n) - 1]
                         / fnr[l] / fnr[n]
                         / denom_w1)

    if wmin == wmax - 1:
        return {w: vcn[w] for w in range(wmin, wmax + 1) if vcn[w] != 0.0}

    # Downward recurrence (3-term)
    vcmax = abs(vcn[wmax]) + abs(vcn[wmax - 1])
    w_switch = wmin

    for w in range(wmax, wmin + 1, -1):
        i_wmk = w - mk
        i_wpk = w + mk
        i_nlw = n - l + w
        i_lnw = l - n + w
        i_nlw1 = n + l - w + 1
        i_nlw2 = n + l + w + 1
        if (i_wmk < 0 or i_wpk < 0 or i_nlw < 0 or i_lnw < 0 or
                i_nlw1 < 0 or i_nlw2 < 0):
            vcn[w - 2] = 0.0
            continue
        denom_t1 = (fnr[i_wpk] * fnr[i_wmk] * fnr[i_nlw] * fnr[i_lnw]
                    * fnr[i_nlw1] * fnr[i_nlw2])
        if abs(denom_t1) < 1e-300:
            vcn[w - 2] = 0.0
            continue
        t1 = 2.0 * w * fnr[2 * w + 1] * fnr[2 * w - 1] / denom_t1

        if w <= 1:
            t2 = 0.0
        else:
            t2 = (float((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1))
                  / float(2 * w * (w - 1)))

        i_wmk1 = w - mk - 1
        i_wpk1 = w + mk - 1
        i_lnw1 = l - n + w - 1
        i_nlw1b = n - l + w - 1
        i_nlw2b = n + l - w + 2
        i_nlwb = n + l + w
        if (i_wmk1 < 0 or i_wpk1 < 0 or i_lnw1 < 0 or i_nlw1b < 0 or
                i_nlw2b < 0 or i_nlwb < 0 or w <= 1):
            vcn[w - 2] = 0.0
            continue
        denom_t3 = float(2 * (w - 1)) * fnr[2 * w - 3] * fnr[2 * w - 1]
        if abs(denom_t3) < 1e-300:
            vcn[w - 2] = 0.0
            continue
        t3 = (fnr[i_wmk1] * fnr[i_wpk1] * fnr[i_lnw1] * fnr[i_nlw1b]
              * fnr[i_nlw2b] * fnr[i_nlwb] / denom_t3)

        vcn[w - 2] = (t2 * vcn[w - 1] - vcn[w] / t1) / t3

        # Check if coefficients are decreasing (switch to upward recurrence)
        if (wmax - w) % 2 == 1:
            vctest = abs(vcn[w - 2]) + abs(vcn[w - 1])
            vcmax = max(vcmax, vctest)
            if vcmax > 0 and vctest / vcmax < 0.01:
                w_switch = w - 3
                break

    # Upward recurrence if needed
    if w_switch > wmin:
        _vcfuncuprec(m, n, k, l, w_switch, vcn, bcof, fnr)

    return {w: vcn.get(w, 0.0) for w in range(wmin, wmax + 1) if vcn.get(w, 0.0) != 0.0}


def _vcfuncuprec(m: int, n: int, k: int, l: int, wmax_up: int, vcn: dict[int, float], bcof: np.ndarray, fnr: np.ndarray) -> None:
    """Upward recurrence for Clebsch-Gordan Gaunt coefficients."""
    mk_abs = abs(m + k)
    nl_abs = abs(n - l)
    mk = m + k

    if nl_abs >= mk_abs:
        w = nl_abs
        if n >= l:
            m1, n1, l1, k1 = m, n, l, k
        else:
            m1, n1, l1, k1 = k, l, m, n
        vc1 = ((-1) ** (k1 + l1)
               * bcof[l1 + k1, w - m1 - k1]
               * bcof[l1 - k1, w + m1 + k1]
               * float((-1) ** (w - m1 - k1))  # Wait, let's keep the exact sign logic
               / bcof[l1 + l1, w + w + 1])
        # Note: the original code was bcof[l1+k1, w-m1-k1] * bcof[l1-k1, w+m1+k1] / bcof[l1+l1, w+w+1]
        # Let's verify the sign logic.
        vc1 = ((-1) ** (k1 + l1)
               * bcof[l1 + k1, w - m1 - k1]
               * bcof[l1 - k1, w + m1 + k1]
               / bcof[l1 + l1, w + w + 1])
    else:
        w = mk_abs
        if m + k >= 0:
            vc1 = ((-1) ** (n + m)
                   * bcof[n - l + w, l - k]
                   * bcof[l - n + w, n - m]
                   / bcof[w + w + 1, n + l - w])
        else:
            vc1 = ((-1) ** (l + k)
                   * bcof[n - l + w, l + k]
                   * bcof[l - n + w, n + m]
                   / bcof[w + w + 1, n + l - w])

    w1 = w
    vcn[w] = vc1
    w = w1 + 1
    w2 = min(wmax_up, n + l)

    if w2 > w1:
        i_wpk = w + mk
        i_wmk = w - mk
        i_nlw = n - l + w
        i_lnw = l - n + w
        i_nlw1 = n + l - w + 1
        i_nlw2 = n + l + w + 1
        denom_t1 = (fnr[i_wpk] * fnr[i_wmk] * fnr[i_nlw] * fnr[i_lnw]
                    * fnr[i_nlw1] * fnr[i_nlw2])
        if abs(denom_t1) > 1e-300:
            t1 = 2.0 * w * fnr[2 * w + 1] * fnr[2 * w - 1] / denom_t1
            if w1 == 0:
                t2 = 0.5 * float(m - k)
            else:
                t2 = (float((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1))
                      / float(2 * w * (w - 1)))
            vcn[w] = t1 * t2 * vcn[w1]

    for w in range(w1 + 2, w2 + 1):
        i_wpk = w + mk
        i_wmk = w - mk
        i_nlw = n - l + w
        i_lnw = l - n + w
        i_nlw1 = n + l - w + 1
        i_nlw2 = n + l + w + 1
        denom_t1 = (fnr[i_wpk] * fnr[i_wmk] * fnr[i_nlw] * fnr[i_lnw]
                    * fnr[i_nlw1] * fnr[i_nlw2])
        if abs(denom_t1) < 1e-300:
            vcn[w] = 0.0
            continue
        t1 = 2.0 * w * fnr[2 * w + 1] * fnr[2 * w - 1] / denom_t1
        t2 = (float((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1))
              / float(2 * w * (w - 1)))
        i_wmk1 = w - mk - 1
        i_wpk1 = w + mk - 1
        i_lnw1 = l - n + w - 1
        i_nlw1b = n - l + w - 1
        i_nlw2b = n + l - w + 2
        i_nlwb = n + l + w
        denom_t3 = float(2 * (w - 1)) * fnr[2 * w - 3] * fnr[2 * w - 1]
        if abs(denom_t3) < 1e-300:
            vcn[w] = 0.0
            continue
        t3 = (fnr[i_wmk1] * fnr[i_wpk1] * fnr[i_lnw1] * fnr[i_nlw1b]
              * fnr[i_nlw2b] * fnr[i_nlwb] / denom_t3)
        vcn[w] = t1 * (t2 * vcn[w - 1] - t3 * vcn[w - 2])


def _mstm_normalizedlegendre_zero(ct: float, nmax: int, fnr: np.ndarray, bcof: np.ndarray) -> dict[tuple[int, int], float]:
    """Compute MSTM normalized Legendre functions Y_w^0(ct) for v=0."""
    sbe = math.sqrt(max(0.0, (1.0 + ct) * (1.0 - ct)))
    dc = {}
    dc[(0, 0)] = 1.0
    dc[(0, 1)] = fnr[1] * ct * dc[(0, 0)]
    for n in range(1, nmax):
        nm = fnr[n] * fnr[n]
        np1m = fnr[n + 1] * fnr[n + 1]
        if abs(np1m) < 1e-300:
            dc[(0, n + 1)] = 0.0
        else:
            dc[(0, n + 1)] = (-(nm) * dc.get((0, n - 1), 0.0) + float(2 * n + 1) * ct * dc[(0, n)]) / np1m
    return dc


def _tran_coef_table(nodr_s: int, nodr_t: int, fnr: np.ndarray, bcof: np.ndarray) -> dict[tuple[int, int, int], float]:
    """Precompute the Gaunt translation coefficient table tran_coef(kl, mn, w)."""
    tran = {}
    ci = 1j
    for n in range(1, nodr_s + 1):
        for m in range(-n, n + 1):
            mn = n * (n + 1) + m
            for l in range(1, nodr_t + 1):
                # For coaxial translation, m = k (m-k = 0).
                k = m
                if abs(k) > l:
                    continue
                kl = l * (l + 1) + k
                
                wmax = n + l
                wmin = max(0, abs(n - l))
                
                c = -(ci ** (l - n)) * fnr[2 * l + 1] * fnr[2 * n + 1]
                m1m = (-1) ** k
                
                vc1 = _vcfunc(-k, l, m, n, bcof, fnr)
                vc2 = _vcfunc(-1, l, 1, n, bcof, fnr)
                
                for w in range(wmin, wmax + 1):
                    v1 = vc1.get(w, 0.0)
                    v2 = vc2.get(w, 0.0)
                    a = (ci ** w) * c * m1m * v1 * v2
                    if (wmax - w) % 2 == 0:
                        val = a.real
                    else:
                        val = a.imag
                    if val != 0.0:
                        tran[(kl, mn, w)] = val
    return tran


@njit(fastmath=True)
def _mstm_normalizedlegendre_zero_jit(ct: float, nmax: int, fnr: np.ndarray) -> np.ndarray:
    """Compute MSTM normalized Legendre functions Y_w^0(ct) with JIT."""
    dc = np.zeros(nmax + 1, dtype=np.float64)
    dc[0] = 1.0
    if nmax >= 1:
        dc[1] = fnr[1] * ct * dc[0]
    for n in range(1, nmax):
        nm = fnr[n] * fnr[n]
        np1m = fnr[n + 1] * fnr[n + 1]
        if abs(np1m) < 1e-300:
            dc[n + 1] = 0.0
        else:
            dc[n + 1] = (-(nm) * dc[n - 1] + float(2 * n + 1) * ct * dc[n]) / np1m
    return dc


@njit(fastmath=True)
def _apply_coaxial_translation_loop(
    p: int,
    nblk_half: int,
    x: np.ndarray,
    hn: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray
) -> np.ndarray:
    """Core translation logic JIT loop - optimized and extremely fast."""
    y = np.zeros_like(x)
    for m in range(-p, p + 1):
        n_start = max(1, abs(m))
        if n_start > p:
            continue
            
        size_m = p - n_start + 1
        for i in range(size_m):
            l = n_start + i
            kl = l * (l + 1) + m
            
            val_p1 = 0.0 + 0.0j
            val_p2 = 0.0 + 0.0j
            
            for j in range(size_m):
                n = n_start + j
                mn = n * (n + 1) + m
                
                wmax = n + l
                wmin = abs(n - l)
                
                src_idx1 = mn - 1
                src_idx2 = src_idx1 + nblk_half
                x_val1 = x[src_idx1]
                x_val2 = x[src_idx2]
                
                a1 = 0.0 + 0.0j
                b1 = 0.0 + 0.0j
                for w in range(wmax, wmin - 1, -1):
                    tc = tran_coef_arr[kl, mn, w]
                    if tc == 0.0:
                        continue
                    hw_dc = hn[w] * dc[w] * tc
                    if (wmax - w) % 2 == 0:
                        a1 += hw_dc
                    else:
                        b1 += hw_dc
                        
                h1 = a1 + 1j * b1
                h2 = a1 - 1j * b1
                
                val_p1 += h1 * x_val1
                val_p2 += h2 * x_val2
                
            tgt_idx1 = kl - 1
            tgt_idx2 = tgt_idx1 + nblk_half
            y[tgt_idx1] = val_p1
            y[tgt_idx2] = val_p2
            
    return y


class CoaxialTranslationProvider:
    """Assembles and applies sparse coaxial block-diagonal translation along z-axis (JIT-accelerated)."""
    
    def __init__(self, p: int):
        """Initialize the coaxial translation provider for order p.
        
        Args:
            p: Truncation order of VSWFs.
        """
        self.p = p
        self.nblk_half = p * (p + 2)
        
        # Precompute constants
        self.fnr = _precompute_fnr(p)
        self.bcof = _precompute_bcof(p, self.fnr)
        self.tran_coef_dict = _tran_coef_table(p, p, self.fnr, self.bcof)
        
        # Precompute dense 3D array for JIT lookup
        wmax_global = 2 * p
        self.tran_coef_arr = np.zeros((self.nblk_half + 1, self.nblk_half + 1, wmax_global + 1), dtype=np.float64)
        for (kl, mn, w), val in self.tran_coef_dict.items():
            self.tran_coef_arr[kl, mn, w] = val

    def apply_coaxial_translation(self, rho: float, x: np.ndarray, regular: bool = False) -> np.ndarray:
        """Apply coaxial translation M_z(rho) along z-axis to vector x using sparse, block-diagonal JIT loop."""
        x = np.asarray(x, dtype=np.complex128)
        wmax_global = 2 * self.p
        
        # Evaluate Hankel functions or regular Bessel functions
        if regular:
            hn = np.array([sp.spherical_jn(w, rho) for w in range(wmax_global + 1)], dtype=np.complex128)
        else:
            hn = evaluate_recurrence_hankel(wmax_global, rho)
        
        # Legendre functions (2-normalization)
        dc = _mstm_normalizedlegendre_zero_jit(1.0, wmax_global, self.fnr)
        
        # Apply JIT-accelerated loop
        return _apply_coaxial_translation_loop(self.p, self.nblk_half, x, hn, dc, self.tran_coef_arr)
