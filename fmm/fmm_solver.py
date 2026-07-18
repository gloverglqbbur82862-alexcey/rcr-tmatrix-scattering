#!/usr/bin/env python3
"""FMM-accelerated multi-sphere solver.

This module provides FMMMultisphereSystem, a drop-in replacement for the
direct O(N^2) MultisphereRCRSystem that uses the hierarchical FMM operator
to reduce the Matvec cost from O(N^2) to O(N log N).

The solver interface is identical to MultisphereRCRSystem so that existing
test infrastructure works without modification.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from solver.bicgstab_solver import (
    apply_tmatrix_model1,
    bicgstab,
    MultisphereRCRSystem,
)
from fmm.fmm_operator import FMMOperator


def bicgstab_batch(
    A_op,
    B: np.ndarray,
    x0: np.ndarray = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve A X = B for multiple RHS columns simultaneously using BiCGSTAB.

    Args:
        A_op:     Operator that takes X of shape (N, B_cols) and returns A(X) of shape (N, B_cols).
        B:        RHS matrix of shape (N, B_cols).
        x0:       Initial guess of shape (N, B_cols).
        tol:      Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        X:        Solution matrix of shape (N, B_cols).
        res_norms:Residual norms of shape (B_cols,).
    """
    N, B_cols = B.shape
    if x0 is None:
        X = np.zeros_like(B)
    else:
        X = x0.copy()

    # Initial residual
    R = B - A_op(X)
    R_tilde = R.copy()

    # norms of RHS to check relative tolerance
    B_norms = np.linalg.norm(B, axis=0)
    # Avoid division by zero
    B_norms = np.where(B_norms < 1e-300, 1.0, B_norms)

    # Initial scalar variables
    rho_prev = np.ones(B_cols, dtype=np.complex128)
    alpha = np.ones(B_cols, dtype=np.complex128)
    omega = np.ones(B_cols, dtype=np.complex128)

    V = np.zeros_like(B)
    P = np.zeros_like(B)

    # Check initial convergence
    res_norms = np.linalg.norm(R, axis=0) / B_norms
    active_mask = res_norms > tol

    if not np.any(active_mask):
        return X, res_norms

    for it in range(max_iter):
        # rho = dot(R_tilde, R) column-wise
        rho = np.sum(R_tilde.conj() * R, axis=0)

        # beta = (rho / rho_prev) * (alpha / omega)
        beta = np.zeros(B_cols, dtype=np.complex128)
        div = rho_prev * omega
        safe_mask = np.abs(div) > 1e-300
        beta[safe_mask] = (rho[safe_mask] / rho_prev[safe_mask]) * (alpha[safe_mask] / omega[safe_mask])

        # P = R + beta * (P - omega * V)
        P = R + beta[np.newaxis, :] * (P - omega[np.newaxis, :] * V)

        # V = A(P)
        V = A_op(P)

        # alpha = rho / dot(R_tilde, V)
        den = np.sum(R_tilde.conj() * V, axis=0)
        alpha = np.zeros(B_cols, dtype=np.complex128)
        safe_den = np.abs(den) > 1e-300
        alpha[safe_den] = rho[safe_den] / den[safe_den]

        # S = R - alpha * V
        S = R - alpha[np.newaxis, :] * V

        # T = A(S)
        T = A_op(S)

        # omega = dot(T, S) / dot(T, T)
        num_om = np.sum(T.conj() * S, axis=0)
        den_om = np.sum(T.conj() * T, axis=0)
        omega = np.zeros(B_cols, dtype=np.complex128)
        safe_om = np.abs(den_om) > 1e-300
        omega[safe_om] = num_om[safe_om] / den_om[safe_om]

        # X = X + alpha * P + omega * S
        X = X + alpha[np.newaxis, :] * P + omega[np.newaxis, :] * S

        # R = S - omega * T
        R = S - omega[np.newaxis, :] * T

        # Check convergence
        res_norms = np.linalg.norm(R, axis=0) / B_norms
        active_mask = res_norms > tol
        if not np.any(active_mask):
            break

        rho_prev = rho

    return X, res_norms


class FMMMultisphereSystem:
    """FMM-accelerated version of MultisphereRCRSystem.

    Parameters
    ----------
    positions:    (N, 3) sphere center positions.
    nodr_list:    Truncation order for each sphere.
    an_list:      Mie coefficient array for each sphere, shape (2, 2, nodr).
    k_medium:     Wave-number in host medium.
    p_fmm:        Truncation order for FMM cell expansions.
                  Defaults to max(nodr_list).
    max_per_leaf: Maximum spheres per leaf cell. Smaller = more FMM levels.
    max_level:    Maximum tree depth.
    use_fmm:      If False, falls back to direct O(N^2) translation.
                  Useful for correctness testing.

    The system matrix is (I - T * M) where M is the FMM translation operator.
    """

    def __init__(
        self,
        positions: np.ndarray,
        nodr_list: list[int],
        an_list: list[np.ndarray],
        k_medium: float,
        p_fmm: Optional[int] = None,
        max_per_leaf: int = 4,
        max_level: int = 8,
        use_fmm: bool = True,
        fmm_mode: str = "reference",
        fmm_tol: Optional[float] = None,
        theta: float = 0.55,
        use_fmm_tmatrix: bool = False,
        N_switch: int = 512,
        use_precomputed_passes: bool = True,
        d_cache_budget_mb: float = 500.0,
        verbose: bool = True,
        backend: str = "auto",
    ):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.nodr_list = list(nodr_list)
        self.an_list = [np.asarray(an, dtype=np.complex128) for an in an_list]
        self.k_medium = float(k_medium)
        self.nsph = len(positions)
        self.fmm_mode = fmm_mode
        self.theta = float(theta)
        self.use_fmm_tmatrix = use_fmm_tmatrix
        self.N_switch = int(N_switch)
        self.use_precomputed_passes = use_precomputed_passes
        self.d_cache_budget_mb = d_cache_budget_mb
        self.verbose = verbose

        # Resolve backend selection
        self.backend = backend
        
        # Check CuPy availability
        cupy_available = False
        try:
            import cupy as cp
            cupy_available = cp.is_available()
        except Exception:
            pass

        # Legacy backward-compatible default resolution of use_fmm
        resolved_use_fmm = use_fmm
        if fmm_mode == "auto":
            # Dynamic crossover: use FMM only if N > N_switch
            if self.nsph <= self.N_switch:
                resolved_use_fmm = False
            else:
                resolved_use_fmm = True

        if self.backend == "auto":
            if resolved_use_fmm:
                # fmm_gpu is explicitly disabled/not implemented, resolve to CPU serial by default
                self.resolved_backend = "fmm_cpu_serial"
            else:
                if cupy_available:
                    self.resolved_backend = "direct_gpu"
                else:
                    self.resolved_backend = "direct_cpu_parallel"
        else:
            self.resolved_backend = self.backend

        if self.resolved_backend == "fmm_gpu":
            if not cupy_available:
                raise ValueError("Requested GPU backend 'fmm_gpu' but CUDA is not available on this system.")
            import warnings
            warnings.warn("fmm_gpu is not implemented. Falling back to fmm_cpu_serial.")
            self.resolved_backend = "fmm_cpu_serial"

        if self.resolved_backend == "fmm_cpu_parallel":
            import warnings
            warnings.warn("fmm_cpu_parallel is not yet implemented (Phase 4). Falling back to fmm_cpu_serial.")
            self.resolved_backend = "fmm_cpu_serial"

        if self.resolved_backend in ("direct_gpu", "direct_multi_gpu", "direct_gpu_dense") and not cupy_available:
            raise ValueError(f"Requested GPU backend '{self.resolved_backend}' but CuPy/CUDA is not available or usable on this system. CUDA is not available on this system.")

        # Map to use_fmm flag
        if self.resolved_backend in ("direct_cpu_serial", "direct_cpu_parallel", "direct_gpu", "direct_multi_gpu", "direct_gpu_dense"):
            self.use_fmm = False
        else:
            self.use_fmm = True

        # Resolve fmm_tol
        if fmm_tol is not None:
            self.fmm_tol = float(fmm_tol)
        else:
            self.fmm_tol = 1e-12 if fmm_mode == "reference" else 1e-6

        # Per-sphere block sizes
        self.nblk_half_list = [n * (n + 2) for n in self.nodr_list]
        self.nblk_full_list = [2 * nb for nb in self.nblk_half_list]
        self.N_total = sum(self.nblk_full_list)

        # Offsets for flat concatenated vector
        self.offsets = [0]
        for nb in self.nblk_full_list:
            self.offsets.append(self.offsets[-1] + nb)

        self._direct_system = None

        if self.use_fmm:
            self.p_fmm = p_fmm if p_fmm is not None else max(nodr_list)
            self.fmm_op = FMMOperator(
                positions=positions,
                nodr_list=nodr_list,
                k_medium=k_medium,
                p_fmm=self.p_fmm,
                max_per_leaf=max_per_leaf,
                max_level=max_level,
                fmm_mode=self.fmm_mode,
                fmm_tol=self.fmm_tol,
                theta=self.theta,
                use_precomputed_passes=self.use_precomputed_passes,
                d_cache_budget_mb=self.d_cache_budget_mb,
                verbose=self.verbose,
            )
        else:
            # Fall-back: use direct solver with chosen backend
            self._direct_system = MultisphereRCRSystem(
                positions, nodr_list, an_list, k_medium, backend=self.resolved_backend
            )

    # ------------------------------------------------------------------ #
    #  System operator                                                     #
    # ------------------------------------------------------------------ #

    def apply_system_operator(self, X: np.ndarray) -> np.ndarray:
        """Apply (I - T * M) X using FMM M-operator.

        Args:
            X: Concatenated scattered coefficient vector, shape (N_total,) or (N_total, B).

        Returns:
            Product vector, shape of same size as X.
        """
        if not self.use_fmm:
            return self._direct_system.apply_system_operator(X)

        X = np.asarray(X, dtype=np.complex128)
        AX = X.copy()
        
        is_2d = (X.ndim == 2)

        # Split X into per-sphere blocks
        if is_2d:
            X_blocks = [
                X[self.offsets[j]: self.offsets[j] + self.nblk_full_list[j], :]
                for j in range(self.nsph)
            ]
        else:
            X_blocks = [
                X[self.offsets[j]: self.offsets[j] + self.nblk_full_list[j]]
                for j in range(self.nsph)
            ]

        # FMM: compute incoming field on each sphere from all others
        incoming_blocks = self.fmm_op.apply_batch(X_blocks)

        # Apply T-matrix to incoming field: Z_i = T_i * incoming_i
        for i in range(self.nsph):
            off_i = self.offsets[i]
            nb_i = self.nblk_full_list[i]
            Z_i = apply_tmatrix_model1(incoming_blocks[i], self.an_list[i], self.nodr_list[i])
            if is_2d:
                AX[off_i: off_i + nb_i, :] -= Z_i
            else:
                AX[off_i: off_i + nb_i] -= Z_i

        return AX

    def solve(self, b_vec: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> tuple[np.ndarray, float]:
        """Solve (I - T * M) X = b using BiCGSTAB with FMM Matvec."""
        if b_vec.ndim == 2:
            return self.solve_batch(b_vec, tol=tol, max_iter=max_iter)
        if not self.use_fmm:
            return self._direct_system.solve(b_vec, tol=tol, max_iter=max_iter)
        x, res, _iters = bicgstab(self.apply_system_operator, b_vec, tol=tol, max_iter=max_iter)
        return x, res

    def solve_batch(self, B_mat: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Solve (I - T * M) X = B for multiple RHS columns using batched BiCGSTAB."""
        if not self.use_fmm:
            if self.resolved_backend in ("direct_gpu", "direct_multi_gpu", "direct_gpu_dense"):
                x, res = self._direct_system.solve(B_mat, tol=tol, max_iter=max_iter)
                if np.isscalar(res):
                    res_arr = np.full(B_mat.shape[1], res, dtype=np.float64)
                else:
                    res_arr = np.asarray(res)
                return x, res_arr
            else:
                # CPU direct backends do not support batched 2D RHS natively.
                # We solve column-by-column sequentially.
                X = np.zeros_like(B_mat)
                res_norms = np.zeros(B_mat.shape[1])
                for col in range(B_mat.shape[1]):
                    X[:, col], res_norms[col] = self._direct_system.solve(B_mat[:, col], tol=tol, max_iter=max_iter)
                return X, res_norms
        X, res_norms = bicgstab_batch(self.apply_system_operator, B_mat, tol=tol, max_iter=max_iter)
        return X, res_norms

    # ------------------------------------------------------------------ #
    #  Passthrough: aggregate T-matrix and orientation averaging          #
    # ------------------------------------------------------------------ #

    def translate_rcr_general(
        self,
        tvec: np.ndarray,
        x_j: np.ndarray,
        source_order: int,
        target_order: int,
        regular: bool = False,
    ) -> np.ndarray:
        """Translate coefficient vector x_j over arbitrary displacement tvec using RCR.
        Delegates to FMMOperator's translate_rcr_general if use_fmm is True.
        """
        if not self.use_fmm:
            return self._direct_system.translate_rcr_general(tvec, x_j, source_order, target_order, regular)
        return self.fmm_op.translate_rcr_general(tvec, x_j, source_order, target_order, regular)

    def calculate_aggregate_tmatrix(self, aggregate_order: int, tol: float = 1e-12) -> np.ndarray:
        """Calculate the full aggregate T-matrix of the sphere cluster using the FMM solver."""
        p_sphere = self.nodr_list[0] if self.nodr_list else 0
        if aggregate_order < p_sphere:
            raise ValueError(
                f"aggregate_order ({aggregate_order}) must be greater than or equal to "
                f"monomer order p_sphere ({p_sphere})."
            )

        if not self.use_fmm or not self.use_fmm_tmatrix:
            if self._direct_system is None:
                self._direct_system = MultisphereRCRSystem(
                    self.positions, self.nodr_list, self.an_list, self.k_medium
                )
            return self._direct_system.calculate_aggregate_tmatrix(aggregate_order, tol)

        import os
        import concurrent.futures

        nblk_half_agg = aggregate_order * (aggregate_order + 2)
        dim_agg = 2 * nblk_half_agg
        T_agg = np.zeros((dim_agg, dim_agg), dtype=np.complex128)

        col_keys = []
        for l in range(1, aggregate_order + 1):
            for k in range(-l, l + 1):
                for q in (1, 2):
                    col_keys.append((k, l, q))

        # Pre-warm/pre-initialize provider cache to avoid dictionary write race conditions
        max_p = max(aggregate_order, max(self.nodr_list))
        if hasattr(self.fmm_op, 'p_levels') and self.fmm_op.p_levels:
            max_p = max(max_p, max(self.fmm_op.p_levels))
        self.fmm_op.prewarm_cache(max_p)

        # Construct single B_mat of shape (self.N_total, dim_agg) for all columns
        B_mat = np.zeros((self.N_total, dim_agg), dtype=np.complex128)
        for idx, (k, l, q) in enumerate(col_keys):
            col_idx = l * (l + 1) + k - 1 + (q - 1) * nblk_half_agg

            # 1. Build common-origin incident vector (model-1 indexing)
            common_rhs = np.zeros(dim_agg, dtype=np.complex128)
            idx_sh = l * (l + 1) + k - 1
            common_rhs[idx_sh]              =  0.5
            common_rhs[idx_sh + nblk_half_agg] = -0.5 * ((-1) ** q)

            # 2. Distribute to each sphere & apply T-matrices → build RHS vector
            b_blocks = []
            for i in range(self.nsph):
                pos_i  = self.positions[i]
                nodr_i = self.nodr_list[i]
                an_i   = self.an_list[i]
                incident_i = self.translate_rcr_general(
                    pos_i, common_rhs, aggregate_order, nodr_i, regular=True
                )
                b_blocks.append(apply_tmatrix_model1(incident_i, an_i, nodr_i))
            b_vec = np.concatenate(b_blocks)
            B_mat[:, col_idx] = b_vec

        # 3. Solve the coupled linear system for all columns at once
        X_mat, res_norms = self.solve_batch(B_mat, tol=tol, max_iter=2000)

        # Enforce convergence validation check
        max_res = np.max(res_norms)
        if max_res > tol:
            raise RuntimeError(
                f"FMM batched solve did not converge to target tolerance {tol}. "
                f"Max residual: {max_res:.4e} across columns."
            )

        # 4. Merge per-sphere scattered fields back to common origin for each column
        for idx, (k, l, q) in enumerate(col_keys):
            col_idx = l * (l + 1) + k - 1 + (q - 1) * nblk_half_agg
            x_m1 = X_mat[:, col_idx]

            merged_lr = np.zeros(dim_agg, dtype=np.complex128)
            for i in range(self.nsph):
                pos_i  = self.positions[i]
                nodr_i = self.nodr_list[i]
                off_i  = self.offsets[i]
                nb_i   = self.nblk_full_list[i]
                x_i    = x_m1[off_i : off_i + nb_i]
                merged_lr += self.translate_rcr_general(
                    -pos_i, x_i, nodr_i, aggregate_order, regular=True
                )

            # 5. MSTM row transformation (model-1 → model-2 column)
            lr1 = merged_lr[:nblk_half_agg]
            lr2 = merged_lr[nblk_half_agg:]
            T_agg[:nblk_half_agg, col_idx] = lr1 + lr2
            T_agg[nblk_half_agg:, col_idx] = lr1 - lr2

        return T_agg

    def calculate_orientation_averages(self, T_agg: np.ndarray, radii_dimless: list[float]) -> dict[str, float]:
        """Compute analytical orientation averages Qext, Qsca, Qabs and asymmetry parameter g from aggregate T-matrix."""
        import math
        from solver.bicgstab_solver import precompute_fnr_bcof, calculate_orientation_averaged_phase_matrix_coefficients_jit, _precompute_vc_caches_jit

        xgeff = sum(r**3 for r in radii_dimless) ** (1.0 / 3.0)
        trace_sum = np.trace(T_agg).real
        fro_sq = np.vdot(T_agg, T_agg).real
        normalizer = 2.0 / (xgeff ** 2)
        Qext = -normalizer * trace_sum
        Qsca = normalizer * fro_sq
        Qabs = Qext - Qsca

        # Compute analytical orientation-averaged asymmetry parameter g
        agg_order = int(round(math.sqrt(T_agg.shape[0] // 2 + 1) - 1.0))
        nbc = 6 * agg_order + 6
        fnr_t, bcof_t = precompute_fnr_bcof(nbc)
        
        # Precompute or retrieve cached Clebsch-Gordan caches
        if not hasattr(self, 'vc_cache_A') or getattr(self, 'vc_cache_agg_order', 0) != agg_order:
            self.vc_cache_A, self.vc_cache_B = _precompute_vc_caches_jit(agg_order, fnr_t, bcof_t)
            self.vc_cache_agg_order = agg_order

        sm_coef = calculate_orientation_averaged_phase_matrix_coefficients_jit(
            T_agg, agg_order, 0.0, fnr_t, bcof_t, self.vc_cache_A, self.vc_cache_B
        )

        g = 0.0
        if abs(sm_coef[0, 0, 0]) > 1e-300:
            g = float(sm_coef[1, 0, 0] / (3.0 * sm_coef[0, 0, 0]))

        return {
            "Qext": float(Qext),
            "Qsca": float(Qsca),
            "Qabs": float(Qabs),
            "g": float(g),
            "xgeff": float(xgeff),
        }

    def calculate_orientation_averaged_phase_matrix_coefficients(self, T_agg: np.ndarray, cbeam: float = 0.0) -> np.ndarray:
        """Calculate the GSF expansion coefficients sm_coef of shape (2 * agg_order + 1, 4, 4)."""
        import math
        from solver.bicgstab_solver import precompute_fnr_bcof, calculate_orientation_averaged_phase_matrix_coefficients_jit, _precompute_vc_caches_jit
        agg_order = int(round(math.sqrt(T_agg.shape[0] // 2 + 1) - 1.0))
        nbc = 6 * agg_order + 6
        fnr_t, bcof_t = precompute_fnr_bcof(nbc)
        
        # Precompute or retrieve cached Clebsch-Gordan caches
        if not hasattr(self, 'vc_cache_A') or getattr(self, 'vc_cache_agg_order', 0) != agg_order:
            self.vc_cache_A, self.vc_cache_B = _precompute_vc_caches_jit(agg_order, fnr_t, bcof_t)
            self.vc_cache_agg_order = agg_order

        return calculate_orientation_averaged_phase_matrix_coefficients_jit(
            T_agg, agg_order, cbeam, fnr_t, bcof_t, self.vc_cache_A, self.vc_cache_B
        )

    def calculate_phase_matrix_from_coefficients(self, sm_coef: np.ndarray, theta_deg: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Calculate the 4x4 Mueller matrix at given theta angles (degrees) from GSF coefficients."""
        import math
        from solver.bicgstab_solver import precompute_fnr_bcof, calculate_phase_matrix_at_angle_jit
        nodrexp = sm_coef.shape[0] - 1
        nbc = 3 * nodrexp + 6
        fnr_t, bcof_t = precompute_fnr_bcof(nbc)
        
        res = np.zeros((len(theta_deg), 4, 4))
        for i in range(len(theta_deg)):
            ct = math.cos(math.radians(theta_deg[i]))
            res[i] = calculate_phase_matrix_at_angle_jit(ct, sm_coef, nodrexp, fnr_t, bcof_t)
            
        if normalize and abs(sm_coef[0, 0, 0]) > 1e-300:
            res /= (4.0 * sm_coef[0, 0, 0])
            
        return res

    def tree_summary(self) -> str:
        """Return octree structure summary string."""
        if self.use_fmm:
            return self.fmm_op.tree.summary()
        return "FMM disabled (using direct O(N^2) solver)"
