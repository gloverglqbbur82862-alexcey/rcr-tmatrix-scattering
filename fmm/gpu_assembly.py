import numpy as np
from numba import njit, prange
from rcr_engine.wigner_rotation import _rotate_to_z_cached_jit, _rotate_from_z_cached_jit
from rcr_engine.coaxial_translation import _apply_coaxial_translation_loop

@njit(parallel=True, fastmath=True, nogil=True)
def _assemble_dense_matrix_jit(
    nsph: int,
    p: int,
    nblk_half: int,
    pair_r: np.ndarray,
    pair_phi: np.ndarray,
    pair_d_blocks: np.ndarray,
    an_array: np.ndarray,
    fnr_table: np.ndarray,
    tran_coef_arr: np.ndarray,
    dc: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Assemble the system matrix (I - T * M) of size (N*C) x (N*C) in-place on CPU.
    
    Uses block-wise translation for each pair of spheres (i, j) to reduce complexity.
    """
    C = 2 * nblk_half
    
    # Pre-fill diagonal blocks with Identity operator
    for i in range(nsph):
        row_offset = i * C
        for k in range(C):
            A[row_offset + k, row_offset + k] = 1.0
            
    wmax_global = 2 * p
    
    # Parallelize over target spheres
    for i in prange(nsph):
        an_i = an_array[i]
        row_offset = i * C
        
        for j in range(nsph):
            if i == j:
                continue
                
            col_offset = j * C
            r = pair_r[i, j]
            phi = pair_phi[i, j]
            d_block_ij = pair_d_blocks[i, j]
            
            # Precompute Hankel functions for the translation distance r
            hn_i = np.empty(wmax_global + 1, dtype=np.complex128)
            exp_phase = np.exp(1j * r)
            hn_i[0] = -1j * exp_phase / r
            hn_i[1] = (-1.0/r - 1j/(r**2)) * exp_phase
            for w in range(2, wmax_global + 1):
                hn_i[w] = ((2 * w - 1) / r) * hn_i[w - 1] - hn_i[w - 2]
                
            # Translate each basis vector e_k from sphere j to sphere i
            for k in range(C):
                e_k = np.zeros(C, dtype=np.complex128)
                e_k[k] = 1.0
                
                # 1. Rotate to coaxial frame (z-axis)
                x_rot = _rotate_to_z_cached_jit(p, nblk_half, phi, e_k, d_block_ij)
                
                # 2. Coaxial translation along the z-axis
                y_rot = _apply_coaxial_translation_loop(p, nblk_half, x_rot, hn_i, dc, tran_coef_arr)
                
                # 3. Rotate back to original frame
                y_incoming = _rotate_from_z_cached_jit(p, nblk_half, phi, y_rot, d_block_ij)
                
                # 4. Multiply by local sphere T-matrix (an_i)
                Z_ik = np.zeros(C, dtype=np.complex128)
                for deg in range(1, p + 1):
                    idx_start = deg * deg - 1
                    idx_end = idx_start + 2 * deg + 1
                    for offset_idx in range(idx_start, idx_end):
                        cx_p1 = y_incoming[offset_idx]
                        cx_p2 = y_incoming[offset_idx + nblk_half]
                        
                        Z_ik[offset_idx] = an_i[0, 0, deg-1] * cx_p1 + an_i[0, 1, deg-1] * cx_p2
                        Z_ik[offset_idx + nblk_half] = an_i[1, 0, deg-1] * cx_p1 + an_i[1, 1, deg-1] * cx_p2
                
                # Off-diagonal block element is -Z_ik
                for r_idx in range(C):
                    A[row_offset + r_idx, col_offset + k] = -Z_ik[r_idx]
                    
    return A

