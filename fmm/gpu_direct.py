import numpy as np
from numba import njit

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cp = None
    cupy_available = False

@njit(fastmath=True, nogil=True)
def _precompute_pair_phase_jit(n: int, p: int, pair_phi: np.ndarray) -> np.ndarray:
    pair_phase = np.zeros((n, n, 2 * p + 1), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                phi = pair_phi[i, j]
                for m in range(-p, p + 1):
                    angle = m * phi
                    pair_phase[i, j, m + p] = np.cos(angle) + 1j * np.sin(angle)
    return pair_phase


@njit(fastmath=True, nogil=True)
def _precompute_pair_hn_jit(n: int, p: int, pair_r: np.ndarray) -> np.ndarray:
    wmax_global = 2 * p
    pair_hn = np.zeros((n, n, wmax_global + 1), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = pair_r[i, j]
                if r > 1e-14:
                    exp_real = np.cos(r)
                    exp_imag = np.sin(r)
                    
                    # w = 0
                    h0 = exp_imag / r - 1j * exp_real / r
                    pair_hn[i, j, 0] = h0
                    
                    # w = 1
                    h1 = (-exp_real / r + exp_imag / (r * r)) + 1j * (-exp_imag / r - exp_real / (r * r))
                    pair_hn[i, j, 1] = h1
                    
                    # Recurrence for w >= 2
                    for w in range(2, wmax_global + 1):
                        factor = (2.0 * w - 1.0) / r
                        pair_hn[i, j, w] = factor * pair_hn[i, j, w - 1] - pair_hn[i, j, w - 2]
    return pair_hn

# C++ CUDA kernel source code
CUDA_KERNEL_SRC = """
struct Complex {
    double x; // real
    double y; // imag
};

__device__ inline Complex c_add(Complex a, Complex b) {
    return {a.x + b.x, a.y + b.y};
}

__device__ inline Complex c_sub(Complex a, Complex b) {
    return {a.x - b.x, a.y - b.y};
}

__device__ inline Complex c_mul(Complex a, Complex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ inline Complex c_conj(Complex a) {
    return {a.x, -a.y};
}

#define MAX_COEFFS 256

extern "C" __global__ void apply_system_operator_gpu_kernel(
    const int n,
    const int p,
    const int nblk_half,
    const int B,
    const Complex* X,             // shape (n, 2*nblk_half, B)
    const double* pair_r,         // shape (n, n)
    const double* pair_phi,       // shape (n, n)
    const double* pair_d_blocks,   // shape (n, n, d_cache_size)
    const Complex* an_array,      // shape (n, 2, 2, p)
    const double* tran_coef_arr,   // shape (nblk_half+1, nblk_half+1, 2*p+1)
    const double* dc,             // shape (2*p + 1)
    const Complex* pair_phase,    // shape (n, n, 2*p+1)
    const Complex* pair_hn,       // shape (n, n, 2*p+1)
    Complex* AX                   // shape (n, 2*nblk_half, B)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= n || b >= B) return;
    
    int d_cache_size = 0;
    for (int deg = 1; deg <= p; ++deg) {
        d_cache_size += (2 * deg + 1) * (2 * deg + 1);
    }
    
    Complex incoming[MAX_COEFFS];
    for (int k = 0; k < 2 * nblk_half; ++k) {
        incoming[k] = {0.0, 0.0};
    }
    
    Complex x_rot[MAX_COEFFS];
    Complex y_rot[MAX_COEFFS];
    
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        
        int d_blocks_offset = (i * n + j) * d_cache_size;
        int phase_offset = (i * n + j) * (2 * p + 1);
        int hn_offset = (i * n + j) * (2 * p + 1);
        
        // 1. Rotate to Z
        int d_offset = 0;
        for (int deg = 1; deg <= p; ++deg) {
            int block_dim = 2 * deg + 1;
            for (int pol = 0; pol < 2; ++pol) {
                int offset = pol * nblk_half;
                int idx_start = offset + deg * deg - 1;
                
                Complex block_x_phased[16];
                for (int jj = 0; jj < block_dim; ++jj) {
                    int m = jj - deg;
                    Complex exp_phase = pair_phase[phase_offset + (m + p)];
                    Complex src_coeff = X[j * 2 * nblk_half * B + (idx_start + jj) * B + b];
                    block_x_phased[jj] = c_mul(src_coeff, exp_phase);
                }
                
                for (int ii = 0; ii < block_dim; ++ii) {
                    Complex val = {0.0, 0.0};
                    for (int jj = 0; jj < block_dim; ++jj) {
                        double d_val = pair_d_blocks[d_blocks_offset + d_offset + ii * block_dim + jj];
                        val.x += d_val * block_x_phased[jj].x;
                        val.y += d_val * block_x_phased[jj].y;
                    }
                    x_rot[idx_start + ii] = val;
                }
            }
            d_offset += block_dim * block_dim;
        }
        
        // 2. Coaxial translation (no Hankel recurrence calculation needed here!)
        for (int k = 0; k < 2 * nblk_half; ++k) {
            y_rot[k] = {0.0, 0.0};
        }
        
        for (int m = -p; m <= p; ++m) {
            int n_start = (1 > abs(m)) ? 1 : abs(m);
            if (n_start > p) continue;
            
            int size_m = p - n_start + 1;
            for (int ii = 0; ii < size_m; ++ii) {
                int l = n_start + ii;
                int kl = l * (l + 1) + m;
                
                Complex val_p1 = {0.0, 0.0};
                Complex val_p2 = {0.0, 0.0};
                
                for (int jj = 0; jj < size_m; ++jj) {
                    int n_deg = n_start + jj;
                    int mn = n_deg * (n_deg + 1) + m;
                    
                    int wmax = n_deg + l;
                    int wmin = abs(n_deg - l);
                    
                    int src_idx1 = mn - 1;
                    int src_idx2 = src_idx1 + nblk_half;
                    
                    Complex x_val1 = x_rot[src_idx1];
                    Complex x_val2 = x_rot[src_idx2];
                    
                    Complex a1 = {0.0, 0.0};
                    Complex b1 = {0.0, 0.0};
                    
                    for (int w = wmax; w >= wmin; --w) {
                        int tc_idx = kl * (nblk_half + 1) * (2 * p + 1) + mn * (2 * p + 1) + w;
                        double tc = tran_coef_arr[tc_idx];
                        if (tc == 0.0) continue;
                        
                        Complex hw = pair_hn[hn_offset + w];
                        double hw_dc_real = hw.x * dc[w] * tc;
                        double hw_dc_imag = hw.y * dc[w] * tc;
                        
                        if ((wmax - w) % 2 == 0) {
                            a1.x += hw_dc_real;
                            a1.y += hw_dc_imag;
                        } else {
                            b1.x += hw_dc_real;
                            b1.y += hw_dc_imag;
                        }
                    }
                    
                    Complex h1 = {a1.x - b1.y, a1.y + b1.x};
                    Complex h2 = {a1.x + b1.y, a1.y - b1.x};
                    
                    val_p1 = c_add(val_p1, c_mul(h1, x_val1));
                    val_p2 = c_add(val_p2, c_mul(h2, x_val2));
                }
                
                int tgt_idx1 = kl - 1;
                int tgt_idx2 = tgt_idx1 + nblk_half;
                y_rot[tgt_idx1] = val_p1;
                y_rot[tgt_idx2] = val_p2;
            }
        }
        
        // 3. Rotate back and accumulate to incoming
        d_offset = 0;
        for (int deg = 1; deg <= p; ++deg) {
            int block_dim = 2 * deg + 1;
            for (int pol = 0; pol < 2; ++pol) {
                int offset = pol * nblk_half;
                int idx_start = offset + deg * deg - 1;
                
                Complex block_x_temp[16];
                for (int jj = 0; jj < block_dim; ++jj) {
                    Complex val = {0.0, 0.0};
                    for (int ii = 0; ii < block_dim; ++ii) {
                        double d_val = pair_d_blocks[d_blocks_offset + d_offset + ii * block_dim + jj];
                        val.x += d_val * y_rot[idx_start + ii].x;
                        val.y += d_val * y_rot[idx_start + ii].y;
                    }
                    block_x_temp[jj] = val;
                }
                
                for (int jj = 0; jj < block_dim; ++jj) {
                    int m = jj - deg;
                    Complex exp_phase = c_conj(pair_phase[phase_offset + (m + p)]);
                    incoming[idx_start + jj] = c_add(incoming[idx_start + jj], c_mul(exp_phase, block_x_temp[jj]));
                }
            }
            d_offset += block_dim * block_dim;
        }
    }
    
    // Apply T-matrix and write output to AX
    for (int deg = 1; deg <= p; ++deg) {
        int idx_start = deg * deg - 1;
        int idx_end = idx_start + 2 * deg + 1;
        
        Complex an_00 = an_array[i * 4 * p + 0 * 2 * p + 0 * p + (deg - 1)];
        Complex an_01 = an_array[i * 4 * p + 0 * 2 * p + 1 * p + (deg - 1)];
        Complex an_10 = an_array[i * 4 * p + 1 * 2 * p + 0 * p + (deg - 1)];
        Complex an_11 = an_array[i * 4 * p + 1 * 2 * p + 1 * p + (deg - 1)];
        
        for (int offset_idx = idx_start; offset_idx < idx_end; ++offset_idx) {
            Complex cx_p1 = incoming[offset_idx];
            Complex cx_p2 = incoming[offset_idx + nblk_half];
            
            Complex z1 = c_add(c_mul(an_00, cx_p1), c_mul(an_01, cx_p2));
            Complex z2 = c_add(c_mul(an_10, cx_p1), c_mul(an_11, cx_p2));
            
            int X_idx1 = i * 2 * nblk_half * B + offset_idx * B + b;
            int X_idx2 = i * 2 * nblk_half * B + (offset_idx + nblk_half) * B + b;
            
            Complex x1 = X[X_idx1];
            Complex x2 = X[X_idx2];
            
            AX[X_idx1] = c_sub(x1, z1);
            AX[X_idx2] = c_sub(x2, z2);
        }
    }
}
"""


class GPUDirectRCRSystem:
    """GPU Direct RCR Solver Backend using CuPy RawKernel dynamic compilation."""

    def __init__(self, positions: np.ndarray, nodr_list: list[int], an_list: list[np.ndarray], k_medium: float, dc: np.ndarray, tran_coef_arr: np.ndarray):
        """Initialize the GPU direct solver by uploading arrays to GPU and compiling kernel."""
        self.profile_data = {
            'assembly_time': 0.0,
            'h2d_time': 0.0,
            'vram_mb': 0.0
        }
        if not cupy_available:
            raise RuntimeError("CuPy is not available on this system.")

        self.nsph = len(positions)
        self.p = nodr_list[0] if len(nodr_list) > 0 else 0
        if self.p > 7:
            raise ValueError(f"GPU direct solver backends only support truncation orders p <= 7, got p={self.p}.")
        self.nblk_half = self.p * (self.p + 2)
        self.k_medium = float(k_medium)

        # 1. Precompute pair geometries on CPU (extremely fast, avoids Py/CuPy roundtrips)
        pair_r_cpu = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        pair_phi_cpu = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        pair_theta_cpu = np.zeros((self.nsph, self.nsph), dtype=np.float64)

        for i in range(self.nsph):
            for j in range(self.nsph):
                if i != j:
                    r_vec = positions[i] - positions[j]
                    r_mag = float(np.linalg.norm(r_vec))
                    pair_r_cpu[i, j] = r_mag * self.k_medium
                    if r_mag > 1e-14:
                        pair_theta_cpu[i, j] = np.arccos(r_vec[2] / r_mag)
                        pair_phi_cpu[i, j] = np.arctan2(r_vec[1], r_vec[0])

        self.positions = cp.asarray(positions, dtype=cp.float64)
        self.pair_r = cp.asarray(pair_r_cpu, dtype=cp.float64)
        self.pair_phi = cp.asarray(pair_phi_cpu, dtype=cp.float64)

        # 2. Precompute pair D-blocks on CPU and upload
        from rcr_engine.wigner_rotation import _build_cached_d_jit
        from solver.bicgstab_solver import FAC_TABLE
        
        d_cache_size = sum((2*n + 1)**2 for n in range(1, self.p + 1))
        pair_d_blocks_cpu = np.zeros((self.nsph, self.nsph, d_cache_size), dtype=np.float64)
        
        for i in range(self.nsph):
            for j in range(self.nsph):
                if i != j:
                    pair_d_blocks_cpu[i, j] = _build_cached_d_jit(self.p, pair_theta_cpu[i, j], FAC_TABLE)
                    
        self.pair_d_blocks = cp.asarray(pair_d_blocks_cpu, dtype=cp.float64)

        # 3. Upload other tables
        self.an_array = cp.asarray(an_list, dtype=cp.complex128)  # shape (N, 2, 2, p)
        self.tran_coef_arr = cp.asarray(tran_coef_arr, dtype=cp.float64)
        self.dc = cp.asarray(dc, dtype=cp.float64)

        # 4. Precompute pair phase and Hankel lookup tables on CPU and upload
        pair_phase_cpu = _precompute_pair_phase_jit(self.nsph, self.p, pair_phi_cpu)
        pair_hn_cpu = _precompute_pair_hn_jit(self.nsph, self.p, pair_r_cpu)
        
        self.pair_phase = cp.asarray(pair_phase_cpu, dtype=cp.complex128)
        self.pair_hn = cp.asarray(pair_hn_cpu, dtype=cp.complex128)

        # 5. Compile the raw CUDA C++ kernel
        self.kernel = cp.RawKernel(CUDA_KERNEL_SRC, "apply_system_operator_gpu_kernel")
        self.kernel.compile()

    def apply_system_operator(self, X):
        """Apply system operator (I - T * M) to the concatenated vector X on GPU."""
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_gpu = cp.asarray(X, dtype=cp.complex128)
        else:
            X_gpu = X

        if not X_gpu.flags.c_contiguous:
            X_gpu = cp.ascontiguousarray(X_gpu)

        # Reshape to 3D: (nsph, 2 * nblk_half, B)
        orig_shape = X_gpu.shape
        if X_gpu.ndim == 1:
            B = 1
            X_reshaped = X_gpu.reshape((self.nsph, 2 * self.nblk_half, 1))
        elif X_gpu.ndim == 2:
            B = X_gpu.shape[1]
            X_reshaped = X_gpu.reshape((self.nsph, 2 * self.nblk_half, B))
        else:
            raise ValueError(f"Input X must be 1D or 2D, got shape {X_gpu.shape}")

        AX_gpu = cp.empty_like(X_reshaped)

        # Grid execution configurations
        block = (16, 16)
        grid = ((self.nsph + 15) // 16, (B + 15) // 16)

        self.kernel(
            grid, block,
            (
                self.nsph,
                self.p,
                self.nblk_half,
                B,
                X_reshaped,
                self.pair_r,
                self.pair_phi,
                self.pair_d_blocks,
                self.an_array,
                self.tran_coef_arr,
                self.dc,
                self.pair_phase,
                self.pair_hn,
                AX_gpu
            )
        )

        # Format output back
        if len(orig_shape) == 1:
            AX_gpu = AX_gpu.ravel()
        else:
            AX_gpu = AX_gpu.reshape((self.nsph * 2 * self.nblk_half, B))

        if is_numpy:
            return AX_gpu.get()
        return AX_gpu

    def solve(self, b_vec, tol=1e-12, max_iter=2000):
        """Solve system (I - T * M) X = b_vec using GPU-native batched BiCGSTAB."""
        is_numpy = isinstance(b_vec, np.ndarray)
        if is_numpy:
            b_gpu = cp.asarray(b_vec, dtype=cp.complex128)
        else:
            b_gpu = b_vec

        if not b_gpu.flags.c_contiguous:
            b_gpu = cp.ascontiguousarray(b_gpu)

        orig_ndim = b_gpu.ndim
        if orig_ndim == 1:
            B_gpu = b_gpu[:, None]
        else:
            B_gpu = b_gpu

        # Run vectorized batched solver
        X_gpu, res = self._bicgstab_gpu_batched(B_gpu, tol, max_iter)

        if orig_ndim == 1:
            X_gpu = X_gpu.ravel()

        if is_numpy:
            return X_gpu.get(), res
        return X_gpu, res

    def _bicgstab_gpu_batched(self, B_gpu, tol=1e-12, max_iter=2000):
        N, n_col = B_gpu.shape
        X = cp.zeros((N, n_col), dtype=cp.complex128)
        R = B_gpu - self.apply_system_operator(X)
        R0_star = R.copy()
        
        rho = cp.sum(R0_star.conj() * R, axis=0)
        P = R.copy()
        
        b_norms = cp.linalg.norm(B_gpu, axis=0)
        b_norms[b_norms == 0.0] = 1.0
        
        iters_run = 0
        for it in range(max_iter):
            iters_run = it
            r_norms = cp.linalg.norm(R, axis=0)
            rel_norms = r_norms / b_norms
            if cp.all(rel_norms < tol):
                break
                
            V = self.apply_system_operator(P)
            denom = cp.sum(R0_star.conj() * V, axis=0)
            denom[denom == 0.0] = 1.0
            alpha = rho / denom
            
            S = R - V * alpha[None, :]
            
            s_norms = cp.linalg.norm(S, axis=0)
            if cp.all(s_norms / b_norms < tol):
                X = X + P * alpha[None, :]
                iters_run = it + 1
                break
                
            T = self.apply_system_operator(S)
            
            t_norm2 = cp.sum(T.conj() * T, axis=0).real
            t_norm2[t_norm2 == 0.0] = 1.0
            omega = cp.sum(T.conj() * S, axis=0) / t_norm2
            
            X = X + P * alpha[None, :] + S * omega[None, :]
            R = S - T * omega[None, :]
            
            rho_new = cp.sum(R0_star.conj() * R, axis=0)
            rho[rho == 0.0] = 1.0
            omega[omega == 0.0] = 1.0
            beta = (rho_new / rho) * (alpha / omega)
            
            rho = rho_new
            P = R + (P - V * omega[None, :]) * beta[None, :]
            iters_run = it + 1
            
        self.profile_data['krylov_iters'] = iters_run
        final_res = float(cp.max(cp.linalg.norm(R, axis=0) / b_norms))
        return X, final_res
