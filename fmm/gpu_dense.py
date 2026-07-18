import numpy as np

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cp = None
    cupy_available = False

from fmm.gpu_assembly import _assemble_dense_matrix_jit

class GPUDenseRCRSystem:
    """GPU Dense RCR Solver Backend.
    
    Pre-assembles the system matrix (I - T * M) on CPU, uploads it to GPU,
    and performs standard matrix-vector products (GEMV/GEMM) and GPU-resident solves.
    """

    def __init__(self, positions: np.ndarray, nodr_list: list[int], an_list: list[np.ndarray], k_medium: float, dc: np.ndarray, tran_coef_arr: np.ndarray, fnr: np.ndarray):
        """Initialize and assemble the system matrix on CPU and upload to GPU."""
        if not cupy_available:
            raise RuntimeError("CuPy is not available on this system.")

        self.nsph = len(positions)
        self.p = nodr_list[0] if len(nodr_list) > 0 else 0
        if self.p > 7:
            raise ValueError(f"GPU direct solver backends only support truncation orders p <= 7, got p={self.p}.")
            
        self.nblk_half = self.p * (self.p + 2)
        self.k_medium = float(k_medium)
        self.C = 2 * self.nblk_half
        self.NC = self.nsph * self.C

        # Memory Guard: Estimate memory for the dense complex128 matrix A (16 bytes per entry)
        mem_required = self.NC * self.NC * 16
        
        # Check available GPU free VRAM
        try:
            free_vram, total_vram = cp.cuda.Device().mem_info
        except Exception:
            free_vram = 0
            
        # Check available CPU RAM
        cpu_avail = None
        try:
            import psutil
            cpu_avail = psutil.virtual_memory().available
        except ImportError:
            pass

        if free_vram > 0 and mem_required > free_vram:
            raise MemoryError(
                f"Required dense matrix size {mem_required / (1024**3):.2f} GiB "
                f"exceeds available GPU free memory {free_vram / (1024**3):.2f} GiB."
            )
            
        if cpu_avail is not None and mem_required > cpu_avail:
            raise MemoryError(
                f"Required dense matrix size {mem_required / (1024**3):.2f} GiB "
                f"exceeds available CPU memory {cpu_avail / (1024**3):.2f} GiB."
            )

        # 1. Precompute pair distances and angles on CPU for assembly
        self.pair_r = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        self.pair_theta = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        self.pair_phi = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        
        for i in range(self.nsph):
            for j in range(self.nsph):
                if i != j:
                    r_vec = positions[i] - positions[j]
                    r_mag = np.linalg.norm(r_vec)
                    self.pair_r[i, j] = r_mag * self.k_medium
                    if r_mag > 1e-14:
                        self.pair_theta[i, j] = np.arccos(r_vec[2] / r_mag)
                        self.pair_phi[i, j] = np.arctan2(r_vec[1], r_vec[0])

        # Precompute pair D-blocks on CPU for assembly
        from rcr_engine.wigner_rotation import _build_cached_d_jit
        from solver.bicgstab_solver import FAC_TABLE
        
        d_cache_size = sum((2*n + 1)**2 for n in range(1, self.p + 1))
        pair_d_blocks = np.zeros((self.nsph, self.nsph, d_cache_size), dtype=np.float64)
        for i in range(self.nsph):
            for j in range(self.nsph):
                if i != j:
                    pair_d_blocks[i, j] = _build_cached_d_jit(self.p, self.pair_theta[i, j], FAC_TABLE)

        an_array = np.array(an_list, dtype=np.complex128)

        # 2. Assemble dense operator matrix A on CPU using the parallel JIT assembler
        import time
        t0_asm = time.perf_counter()
        A_cpu = np.zeros((self.NC, self.NC), dtype=np.complex128)
        _assemble_dense_matrix_jit(
            self.nsph,
            self.p,
            self.nblk_half,
            self.pair_r,
            self.pair_phi,
            pair_d_blocks,
            an_array,
            fnr,
            tran_coef_arr,
            dc,
            A_cpu
        )
        t_asm = time.perf_counter() - t0_asm

        # 3. Upload A to GPU
        t0_h2d = time.perf_counter()
        self.A_gpu = cp.asarray(A_cpu, dtype=cp.complex128)
        t_h2d = time.perf_counter() - t0_h2d

        self.profile_data = {
            'assembly_time': t_asm,
            'h2d_time': t_h2d,
            'vram_mb': float(mem_required / (1024**2))
        }

    def apply_system_operator(self, X):
        """Apply system operator (I - T * M) using GPU dense matrix-vector product."""
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_gpu = cp.asarray(X, dtype=cp.complex128)
        else:
            X_gpu = X

        if not X_gpu.flags.c_contiguous:
            X_gpu = cp.ascontiguousarray(X_gpu)

        # Reshape to 2D for matmul if 1D
        orig_ndim = X_gpu.ndim
        if orig_ndim == 1:
            X_gpu = X_gpu[:, None]

        Y_gpu = self.A_gpu @ X_gpu

        if orig_ndim == 1:
            Y_gpu = Y_gpu.ravel()

        if is_numpy:
            return Y_gpu.get()
        return Y_gpu

    def solve(self, b_vec, tol=1e-12, max_iter=2000):
        """Solve system (I - T * M) X = b using a fully GPU-resident BiCGSTAB solver."""
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

        # Run GPU-resident batched solver
        X_gpu, res = self._bicgstab_gpu_resident(B_gpu, tol, max_iter)

        if orig_ndim == 1:
            X_gpu = X_gpu.ravel()

        if is_numpy:
            return X_gpu.get(), res
        return X_gpu, res

    def _bicgstab_single_precision(self, A_single, B_single, tol=1e-6, max_iter=150):
        """Single-precision inner solver. Returns (X, iters_run)."""
        N, n_col = B_single.shape
        X = cp.zeros((N, n_col), dtype=cp.complex64)
        
        R = B_single - A_single @ X
        R0_star = R.copy()
        
        rho = cp.sum(R0_star.conj() * R, axis=0)
        P = R.copy()
        
        b_norms = cp.linalg.norm(B_single, axis=0)
        b_norms[b_norms == 0.0] = 1.0
        
        iters_run = 0
        for it in range(max_iter):
            r_norms = cp.linalg.norm(R, axis=0)
            if cp.all(r_norms / b_norms < tol):
                break
            iters_run = it + 1
                
            V = A_single @ P
            denom = cp.sum(R0_star.conj() * V, axis=0)
            denom[denom == 0.0] = 1.0
            alpha = rho / denom
            
            S = R - V * alpha[None, :]
            s_norms = cp.linalg.norm(S, axis=0)
            if cp.all(s_norms / b_norms < tol):
                X = X + P * alpha[None, :]
                break
                
            T = A_single @ S
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
            
        return X, iters_run

    def _bicgstab_double_precision(self, B_gpu, tol=1e-12, max_iter=2000, X_init=None):
        """Standard double-precision BiCGSTAB solve, with optional initial guess X_init."""
        N, n_col = B_gpu.shape
        if X_init is not None:
            X = X_init.copy()
        else:
            X = cp.zeros((N, n_col), dtype=cp.complex128)
        
        R = B_gpu - self.A_gpu @ X
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
                
            V = self.A_gpu @ P
            denom = cp.sum(R0_star.conj() * V, axis=0)
            denom[denom == 0.0] = 1.0
            alpha = rho / denom
            
            S = R - V * alpha[None, :]
            s_norms = cp.linalg.norm(S, axis=0)
            if cp.all(s_norms / b_norms < tol):
                X = X + P * alpha[None, :]
                iters_run = it + 1
                break
                
            T = self.A_gpu @ S
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

    def _bicgstab_gpu_resident(self, B_gpu, tol=1e-12, max_iter=2000):
        """Mixed-precision GPU resident BiCGSTAB solver with iterative refinement."""
        self.profile_data['used_mixed_precision'] = True
        self.profile_data['used_fallback'] = False
        
        # Convert A to single precision
        A_single = self.A_gpu.astype(cp.complex64)
        
        # Initial solve in single precision
        B_single = B_gpu.astype(cp.complex64)
        X_single, init_iters = self._bicgstab_single_precision(A_single, B_single, tol=1e-6, max_iter=150)
        X = X_single.astype(cp.complex128)
        
        # Compute double-precision residual
        R = B_gpu - self.A_gpu @ X
        b_norms = cp.linalg.norm(B_gpu, axis=0)
        b_norms[b_norms == 0.0] = 1.0
        
        res = float(cp.max(cp.linalg.norm(R, axis=0) / b_norms))
        
        # Iterative refinement loop
        ref_iters = 0
        total_sp_iters = init_iters
        while res > tol and ref_iters < 5:
            R_single = R.astype(cp.complex64)
            D_single, ref_sp_iters = self._bicgstab_single_precision(A_single, R_single, tol=1e-6, max_iter=150)
            total_sp_iters += ref_sp_iters
            D = D_single.astype(cp.complex128)
            X = X + D
            
            R = B_gpu - self.A_gpu @ X
            res = float(cp.max(cp.linalg.norm(R, axis=0) / b_norms))
            ref_iters += 1
            
        self.profile_data['refinement_iters'] = ref_iters
        self.profile_data['sp_krylov_iters'] = total_sp_iters
        
        # Fallback if convergence is not reached
        if res > tol:
            self.profile_data['used_fallback'] = True
            X, res = self._bicgstab_double_precision(B_gpu, tol, max_iter, X_init=X)
        else:
            self.profile_data['krylov_iters'] = ref_iters
            
        self.profile_data['final_residual'] = res
        return X, res
