import time
import numpy as np
from scipy.special import spherical_jn, spherical_yn
from typing import Tuple

from geometry.geometry import build_bcca, build_bpca
from solver.bicgstab_solver import MultisphereRCRSystem
from apps.rcr_workbench.models import CaseConfig, SolverResult
from apps.rcr_workbench.demo_loader import load_demo_result

def mie_coefficients_manual(m_complex: complex, x: float, n_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Mie coefficients an and bn for a sphere."""
    m = m_complex
    an_list = []
    bn_list = []
    
    def psi(nn: int, z: complex) -> complex:
        return z * spherical_jn(nn, z)
        
    def dpsi(nn: int, z: complex) -> complex:
        return spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)
        
    def xi(nn: int, z: complex) -> complex:
        return z * (spherical_jn(nn, z) + 1j * spherical_yn(nn, z))
        
    def dxi(nn: int, z: complex) -> complex:
        h = spherical_jn(nn, z) + 1j * spherical_yn(nn, z)
        dh = (spherical_jn(nn, z, derivative=True) + 1j * spherical_yn(nn, z, derivative=True))
        return h + z * dh
        
    for n in range(1, n_max + 1):
        psi_n_x = psi(n, x)
        dpsi_n_x = dpsi(n, x)
        psi_n_mx = psi(n, m * x)
        dpsi_n_mx = dpsi(n, m * x)
        xi_n_x = xi(n, x)
        dxi_n_x = dxi(n, x)
        
        a_n = (m * psi_n_mx * dpsi_n_x - psi_n_x * dpsi_n_mx) / (m * psi_n_mx * dxi_n_x - xi_n_x * dpsi_n_mx)
        b_n = (psi_n_mx * dpsi_n_x - m * psi_n_x * dpsi_n_mx) / (psi_n_mx * dxi_n_x - m * xi_n_x * dpsi_n_mx)
        
        an_list.append(complex(a_n))
        bn_list.append(complex(b_n))
        
    return np.array(an_list), np.array(bn_list)

def run_case(case: CaseConfig, progress_callback=None) -> SolverResult:
    """Execute the electromagnetic scattering calculation or load demo data."""
    if case.solver.demo_mode or case.solver.realization == "demo":
        res = load_demo_result(case)
        if res is not None:
            return res
        raise FileNotFoundError(
            f"No precomputed demo result found for: {case.geometry.kind} N={case.geometry.N} "
            f"wavelength={case.optics.wavelength_nm}nm seed={case.geometry.seed}."
        )

    t_start = time.time()
    
    # 1. Build geometry
    kind = case.geometry.kind.lower()
    N = case.geometry.N
    radius_nm = case.geometry.monomer_radius_nm
    seed = case.geometry.seed or 42
    
    t_geom_start = time.time()
    if progress_callback:
        progress_callback("Generating aggregate geometry...")
        
    if kind == "bcca":
        pos_raw, radii_raw = build_bcca(n_monomers=N, radius=radius_nm, overlap=0.0, seed=seed)
    elif kind == "bpca":
        pos_raw, radii_raw = build_bpca(n_monomers=N, radius=radius_nm, overlap=0.0, seed=seed)
    elif kind == "upload":
        if not case.geometry.centers:
            raise ValueError("No coordinate centers provided for uploaded geometry.")
        pos_raw = np.array(case.geometry.centers)
        radii_raw = np.full(len(pos_raw), radius_nm)
    else:
        raise ValueError(f"Unknown geometry kind '{kind}'")
        
    t_geom_end = time.time()

    # 2. Formulate wave vector and Mie coefficients
    wavelength_nm = case.optics.wavelength_nm
    k = 2.0 * np.pi / wavelength_nm
    x_param = radius_nm * k
    m_complex = case.optics.refractive_index_real + 1j * case.optics.refractive_index_imag
    
    pos_d = pos_raw * k
    monomer_order = case.solver.sphere_order

    if progress_callback:
        progress_callback("Calculating single-sphere Mie coefficients...")
        
    a, b = mie_coefficients_manual(m_complex, x_param, monomer_order)
    an_rcr = np.zeros((2, 2, monomer_order), dtype=np.complex128)
    diag = -0.5 * (a + b)
    off = -0.5 * (a - b)
    an_rcr[0, 0, :] = diag
    an_rcr[1, 1, :] = diag
    an_rcr[0, 1, :] = off
    an_rcr[1, 0, :] = off

    # Map public solver realizations to the internal core solver names.
    core_solver_map = {
        "cpu_parallel": "direct_cpu_parallel",
        "cpu_serial": "direct_cpu_serial",
        "gpu_matrix_free": "direct_gpu",
        "gpu_dense": "direct_gpu_dense"
    }
    core_solver = core_solver_map.get(case.solver.realization, "direct_cpu_parallel")

    # Determine aggregate truncation order
    if isinstance(case.solver.aggregate_order, str) and case.solver.aggregate_order.lower() == "auto":
        aggregate_order = MultisphereRCRSystem.auto_aggregate_order(pos_d, x_param, k_medium=1.0)
    else:
        try:
            aggregate_order = int(case.solver.aggregate_order)
        except ValueError:
            aggregate_order = MultisphereRCRSystem.auto_aggregate_order(pos_d, x_param, k_medium=1.0)

    # 3. Formulate RCR Multisphere system and solve
    if progress_callback:
        progress_callback(f"Initializing Multisphere RCR system (aggregate order={aggregate_order}, solver={case.solver.realization})...")
        
    sys_rcr = MultisphereRCRSystem(
        pos_d,
        [monomer_order] * len(pos_raw),
        [an_rcr] * len(pos_raw),
        k_medium=1.0,
        backend=core_solver
    )
    
    t_solve_start = time.time()
    if progress_callback:
        progress_callback("Calculating aggregate T-matrix (solving column equations)...")
        
    T_agg = sys_rcr.calculate_aggregate_tmatrix(aggregate_order=aggregate_order, tol=1e-12)
    t_solve_end = time.time()

    # 4. Calculate global observables and phase matrix
    t_phase_start = time.time()
    if progress_callback:
        progress_callback("Computing analytical orientation averages and phase matrix...")
        
    avgs = sys_rcr.calculate_orientation_averages(T_agg, [x_param] * len(pos_raw))
    
    # Generate 181 angles for phase matrix
    theta_deg = np.linspace(0.0, 180.0, 181)
    sm_coef = sys_rcr.calculate_orientation_averaged_phase_matrix_coefficients(T_agg)
    pm = sys_rcr.calculate_phase_matrix_from_coefficients(sm_coef, theta_deg, normalize=True)
    t_phase_end = time.time()

    # 5. Extract values
    Qext = float(avgs["Qext"])
    Qsca = float(avgs["Qsca"])
    Qabs = float(avgs["Qabs"])
    SSA = float(Qsca / Qext if Qext > 0.0 else 0.0)
    g = float(avgs["g"])

    # Extract normalized Mueller matrix elements safely
    f11_den = pm[:, 0, 0]
    f11_safe = np.where(f11_den == 0.0, 1.0, f11_den)
    
    F11 = pm[:, 0, 0].tolist()
    F12_over_F11 = (pm[:, 0, 1] / f11_safe).tolist()
    F22_over_F11 = (pm[:, 1, 1] / f11_safe).tolist()
    F33_over_F11 = (pm[:, 2, 2] / f11_safe).tolist()
    F34_over_F11 = (pm[:, 2, 3] / f11_safe).tolist()
    F44_over_F11 = (pm[:, 3, 3] / f11_safe).tolist()

    phase_matrix = {
        "F11": F11,
        "F12_over_F11": F12_over_F11,
        "F22_over_F11": F22_over_F11,
        "F33_over_F11": F33_over_F11,
        "F34_over_F11": F34_over_F11,
        "F44_over_F11": F44_over_F11
    }

    t_total = time.time() - t_start

    if progress_callback:
        progress_callback("Scattering calculation complete!")

    return SolverResult(
        qext=Qext,
        qsca=Qsca,
        qabs=Qabs,
        ssa=SSA,
        g=g,
        angles_deg=theta_deg.tolist(),
        phase_matrix=phase_matrix,
        timing={
            "geometry_s": t_geom_end - t_geom_start,
            "solve_s": t_solve_end - t_solve_start,
            "phase_s": t_phase_end - t_phase_start,
            "total_s": t_total
        },
        metadata={
            "demo": False,
            "solver_realization": case.solver.realization,
            "core_solver": core_solver,
            "aggregate_order": aggregate_order,
            "monomer_count": len(pos_raw),
            "circumscribing_x": float(avgs.get("xgeff", 0.0)) # or similar size parameters
        }
    )
