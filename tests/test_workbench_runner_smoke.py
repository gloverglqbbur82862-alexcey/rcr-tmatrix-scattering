import pytest
from apps.rcr_workbench.models import CaseConfig, GeometryConfig, OpticalConfig, SolverConfig
from apps.rcr_workbench.runner import run_case

def test_runner_demo_mode():
    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=16, monomer_radius_nm=25.0, seed=42),
        optics=OpticalConfig(wavelength_nm=550.0, refractive_index_real=1.95, refractive_index_imag=0.79),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order=6, demo_mode=True)
    )
    # Runner in demo mode should load the demo data without executing solver JIT
    res = run_case(case)
    assert res is not None
    assert res.metadata.get("demo") is True
    assert res.qext == pytest.approx(1.160511, abs=1e-4)

def test_runner_live_cpu_solve_smoke():
    # Setup a very small N=4 cluster with low truncation order p=1 for ultra-fast execution
    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=4, monomer_radius_nm=25.0, seed=42),
        optics=OpticalConfig(wavelength_nm=550.0, refractive_index_real=1.95, refractive_index_imag=0.79),
        solver=SolverConfig(realization="cpu_serial", sphere_order=1, aggregate_order=2, demo_mode=False)
    )
    
    # We call run_case to perform a live calculation
    res = run_case(case)
    
    assert res is not None
    assert res.metadata.get("demo") is False
    assert res.metadata.get("solver_realization") == "cpu_serial"
    assert "backend" not in res.metadata
    assert res.qext > 0.0
    assert res.qsca > 0.0
    assert res.ssa > 0.0
    assert len(res.angles_deg) == 181
    assert "F11" in res.phase_matrix
    assert len(res.phase_matrix["F11"]) == 181
    assert res.timing.get("solve_s") > 0.0
    assert res.timing.get("total_s") > 0.0
