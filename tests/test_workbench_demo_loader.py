import pytest
from apps.rcr_workbench.models import CaseConfig, GeometryConfig, OpticalConfig, SolverConfig
from apps.rcr_workbench.demo_loader import find_demo_file, load_demo_result

def test_find_demo_file():
    # Setup matching N=64 BCCA case
    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=64, monomer_radius_nm=25.0, seed=42),
        optics=OpticalConfig(wavelength_nm=550.0, refractive_index_real=1.95, refractive_index_imag=0.79),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order=6, demo_mode=True)
    )
    p = find_demo_file(case)
    assert p is not None
    assert p.exists()
    assert "bcca_n64" in p.name

def test_load_demo_result():
    # Setup matching N=64 BCCA case
    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=64, monomer_radius_nm=25.0, seed=42),
        optics=OpticalConfig(wavelength_nm=550.0, refractive_index_real=1.95, refractive_index_imag=0.79),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order=6, demo_mode=True)
    )
    res = load_demo_result(case)
    assert res is not None
    assert res.qext > 0.0
    assert res.qsca > 0.0
    assert res.ssa > 0.0
    assert len(res.angles_deg) == 181
    assert "F11" in res.phase_matrix
    assert len(res.phase_matrix["F11"]) == 181
    assert len(res.phase_matrix["F12_over_F11"]) == 181
    assert res.metadata.get("demo") is True
    assert res.metadata.get("solver_realization") == "demo"
    assert "backend" not in res.metadata

def test_load_demo_result_not_found():
    # Setup case that does not exist in demo files
    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=999, monomer_radius_nm=25.0, seed=99),
        optics=OpticalConfig(wavelength_nm=123.0, refractive_index_real=1.0, refractive_index_imag=0.0),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order=6, demo_mode=True)
    )
    p = find_demo_file(case)
    assert p is None
    res = load_demo_result(case)
    assert res is None
