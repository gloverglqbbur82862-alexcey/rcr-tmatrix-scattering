import pytest
from pydantic import ValidationError
from apps.rcr_workbench.models import (
    GeometryConfig,
    OpticalConfig,
    SolverConfig,
    CaseConfig,
    SolverResult
)
from apps.rcr_workbench import components

def test_geometry_config_validation():
    # Valid config
    cfg = GeometryConfig(
        kind="bcca",
        N=16,
        monomer_radius_nm=25.0,
        seed=42,
        centers=[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]
    )
    assert cfg.kind == "bcca"
    assert cfg.N == 16
    assert cfg.monomer_radius_nm == 25.0
    assert len(cfg.centers) == 2

    # Invalid monomer radius
    with pytest.raises(ValidationError):
        GeometryConfig(
            kind="bcca",
            N=16,
            monomer_radius_nm="invalid_radius"
        )

def test_optical_config_validation():
    cfg = OpticalConfig(
        wavelength_nm=550.0,
        refractive_index_real=1.95,
        refractive_index_imag=0.79
    )
    assert cfg.wavelength_nm == 550.0
    assert cfg.refractive_index_real == 1.95
    assert cfg.refractive_index_imag == 0.79

def test_solver_config_validation():
    cfg = SolverConfig(
        realization="cpu_parallel",
        sphere_order=3,
        aggregate_order="auto",
        demo_mode=False
    )
    assert cfg.realization == "cpu_parallel"
    assert cfg.aggregate_order == "auto"
    assert cfg.demo_mode is False

    # Check aggregate order as integer
    cfg_int = SolverConfig(
        realization="cpu_parallel",
        sphere_order=3,
        aggregate_order=6,
        demo_mode=True
    )
    assert cfg_int.aggregate_order == 6
    assert cfg_int.demo_mode is True

def test_case_config_serialization():
    case = CaseConfig(
        geometry=GeometryConfig(kind="bpca", N=8, monomer_radius_nm=25.0, seed=123),
        optics=OpticalConfig(wavelength_nm=450.0, refractive_index_real=1.5, refractive_index_imag=0.01),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order="auto", demo_mode=True)
    )
    dumped = case.model_dump()
    assert dumped["geometry"]["kind"] == "bpca"
    assert dumped["optics"]["wavelength_nm"] == 450.0
    assert dumped["solver"]["demo_mode"] is True

def test_variables_panel_render_smoke(monkeypatch):
    """Render the variables panel with a stubbed Streamlit object to catch UI-only NameErrors."""
    rendered = []

    class StubStreamlit:
        @staticmethod
        def markdown(body, unsafe_allow_html=False):
            rendered.append(body)

    monkeypatch.setattr(components, "st", StubStreamlit)

    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=16, monomer_radius_nm=25.0, seed=42),
        optics=OpticalConfig(wavelength_nm=550.0, refractive_index_real=1.95, refractive_index_imag=0.79),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order="auto", demo_mode=True)
    )

    components.render_variables_panel(case, result=None)

    assert any("Variables" in body for body in rendered)
    assert any("centers" in body for body in rendered)
