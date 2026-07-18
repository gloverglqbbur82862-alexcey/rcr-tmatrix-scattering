import pytest
import yaml
import json
import csv
from apps.rcr_workbench.models import CaseConfig, GeometryConfig, OpticalConfig, SolverConfig
from apps.rcr_workbench.demo_loader import load_demo_result
from apps.rcr_workbench.exports import (
    export_case_yaml,
    export_results_json,
    export_global_observables_csv,
    export_phase_matrix_csv,
    generate_markdown_report
)

@pytest.fixture
def sample_case_and_result():
    case = CaseConfig(
        geometry=GeometryConfig(kind="bcca", N=16, monomer_radius_nm=25.0, seed=42),
        optics=OpticalConfig(wavelength_nm=550.0, refractive_index_real=1.95, refractive_index_imag=0.79),
        solver=SolverConfig(realization="demo", sphere_order=3, aggregate_order=6, demo_mode=True)
    )
    result = load_demo_result(case)
    return case, result

def test_export_case_yaml(sample_case_and_result):
    case, _ = sample_case_and_result
    yaml_str = export_case_yaml(case)
    assert yaml_str is not None
    assert "geometry:" in yaml_str
    assert "optics:" in yaml_str
    
    # Verify it is valid YAML
    parsed = yaml.safe_load(yaml_str)
    assert parsed["geometry"]["kind"] == "bcca"
    assert parsed["optics"]["wavelength_nm"] == 550.0

def test_export_results_json(sample_case_and_result):
    _, result = sample_case_and_result
    json_str = export_results_json(result)
    assert json_str is not None
    assert "qext" in json_str
    
    # Verify it is valid JSON
    parsed = json.loads(json_str)
    assert parsed["qext"] == result.qext
    assert "F11" in parsed["phase_matrix"]

def test_export_global_observables_csv(sample_case_and_result):
    case, result = sample_case_and_result
    csv_str = export_global_observables_csv(case, result)
    assert csv_str is not None
    
    # Parse CSV
    reader = csv.reader(csv_str.splitlines())
    rows = list(reader)
    assert len(rows) > 5
    assert rows[0] == ["Observables", "Value", "Description"]
    
    # find row for Qext
    qext_row = [r for r in rows if r[0] == "Qext"]
    assert len(qext_row) == 1
    assert float(qext_row[0][1]) == pytest.approx(result.qext, 1e-6)

    solver_row = [r for r in rows if r[0] == "solver_realization"]
    assert len(solver_row) == 1
    assert solver_row[0][1] == "precomputed reference"

def test_export_phase_matrix_csv(sample_case_and_result):
    _, result = sample_case_and_result
    csv_str = export_phase_matrix_csv(result)
    assert csv_str is not None
    
    reader = csv.reader(csv_str.splitlines())
    rows = list(reader)
    assert len(rows) == 182 # 1 header + 181 angles
    assert rows[0] == ["Angle_deg", "F11", "F12_over_F11", "F22_over_F11", "F33_over_F11", "F34_over_F11", "F44_over_F11"]

def test_generate_markdown_report(sample_case_and_result):
    case, result = sample_case_and_result
    report = generate_markdown_report(case, result)
    assert report is not None
    assert "# RCR Scattering Workbench Calculation Report" in report
    assert "## Case Settings" in report
    assert "## Integrated Optical Properties" in report
