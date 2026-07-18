import yaml
import json
import csv
import io
import time
from apps.rcr_workbench.models import CaseConfig, SolverResult, solver_realization_label

def export_case_yaml(case: CaseConfig) -> str:
    """Serialize CaseConfig model into YAML string."""
    return yaml.safe_dump(case.model_dump(), default_flow_style=False)

def export_results_json(result: SolverResult) -> str:
    """Serialize SolverResult model into JSON string."""
    return result.model_dump_json(indent=2)

def export_global_observables_csv(case: CaseConfig, result: SolverResult) -> str:
    """Format integrated optical properties into a CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Observables", "Value", "Description"])
    writer.writerow(["Qext", f"{result.qext:.8f}", "Extinction Efficiency Factor"])
    writer.writerow(["Qsca", f"{result.qsca:.8f}", "Scattering Efficiency Factor"])
    writer.writerow(["Qabs", f"{result.qabs:.8f}", "Absorption Efficiency Factor"])
    writer.writerow(["SSA", f"{result.ssa:.8f}", "Single Scattering Albedo (Qsca / Qext)"])
    writer.writerow(["g", f"{result.g:.8f}", "Asymmetry Parameter"])
    writer.writerow(["wavelength_nm", f"{case.optics.wavelength_nm:.2f}", "Incident Light Wavelength (nm)"])
    writer.writerow(["monomer_radius_nm", f"{case.geometry.monomer_radius_nm:.2f}", "Monomer Sphere Radius (nm)"])
    writer.writerow(["monomer_count_N", case.geometry.N, "Total number of monomers in aggregate"])
    solver_value = result.metadata.get("solver_realization", case.solver.realization)
    writer.writerow(["solver_realization", solver_realization_label(solver_value), "Solver realization"])
    return output.getvalue()

def export_phase_matrix_csv(result: SolverResult) -> str:
    """Format normalized Mueller matrix elements into columns in a CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Angle_deg", "F11", "F12_over_F11", "F22_over_F11", "F33_over_F11", "F34_over_F11", "F44_over_F11"])
    
    angles = result.angles_deg
    pm = result.phase_matrix
    
    for i in range(len(angles)):
        writer.writerow([
            f"{angles[i]:.2f}",
            f"{pm['F11'][i]:.8e}" if i < len(pm['F11']) else "",
            f"{pm['F12_over_F11'][i]:.8f}" if i < len(pm['F12_over_F11']) else "",
            f"{pm['F22_over_F11'][i]:.8f}" if i < len(pm['F22_over_F11']) else "",
            f"{pm['F33_over_F11'][i]:.8f}" if i < len(pm['F33_over_F11']) else "",
            f"{pm['F34_over_F11'][i]:.8f}" if i < len(pm['F34_over_F11']) else "",
            f"{pm['F44_over_F11'][i]:.8f}" if i < len(pm['F44_over_F11']) else ""
        ])
    return output.getvalue()

def generate_markdown_report(case: CaseConfig, result: SolverResult) -> str:
    """Generate a scientific summary report in markdown format."""
    report = f"""# RCR Scattering Workbench Calculation Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## Case Settings
- **Geometry Type**: {case.geometry.kind.upper()}
- **Number of Monomers (N)**: {case.geometry.N}
- **Monomer Radius**: {case.geometry.monomer_radius_nm} nm
- **Wavelength**: {case.optics.wavelength_nm} nm
- **Refractive Index**: {case.optics.refractive_index_real} + {case.optics.refractive_index_imag}j
- **Truncation Order (p)**: {case.solver.sphere_order}
- **Aggregate Order (P)**: {case.solver.aggregate_order}
- **Solver Realization**: {solver_realization_label(result.metadata.get("solver_realization", case.solver.realization))}

## Integrated Optical Properties
| Observable | Value | Description |
|---|---|---|
| **Qext** | {result.qext:.6f} | Extinction Efficiency Factor |
| **Qsca** | {result.qsca:.6f} | Scattering Efficiency Factor |
| **Qabs** | {result.qabs:.6f} | Absorption Efficiency Factor |
| **SSA** | {result.ssa:.6f} | Single Scattering Albedo |
| **g** | {result.g:.6f} | Asymmetry Parameter |

## Performance & Metadata
- **Total Duration**: {result.timing.get('total_s', 0.0):.3f} seconds
  - Geometry construction: {result.timing.get('geometry_s', 0.0):.3f} s
  - Solver execution: {result.timing.get('solve_s', 0.0):.3f} s
  - Phase matrix computation: {result.timing.get('phase_s', 0.0):.3f} s
- **Precomputed Demo Result**: {result.metadata.get('demo', False)}
- **Metadata Summary**: {json.dumps(result.metadata)}
"""
    return report
