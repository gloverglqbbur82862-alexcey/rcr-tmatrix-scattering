import json
from pathlib import Path
from typing import Optional
from apps.rcr_workbench.models import CaseConfig, SolverResult

def find_demo_file(case: CaseConfig) -> Optional[Path]:
    """Search for a precomputed JSON result matching CaseConfig parameters."""
    kind = case.geometry.kind.lower()
    N = case.geometry.N
    wl = int(round(case.optics.wavelength_nm))
    seed = case.geometry.seed or 42
    
    # Try multiple naming candidates
    candidates = [
        f"{kind}_n{N}_w{wl}_seed{seed}.json",
        f"{kind}_n{N}_{wl}nm_result.json",
        f"{kind}_n{N}_w{wl}_seed42.json",
    ]
    
    # Check possible data directories
    search_dirs = [
        Path(__file__).resolve().parents[2] / "demo_data",
        Path(__file__).resolve().parents[2] / "apps" / "rcr_workbench" / "demo_data",
        Path.cwd() / "demo_data"
    ]
    
    for d in search_dirs:
        if d.exists():
            for c in candidates:
                p = d / c
                if p.exists():
                    return p
    return None

def load_demo_result(case: CaseConfig) -> Optional[SolverResult]:
    """Parse and load precomputed JSON data into a SolverResult object."""
    p = find_demo_file(case)
    if p is None:
        return None
        
    with open(p, "r") as f:
        data = json.load(f)
        
    obs = data.get("observables", {})
    mueller = data.get("mueller", {})
    timing = data.get("timing_metadata", {})
    
    # Safely get Mueller elements (supporting both formats: F12_over_F11 and F12_ratio)
    phase_matrix = {}
    for elem in ["F11", "F12_over_F11", "F22_over_F11", "F33_over_F11", "F34_over_F11", "F44_over_F11"]:
        if elem in mueller:
            phase_matrix[elem] = mueller[elem]
        else:
            # Fallback checks
            alt_key = elem.replace("_over_F11", "_ratio")
            if alt_key in mueller:
                phase_matrix[elem] = mueller[alt_key]
            else:
                phase_matrix[elem] = []
                
    qext = obs.get("Qext", 0.0)
    qsca = obs.get("Qsca", 0.0)
    qabs = obs.get("Qabs", qext - qsca)
    ssa = obs.get("SSA", qsca / qext if qext > 0.0 else 0.0)
    g = obs.get("g", 0.0)
    
    return SolverResult(
        qext=qext,
        qsca=qsca,
        qabs=qabs,
        ssa=ssa,
        g=g,
        angles_deg=data.get("theta_deg", []),
        phase_matrix=phase_matrix,
        timing={
            "geometry_s": timing.get("time_geom_s", 0.0),
            "solve_s": timing.get("time_solve_s", 0.0),
            "phase_s": timing.get("time_phase_s", 0.0),
            "total_s": timing.get("time_total_s", 0.0)
        },
        metadata={
            "demo": True,
            "solver_realization": "demo",
            "core_solver": "precomputed_reference",
            "file_source": str(p.name),
            "input_parameters": data.get("input_parameters", {})
        }
    )
