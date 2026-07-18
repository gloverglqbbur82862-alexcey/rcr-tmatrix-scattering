import streamlit as st
import hashlib
import json
from apps.rcr_workbench.models import CaseConfig, GeometryConfig, OpticalConfig, SolverConfig

def init_state():
    """Initialize Streamlit session state fields if not already set."""
    if "current_case" not in st.session_state:
        from geometry.geometry import build_bcca
        try:
            pos_raw, _ = build_bcca(n_monomers=64, radius=25.0, overlap=0.0, seed=42)
            centers_val = pos_raw.tolist()
        except Exception:
            centers_val = None
            
        st.session_state.current_case = CaseConfig(
            geometry=GeometryConfig(
                kind="bcca",
                N=64,
                monomer_radius_nm=25.0,
                seed=42,
                centers=centers_val
            ),
            optics=OpticalConfig(
                wavelength_nm=550.0,
                refractive_index_real=1.95,
                refractive_index_imag=0.79
            ),
            solver=SolverConfig(
                realization="demo",
                sphere_order=3,
                aggregate_order="auto",
                demo_mode=True
            )
        )
    if "current_result" not in st.session_state:
        from apps.rcr_workbench.demo_loader import load_demo_result
        try:
            st.session_state.current_result = load_demo_result(st.session_state.current_case)
        except Exception:
            st.session_state.current_result = None
            
    if "solver_logs" not in st.session_state:
        st.session_state.solver_logs = []
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False
    
    # New Case Workspace states
    if "selected_context" not in st.session_state:
        st.session_state.selected_context = "solver"
    if "command_log" not in st.session_state:
        # Add default loaded command to log
        st.session_state.command_log = ["load_demo_result(case_id=\"BCCA_N64_550nm_seed42\")"]
    if "case_dirty" not in st.session_state:
        st.session_state.case_dirty = False
    if "run_status" not in st.session_state:
        st.session_state.run_status = "Completed"
    if "last_run_summary" not in st.session_state:
        st.session_state.last_run_summary = None

def make_case_id(case: CaseConfig) -> str:
    """Generate a readable case identifier, e.g., 'BCCA_N64_550nm_seed42'."""
    kind = case.geometry.kind.upper()
    N = case.geometry.N
    wl = int(round(case.optics.wavelength_nm))
    if kind == "UPLOAD":
        return f"UPLOAD_N{N}_{wl}nm"
    else:
        seed = case.geometry.seed or 42
        return f"{kind}_N{N}_{wl}nm_seed{seed}"

def make_case_hash(case: CaseConfig) -> str:
    """Generate a stable reproducibility hash of the CaseConfig."""
    d = case.model_dump()
    # Normalize list format if centers is present to make hash stable
    data_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()[:16]

def mark_case_dirty():
    """Mark the active case as dirty/modified."""
    st.session_state.case_dirty = True
    st.session_state.run_status = "Dirty"

def append_command_log(text: str):
    """Append a message/command to the command log."""
    if "command_log" not in st.session_state:
        st.session_state.command_log = []
    st.session_state.command_log.append(text)
