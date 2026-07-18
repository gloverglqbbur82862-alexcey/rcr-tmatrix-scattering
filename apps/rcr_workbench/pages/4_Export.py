import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import io
import matplotlib.pyplot as plt
import numpy as np

from apps.rcr_workbench.ui_state import init_state
from apps.rcr_workbench.exports import (
    export_case_yaml,
    export_results_json,
    export_global_observables_csv,
    export_phase_matrix_csv,
    generate_markdown_report
)
from apps.rcr_workbench.plotting import (
    render_geometry_matplotlib,
    render_phase_matrix_matplotlib
)
from geometry.geometry import build_bcca, build_bpca

# Page configuration
st.set_page_config(page_title="Export Center - RCR Workbench", page_icon="R", layout="wide")
init_state()

# Notice and redirect button
st.warning("**Workspace notice**: The workbench has been unified into a single-page MATLAB-inspired Workspace dashboard. All controls are now available on the Home page.")
if st.button("Return to Main Workspace", key="redirect_export"):
    st.switch_page("app.py")

st.title("Data Export Center")
st.write("Download configuration files, numerical results tables, publication-grade figures, and summary reports.")

st.divider()

res = st.session_state.current_result
case = st.session_state.current_case


def resolve_geometry_centers(active_case):
    """Return centers for exported geometry figures, rebuilding deterministic cases when needed."""
    if active_case.geometry.centers is not None:
        return np.array(active_case.geometry.centers)
    kind = active_case.geometry.kind.lower()
    if kind == "bcca":
        centers, _ = build_bcca(
            n_monomers=active_case.geometry.N,
            radius=active_case.geometry.monomer_radius_nm,
            overlap=0.0,
            seed=active_case.geometry.seed,
        )
        return centers
    if kind == "bpca":
        centers, _ = build_bpca(
            n_monomers=active_case.geometry.N,
            radius=active_case.geometry.monomer_radius_nm,
            overlap=0.0,
            seed=active_case.geometry.seed,
        )
        return centers
    return None

if res is None:
    st.warning("No active calculation results found. Please configure the geometry and run the solver on the **Solver** page.")
else:
    # Split page into layout columns
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Configs & Data Files")
        st.write("Download standard scientific configuration and data files.")
        
        # 1. Case YAML Config
        yaml_str = export_case_yaml(case)
        st.download_button(
            label="Download case.yaml (Configuration)",
            data=yaml_str,
            file_name=f"case_{case.geometry.kind}_n{case.geometry.N}.yaml",
            mime="text/yaml",
            width='stretch'
        )
        
        # 2. Results JSON
        json_str = export_results_json(res)
        st.download_button(
            label="Download results.json (Full Data Dump)",
            data=json_str,
            file_name=f"results_{case.geometry.kind}_n{case.geometry.N}.json",
            mime="application/json",
            width='stretch'
        )
        
        # 3. Global Observables CSV
        observables_csv = export_global_observables_csv(case, res)
        st.download_button(
            label="Download global_observables.csv",
            data=observables_csv,
            file_name=f"global_observables_{case.geometry.kind}_n{case.geometry.N}.csv",
            mime="text/csv",
            width='stretch'
        )
        
        # 4. Phase Matrix Elements CSV
        phase_csv = export_phase_matrix_csv(res)
        st.download_button(
            label="Download phase_matrix.csv (Theta vs Fij/F11)",
            data=phase_csv,
            file_name=f"phase_matrix_{case.geometry.kind}_n{case.geometry.N}.csv",
            mime="text/csv",
            width='stretch'
        )
        
        # 5. Markdown Report
        md_report = generate_markdown_report(case, res)
        st.download_button(
            label="Download summary_report.md (Markdown)",
            data=md_report,
            file_name=f"report_{case.geometry.kind}_n{case.geometry.N}.md",
            mime="text/markdown",
            width='stretch'
        )
        
    with col_right:
        st.subheader("Publication-Grade Figures")
        st.write("Generate and download high-resolution vector and raster figures.")
        
        with st.spinner("Preparing Matplotlib figures (300 DPI)..."):
            # A. Phase Matrix Plots
            # check if reference data can be overlaid for Matplotlib as well
            overlay_ref = st.checkbox("Overlay Reference (Demo) Data in Exported Plots", value=True)
            ref_pm = None
            if overlay_ref:
                from apps.rcr_workbench.demo_loader import load_demo_result
                ref_case = case.model_copy(deep=True)
                ref_case.solver.demo_mode = True
                ref_res = load_demo_result(ref_case)
                if ref_res:
                    ref_pm = ref_res.phase_matrix
            
            fig_pm = render_phase_matrix_matplotlib(
                angles=res.angles_deg,
                pm_dict=res.phase_matrix,
                ref_pm_dict=ref_pm,
                ref_name="Reference"
            )
            
            # Save to buffers
            buf_pm_png = io.BytesIO()
            fig_pm.savefig(buf_pm_png, format="png", dpi=300, bbox_inches="tight")
            buf_pm_png.seek(0)
            
            buf_pm_pdf = io.BytesIO()
            fig_pm.savefig(buf_pm_pdf, format="pdf", bbox_inches="tight")
            buf_pm_pdf.seek(0)
            plt.close(fig_pm)
            
            # B. Geometry Plots
            centers_for_export = resolve_geometry_centers(case)
            if centers_for_export is not None:
                fig_geom = render_geometry_matplotlib(
                    centers_for_export,
                    case.geometry.monomer_radius_nm
                )
                
                buf_geom_png = io.BytesIO()
                fig_geom.savefig(buf_geom_png, format="png", dpi=300, bbox_inches="tight")
                buf_geom_png.seek(0)
                
                buf_geom_pdf = io.BytesIO()
                fig_geom.savefig(buf_geom_pdf, format="pdf", bbox_inches="tight")
                buf_geom_pdf.seek(0)
                plt.close(fig_geom)
            else:
                buf_geom_png = None
                buf_geom_pdf = None
            
        st.write("##### Phase Matrix Plot")
        st.download_button(
            label="Download phase_matrix.png (300 DPI)",
            data=buf_pm_png,
            file_name=f"phase_matrix_{case.geometry.kind}_n{case.geometry.N}.png",
            mime="image/png",
            width='stretch'
        )
        st.download_button(
            label="Download phase_matrix.pdf (Vector PDF)",
            data=buf_pm_pdf,
            file_name=f"phase_matrix_{case.geometry.kind}_n{case.geometry.N}.pdf",
            mime="application/pdf",
            width='stretch'
        )
        
        st.write("##### Aggregate Geometry Plot")
        if buf_geom_png is None or buf_geom_pdf is None:
            st.info("Geometry coordinates are not available for this case, so geometry figure export is skipped.")
        else:
            st.download_button(
                label="Download aggregate_geometry.png (300 DPI)",
                data=buf_geom_png,
                file_name=f"geometry_{case.geometry.kind}_n{case.geometry.N}.png",
                mime="image/png",
                width='stretch'
            )
            st.download_button(
                label="Download aggregate_geometry.pdf (Vector PDF)",
                data=buf_geom_pdf,
                file_name=f"geometry_{case.geometry.kind}_n{case.geometry.N}.pdf",
                mime="application/pdf",
                width='stretch'
            )
        
    st.divider()
    st.info("CSV and JSON formats are structured for standard programming languages (Python, MATLAB, R). Matplotlib plots are rendered with publication-grade sizing and borders.")
