import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from apps.rcr_workbench.models import solver_realization_label
from apps.rcr_workbench.ui_state import init_state
from apps.rcr_workbench.plotting import plot_phase_matrix_plotly
from apps.rcr_workbench.demo_loader import load_demo_result

# Page configuration
st.set_page_config(page_title="Simulation Results - RCR Workbench", page_icon="R", layout="wide")
init_state()

# Notice and redirect button
st.warning("**Workspace notice**: The workbench has been unified into a single-page MATLAB-inspired Workspace dashboard. All controls are now available on the Home page.")
if st.button("Return to Main Workspace", key="redirect_results"):
    st.switch_page("app.py")

st.title("Simulation Results")
st.write("Analyze the calculated electromagnetic scattering efficiencies and phase matrix elements.")

st.divider()

res = st.session_state.current_result
case = st.session_state.current_case

if res is None:
    st.warning("No simulation results found. Please configure the geometry and run the solver on the **Solver** page.")
else:
    # 1. Top Section - Big KPI Metrics
    st.subheader("Integrated Optical Properties")
    
    k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)
    k_col1.metric("Extinction (Qext)", f"{res.qext:.6f}")
    k_col2.metric("Scattering (Qsca)", f"{res.qsca:.6f}")
    k_col3.metric("Absorption (Qabs)", f"{res.qabs:.6f}")
    k_col4.metric("Albedo (SSA)", f"{res.ssa:.6f}")
    k_col5.metric("Asymmetry (g)", f"{res.g:.6f}")
    
    st.divider()
    
    # 2. Main Section - Layout with Comparison and Plots
    col_plot, col_meta = st.columns([3, 1])
    
    with col_meta:
        st.subheader("Run Metadata")
        
        is_demo = res.metadata.get("demo", False)
        
        solver_value = res.metadata.get("solver_realization", case.solver.realization)
        st.markdown(f"""
        **Calculation Mode:** {"Demo (Precomputed)" if is_demo else "Live Simulation"}<br>
        **Geometry:** {case.geometry.kind.upper()} (N={case.geometry.N})<br>
        **Wavelength:** {case.optics.wavelength_nm} nm<br>
        **Refractive Index:** {case.optics.refractive_index_real} + {case.optics.refractive_index_imag}j<br>
        **Monomer Order (p):** {case.solver.sphere_order}<br>
        **Aggregate Order (P):** {res.metadata.get("aggregate_order", case.solver.aggregate_order)}<br>
        **Solver Realization:** {solver_realization_label(solver_value)}
        """, unsafe_allow_html=True)
        
        st.divider()
        st.subheader("Timing Breakdown")
        
        t_geom = res.timing.get("geometry_s", 0.0)
        t_solve = res.timing.get("solve_s", 0.0)
        t_phase = res.timing.get("phase_s", 0.0)
        t_total = res.timing.get("total_s", 0.0)
        
        st.markdown(f"""
        - **Geometry Setup:** `{t_geom:.3f} s`
        - **Solver Solve:** `{t_solve:.3f} s`
        - **Phase Matrix:** `{t_phase:.3f} s`
        - **Total Duration:** `{t_total:.3f} s`
        """)
        
        st.divider()
        st.subheader("Verification Overlay")
        
        # Add checkbox to load and overlay reference demo data
        overlay_ref = st.checkbox("Overlay Reference (Demo) Data", value=True)
        
        ref_pm = None
        ref_name = "Demo Reference"
        
        if overlay_ref:
            # Construct case config for looking up corresponding demo result
            ref_case = case.model_copy(deep=True)
            # Make sure we search for the demo matching this N/seed
            ref_case.solver.demo_mode = True
            ref_res = load_demo_result(ref_case)
            
            if ref_res:
                ref_pm = ref_res.phase_matrix
                st.success("Reference loaded and overlaid.")
                # Show reference efficiencies for comparison
                st.markdown(f"""
                **Reference Efficiencies:**
                - *Ref Qext*: `{ref_res.qext:.6f}`
                - *Ref Qsca*: `{ref_res.qsca:.6f}`
                - *Ref g*: `{ref_res.g:.6f}`
                """)
            else:
                st.info("No precomputed demo file matching these parameters (N/wavelength/index) was found for overlay.")
                
    with col_plot:
        st.subheader("Phase Matrix Elements")
        st.write("Shows the 6 independent elements of the 4x4 scattering matrix averaged over random orientations.")
        
        # Generate the interactive Plotly 6-panel figure
        fig = plot_phase_matrix_plotly(
            angles=res.angles_deg,
            pm_dict=res.phase_matrix,
            ref_pm_dict=ref_pm,
            ref_name=ref_name
        )
        st.plotly_chart(fig, width='stretch')
