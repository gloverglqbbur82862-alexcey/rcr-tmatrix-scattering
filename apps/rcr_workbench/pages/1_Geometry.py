import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import numpy as np

from apps.rcr_workbench.ui_state import init_state
from apps.rcr_workbench.plotting import plot_geometry_3d
from geometry.geometry import build_bcca, build_bpca, gyration_radius

# Page configuration
st.set_page_config(page_title="Geometry & Physics - RCR Workbench", page_icon="R", layout="wide")
init_state()

# Notice and redirect button
st.warning("**Workspace notice**: The workbench has been unified into a single-page MATLAB-inspired Workspace dashboard. All controls are now available on the Home page.")
if st.button("Return to Main Workspace", key="redirect_geom"):
    st.switch_page("app.py")

st.title("Geometry and Physical Parameters")
st.write("Configure the cluster morphology, monomer properties, incident light wavelength, and truncation orders.")

st.divider()

# Get active config references
case = st.session_state.current_case

# Divide page into Sidebar for inputs and Main section for outputs
col_input, col_display = st.columns([1, 2])

with col_input:
    st.subheader("Physical Setup")
    
    # 1. Geometry Type Selection
    geom_type = st.selectbox(
        "Aggregate Morphology",
        ["BCCA (Cluster-Cluster)", "BPCA (Particle-Cluster)", "Upload Coordinates"],
        index=0 if case.geometry.kind == "bcca" else (1 if case.geometry.kind == "bpca" else 2)
    )
    kind_map = {
        "BCCA (Cluster-Cluster)": "bcca",
        "BPCA (Particle-Cluster)": "bpca",
        "Upload Coordinates": "upload"
    }
    case.geometry.kind = kind_map[geom_type]

    # 2. Number of Monomers (N)
    if case.geometry.kind == "bcca":
        # BCCA requires power of 2
        powers_of_two = [2, 4, 8, 16, 32, 64, 128, 256]
        default_idx = powers_of_two.index(case.geometry.N) if case.geometry.N in powers_of_two else 3
        N = st.selectbox("Monomers (N) [Power of 2 for BCCA]", powers_of_two, index=default_idx)
    elif case.geometry.kind == "bpca":
        N = st.number_input("Monomers (N)", min_value=1, max_value=1000, value=int(case.geometry.N), step=1)
    else:
        N = int(case.geometry.N) # will be set by uploaded file
        st.info("Monomer count will be determined by the uploaded coordinate file.")

    case.geometry.N = N

    # 3. Random seed
    if case.geometry.kind != "upload":
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=int(case.geometry.seed or 42))
        case.geometry.seed = seed
    
    # 4. Monomer radius & wavelength
    r_mono = st.number_input("Monomer Radius (a) [nm]", min_value=1.0, max_value=1000.0, value=float(case.geometry.monomer_radius_nm), step=1.0)
    case.geometry.monomer_radius_nm = r_mono
    
    wl = st.number_input("Wavelength (λ) [nm]", min_value=100.0, max_value=5000.0, value=float(case.optics.wavelength_nm), step=10.0)
    case.optics.wavelength_nm = wl

    # 5. Complex Refractive Index
    st.write("##### Complex Refractive Index (m = n + ik)")
    col_n, col_k = st.columns(2)
    with col_n:
        m_real = st.number_input("n (Real part)", min_value=0.1, max_value=10.0, value=float(case.optics.refractive_index_real), step=0.01)
    with col_k:
        m_imag = st.number_input("k (Imag part)", min_value=0.0, max_value=10.0, value=float(case.optics.refractive_index_imag), step=0.01)
        
    case.optics.refractive_index_real = m_real
    case.optics.refractive_index_imag = m_imag

    st.divider()
    st.subheader("Truncation Settings")

    # 6. Monomer sphere truncation order (p)
    p = st.number_input("Monomer Truncation Order (p)", min_value=1, max_value=30, value=int(case.solver.sphere_order), step=1)
    case.solver.sphere_order = p
    
    # 7. Aggregate truncation order (P)
    agg_order_mode = st.radio("Aggregate Truncation Order (P)", ["Auto (size-parameter rule)", "Custom"], index=0 if case.solver.aggregate_order == "auto" else 1)
    if agg_order_mode == "Auto (size-parameter rule)":
        case.solver.aggregate_order = "auto"
    else:
        # custom order
        default_P = int(case.solver.aggregate_order) if isinstance(case.solver.aggregate_order, int) else p + 3
        custom_P = st.number_input("Custom Order (P)", min_value=int(p), max_value=100, value=default_P, step=1)
        case.solver.aggregate_order = custom_P

    # File uploader for custom coordinate files
    if case.geometry.kind == "upload":
        st.divider()
        st.subheader("Upload Coordinate File")
        uploaded_file = st.file_uploader("Upload CSV or XYZ (format: x, y, z lines)", type=["csv", "xyz", "txt"])
        
        if uploaded_file is not None:
            try:
                # Read coordinates
                content = uploaded_file.read().decode("utf-8")
                lines = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Replace commas with spaces and split
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 3:
                        try:
                            lines.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            pass
                
                if len(lines) > 0:
                    centers = lines
                    case.geometry.centers = centers
                    case.geometry.N = len(centers)
                    st.success(f"Loaded {len(centers)} monomer coordinates!")
                else:
                    st.error("Could not parse coordinates. Ensure file has columns for X, Y, Z.")
            except Exception as e:
                st.error(f"Error reading coordinate file: {e}")

    # Build button
    build_clicked = st.button("Apply & Generate Morphology", width='stretch', type="primary")

# Compute / build geometry
if build_clicked or st.session_state.current_case.geometry.centers is None:
    clear_existing_result = build_clicked
    if case.geometry.kind == "bcca":
        with st.spinner("Building BCCA aggregate geometry..."):
            pos_raw, _ = build_bcca(n_monomers=case.geometry.N, radius=case.geometry.monomer_radius_nm, overlap=0.0, seed=case.geometry.seed)
            case.geometry.centers = pos_raw.tolist()
            if clear_existing_result:
                st.session_state.current_result = None
    elif case.geometry.kind == "bpca":
        with st.spinner("Building BPCA aggregate geometry..."):
            pos_raw, _ = build_bpca(n_monomers=case.geometry.N, radius=case.geometry.monomer_radius_nm, overlap=0.0, seed=case.geometry.seed)
            case.geometry.centers = pos_raw.tolist()
            if clear_existing_result:
                st.session_state.current_result = None
    elif case.geometry.kind == "upload" and case.geometry.centers is None:
        st.warning("Please upload a coordinate file to display coordinates.")

# Show geometry & dimensions on the right column
with col_display:
    if case.geometry.centers is not None:
        centers = np.array(case.geometry.centers)
        
        # Display 3D Plotly plot
        fig = plot_geometry_3d(centers, case.geometry.monomer_radius_nm)
        st.plotly_chart(fig, width='stretch')
        
        # Geometry metrics
        r_gyration = gyration_radius(centers)
        k = 2.0 * np.pi / case.optics.wavelength_nm
        x_monomer = case.geometry.monomer_radius_nm * k
        x_gyration = r_gyration * k
        
        # Circumscribing sphere radius (rough estimate)
        r_max = float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1))) + case.geometry.monomer_radius_nm
        x_circ = r_max * k
        
        # Solver auto-order calculation
        auto_P = int(np.ceil(x_circ + 4.05 * (x_circ ** (1/3)) + 2)) if len(centers) > 1 else max(3, int(np.ceil(x_monomer + 4.05 * x_monomer**(1/3) + 2)))
        resolved_P = auto_P if case.solver.aggregate_order == "auto" else int(case.solver.aggregate_order)
        
        # Matrix dimension: c = 2p(p+2) per sphere. Total equations = N * c
        c_vars = 2 * case.solver.sphere_order * (case.solver.sphere_order + 2)
        total_equations = case.geometry.N * c_vars
        
        # Memory estimation: Dense system matrix is of size total_equations * total_equations.
        # complex128 takes 16 bytes.
        dense_mem_bytes = (total_equations ** 2) * 16
        dense_mem_mb = dense_mem_bytes / (1024 ** 2)
        dense_mem_gb = dense_mem_bytes / (1024 ** 3)
        
        st.subheader("Cluster Statistics & Estimated Solver Dimensions")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Radius of Gyration (Rg)", f"{r_gyration:.2f} nm", f"Size parameter: {x_gyration:.3f}")
        stat_col2.metric("Circumscribing Radius", f"{r_max:.2f} nm", f"Size parameter: {x_circ:.3f}")
        stat_col3.metric("Monomer Size Parameter (x)", f"{x_monomer:.4f}")
        
        st.markdown(f"""
        ### Solver Dimension & Memory footprint
        - **Monomer degrees of freedom (c):** `{c_vars}` equations per sphere
        - **Total degrees of freedom (D):** `{total_equations}` equations across `{case.geometry.N}` spheres
        - **Aggregate truncation order (P):** `{resolved_P}` (calculated auto-order: `{auto_P}`)
        """)
        
        if dense_mem_gb >= 1.0:
            mem_text = f"**{dense_mem_gb:.2f} GiB**"
        else:
            mem_text = f"**{dense_mem_mb:.2f} MiB**"
            
        if dense_mem_gb > 8.0:
            st.error(f"**Dense preassembled solver matrix will require {mem_text} VRAM/RAM.** Running the dense solver may cause out-of-memory errors. Use the **matrix-free** solver realization for this case.")
        elif dense_mem_gb > 2.0:
            st.warning(f"**Dense preassembled solver matrix will require {mem_text} VRAM/RAM.** Run with caution.")
        else:
            st.success(f"**Dense preassembled matrix footprint is small: {mem_text}.** This case is safe for all solver realizations.")
    else:
        st.info("Select a morphology type and click 'Apply & Generate Morphology' to show the 3D visualization.")
