import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import time

from apps.rcr_workbench.ui_state import init_state
from apps.rcr_workbench.runner import run_case

# Page configuration
st.set_page_config(page_title="Solver Execution - RCR Workbench", page_icon="R", layout="wide")
init_state()

# Notice and redirect button
st.warning("**Workspace notice**: The workbench has been unified into a single-page MATLAB-inspired Workspace dashboard. All controls are now available on the Home page.")
if st.button("Return to Main Workspace", key="redirect_solver"):
    st.switch_page("app.py")

st.title("Solver Execution")
st.write("Select a solver realization, verify the pre-run conditions, and run the calculation.")

st.divider()

case = st.session_state.current_case

# Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Solver Configuration")
    
    # Check if GPU solver is available
    cuda_available = False
    try:
        import cupy as cp
        cuda_available = cp.is_available()
    except Exception:
        pass
        
    options = ["CPU-Parallel Direct", "CPU-Serial Direct", "Demo Mode (Precomputed Result)"]
    realization_map = {
        "CPU-Parallel Direct": "cpu_parallel",
        "CPU-Serial Direct": "cpu_serial",
        "Demo Mode (Precomputed Result)": "demo"
    }
    
    if cuda_available:
        options.insert(2, "GPU Matrix-Free Direct")
        options.insert(3, "GPU Dense Preassembled")
        realization_map["GPU Matrix-Free Direct"] = "gpu_matrix_free"
        realization_map["GPU Dense Preassembled"] = "gpu_dense"
    else:
        st.info("CUDA/CuPy was not detected. GPU solvers are disabled. To inspect larger cases without GPU hardware, enable **Demo Mode**.")
        
    # Set default index
    current_real = case.solver.realization
    if current_real == "demo" or case.solver.demo_mode:
        default_idx = len(options) - 1
    elif current_real == "cpu_serial":
        default_idx = 1
    elif current_real == "gpu_matrix_free" and cuda_available:
        default_idx = 2
    elif current_real == "gpu_dense" and cuda_available:
        default_idx = 3
    else:
        default_idx = 0 # cpu_parallel
        
    selected_option = st.selectbox(
        "Solver Realization",
        options,
        index=default_idx
    )
    
    real = realization_map[selected_option]
    case.solver.realization = real
    case.solver.demo_mode = (real == "demo")
    
    # Solver Descriptions
    st.write("##### Solver Realization Guide")
    if real == "cpu_parallel":
        st.markdown("""
        **CPU-Parallel Direct**: Uses multi-core CPU execution for JIT-compiled RCR operations.
        - *Best for*: Laptops and workstations without CUDA.
        - *Scaling*: Good scaling up to ~64 spheres, slower for massive clusters.
        """)
    elif real == "cpu_serial":
        st.markdown("""
        **CPU-Serial Direct**: Runs the solver on a single CPU thread.
        - *Best for*: Debugging and benchmarking numba overheads.
        """)
    elif real == "gpu_matrix_free":
        st.markdown("""
        **GPU Matrix-Free Direct**: Low-memory GPU realization using CuPy. It applies the RCR operator without assembling the full dense matrix.
        - *Best for*: Very large aggregates (N > 128) where dense matrix memory is prohibitive.
        """)
    elif real == "gpu_dense":
        st.markdown("""
        **GPU Dense Preassembled**: Builds the complete system matrix in GPU memory and runs GPU-resident Krylov iterations.
        - *Best for*: Rapid calculations for small to mid-sized aggregates.
        - *Warning*: Out-of-memory errors for D > 10,000.
        """)
    elif real == "demo":
        st.markdown("""
        **Demo Mode (Precomputed Result)**: Loads verified reference results without running a live solve.
        - *Best for*: Instantly loading paper verification cases without waiting for calculations.
        - *Availability*: Available for N=8, 16, 32, 64 aggregates at wavelength=550nm, refractive index=1.95+0.79j.
        """)

with col2:
    st.subheader("Pre-run Verification")
    
    ready = True
    
    # check geometry
    if case.geometry.centers is None:
        st.error("**Geometry has not been generated.** Please go to the **Geometry** page and build it first.")
        ready = False
    else:
        st.success(f"Geometry generated ({case.geometry.N} monomers).")
        
    # check params
    if case.optics.wavelength_nm <= 0.0 or case.geometry.monomer_radius_nm <= 0.0:
        st.error("**Physical parameters are invalid.**")
        ready = False
    else:
        st.success("Physical parameters validated.")
        
    # check memory footprint for dense GPU
    if real == "gpu_dense" and case.geometry.centers is not None:
        c_vars = 2 * case.solver.sphere_order * (case.solver.sphere_order + 2)
        total_equations = case.geometry.N * c_vars
        dense_mem_bytes = (total_equations ** 2) * 16
        dense_mem_gb = dense_mem_bytes / (1024 ** 3)
        
        if dense_mem_gb > 6.0:
            st.error(f"**Dense GPU matrix footprint is {dense_mem_gb:.2f} GiB.** Exceeds standard GPU limits. Please use Matrix-Free realization.")
            ready = False
        else:
            st.warning(f"Dense GPU matrix footprint: {dense_mem_gb:.2f} GiB.")

    # Warnings for large cases on CPU
    if real in ("cpu_parallel", "cpu_serial") and case.geometry.centers is not None:
        N = case.geometry.N
        p = case.solver.sphere_order
        if N > 64 or p > 7:
            st.warning("**Large configuration on CPU.** This solve might take several minutes. Consider reducing monomers, order, or using Demo Mode.")
            
    # Run button
    st.write("")
    if ready:
        run_btn = st.button("Execute Scattering Simulation", width='stretch', type="primary")
    else:
        st.button("Execute Scattering Simulation (Disabled)", width='stretch', disabled=True)
        run_btn = False

if run_btn and ready:
    # Set up progress/logs callback
    progress_bar = st.progress(0, text="Initializing runner...")
    log_area = st.empty()
    
    log_list = []
    def progress_cb(msg):
        log_list.append(msg)
        # Display the last few logs
        log_area.code("\n".join(log_list[-6:]))
        
    try:
        t0 = time.time()
        progress_bar.progress(10, text="Running solver...")
        
        res = run_case(case, progress_callback=progress_cb)
        
        progress_bar.progress(100, text="Calculation complete!")
        st.session_state.current_result = res
        
        st.success(f"**Simulation successfully completed in {time.time() - t0:.2f} seconds.** Go to the **Results** page to view outputs.")
        
        # Display short summary
        st.markdown(f"""
        ### Quick Metrics
        - **Qext:** `{res.qext:.6f}`
        - **Qsca:** `{res.qsca:.6f}`
        - **SSA:** `{res.ssa:.6f}`
        - **Asymmetry (g):** `{res.g:.6f}`
        """)
    except FileNotFoundError as fnf:
        st.error(f"Demo File Error: {fnf}")
        st.info("Please match the case settings (N=8/16/32/64, wavelength=550nm, refractive index=1.95+0.79j) or disable Demo Mode.")
    except Exception as e:
        st.error(f"Execution failed: {e}")
        import traceback
        st.code(traceback.format_exc())
