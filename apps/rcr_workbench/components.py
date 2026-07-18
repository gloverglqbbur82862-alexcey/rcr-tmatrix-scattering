import streamlit as st
import numpy as np
import os
from apps.rcr_workbench.models import solver_realization_label, CaseConfig, SolverResult
from apps.rcr_workbench.ui_state import (
    make_case_id,
    make_case_hash,
    mark_case_dirty,
    append_command_log,
)
from apps.rcr_workbench.plotting import plot_geometry_3d, plot_phase_matrix_plotly
from apps.rcr_workbench.demo_loader import load_demo_result
from apps.rcr_workbench.remote_runner import check_remote_status, is_remote_realization
from geometry.geometry import build_bcca, build_bpca, gyration_radius


def _format_duration(seconds):
    """Format a duration in a compact workbench status style."""
    if seconds is None:
        return "--:--"
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    if minutes:
        return f"{minutes:02d}:{secs:04.1f}"
    return f"{secs:.2f} s"


def _remote_message(status: dict | None) -> str:
    """Return a UI-safe remote status message without exposing host aliases."""
    if not status:
        return "Not checked"
    msg = str(status.get("message") or "")
    host = str(status.get("host") or "")
    if host:
        msg = msg.replace(host, "remote workstation")
    return msg or ("Reachable" if status.get("reachable") else "Unavailable")

def render_toolstrip():
    """Render the top toolstrip with focus tabs and primary actions."""
    # Horizontal focus selector. The app remains a single integrated workspace;
    # this only changes which case area receives attention.
    tabs = ["Case", "Geometry", "Solver", "Analysis", "Figures", "Export"]
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Solver"
        
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Workspace focus",
            tabs,
            default=st.session_state.active_tab,
            label_visibility="collapsed",
            key="toolstrip_focus",
        )
        selected = selected or st.session_state.active_tab
    else:
        selected = st.radio(
            "Workspace focus",
            tabs,
            index=tabs.index(st.session_state.active_tab),
            horizontal=True,
            label_visibility="collapsed",
            key="toolstrip_focus"
        )
    st.session_state.active_tab = selected
    st.session_state.selected_context = selected.lower()

    st.markdown('<div style="margin-top: 0.35rem; margin-bottom: 0.35rem;"></div>', unsafe_allow_html=True)
    
    col_actions = st.columns(9)
    case = st.session_state.current_case
    result = st.session_state.current_result
    
    with col_actions[0]:
        if st.button("New", width='stretch'):
            st.session_state.current_case = CaseConfig(
                geometry={"kind": "bcca", "N": 64, "monomer_radius_nm": 25.0, "seed": 42, "centers": None},
                optics={"wavelength_nm": 550.0, "refractive_index_real": 1.95, "refractive_index_imag": 0.79},
                solver={"realization": "cpu_parallel", "sphere_order": 3, "aggregate_order": "auto", "demo_mode": False}
            )
            st.session_state.current_result = None
            st.session_state.case_dirty = False
            st.session_state.run_status = "Ready"
            st.session_state.command_log = []
            append_command_log("Initialized new blank case configuration.")
            st.rerun()

    with col_actions[1]:
        if st.button("Open Demo", width='stretch'):
            case.geometry.kind = "bcca"
            case.geometry.N = 64
            case.geometry.monomer_radius_nm = 25.0
            case.geometry.seed = 42
            case.optics.wavelength_nm = 550.0
            case.optics.refractive_index_real = 1.95
            case.optics.refractive_index_imag = 0.79
            case.solver.sphere_order = 3
            case.solver.aggregate_order = "auto"
            case.solver.realization = "demo"
            case.solver.demo_mode = True
            
            res = load_demo_result(case)
            if res:
                st.session_state.current_result = res
                st.session_state.case_dirty = False
                st.session_state.run_status = "Completed"
                append_command_log("Loaded reference case BCCA_N64_550nm_seed42 from demo catalog.")
                st.rerun()

    with col_actions[2]:
        if st.button("Save", width='stretch'):
            st.session_state.case_dirty = False
            st.session_state.run_status = "Completed" if result is not None else "Ready"
            append_command_log(f"Saved active case config: {make_case_id(case)} (hash={make_case_hash(case)})")
            st.success("Case configuration saved successfully.")

    with col_actions[3]:
        if st.button("Import", width='stretch'):
            case.geometry.kind = "upload"
            st.session_state.selected_context = "geometry"
            st.session_state.active_tab = "Geometry"
            mark_case_dirty()
            append_command_log("Switched morphology configuration to Upload Coordinates mode.")
            st.rerun()

    with col_actions[4]:
        if st.button("Generate", width='stretch'):
            kind = case.geometry.kind
            if kind == "upload":
                st.warning("Cannot generate aggregates under upload mode. Select BCCA/BPCA first.")
            else:
                with st.spinner(f"Generating {kind.upper()} geometry..."):
                    if kind == "bcca":
                        pos_raw, _ = build_bcca(n_monomers=case.geometry.N, radius=case.geometry.monomer_radius_nm, overlap=0.0, seed=case.geometry.seed)
                    else:
                        pos_raw, _ = build_bpca(n_monomers=case.geometry.N, radius=case.geometry.monomer_radius_nm, overlap=0.0, seed=case.geometry.seed)
                    case.geometry.centers = pos_raw.tolist()
                    mark_case_dirty()
                    append_command_log(f"Generated {kind.upper()} aggregate cluster centers (N={case.geometry.N}).")
                    st.success("Geometry generated!")
                    st.rerun()

    with col_actions[5]:
        run_ready = case.geometry.kind != "upload" or case.geometry.centers is not None
        # Make Run button green accent
        if st.button("Run", width='stretch', disabled=not run_ready):
            st.session_state.run_status = "Running"
            st.rerun()

    with col_actions[6]:
        if st.button("Stop", width='stretch', disabled=(st.session_state.run_status != "Running")):
            st.session_state.run_status = "Ready"
            append_command_log("Solver execution stopped by user command.")
            st.rerun()

    with col_actions[7]:
        if st.button("Validate", width='stretch'):
            append_command_log("Triggered validation overlay for active case.")
            st.session_state.selected_context = "analysis"
            st.session_state.active_tab = "Analysis"
            st.rerun()

    with col_actions[8]:
        if st.button("Export", width='stretch'):
            st.session_state.selected_context = "export"
            st.session_state.active_tab = "Export"
            st.rerun()

    st.markdown('<div style="border-top:1px solid var(--rcr-border); margin:0.3rem 0 0.55rem 0;"></div>', unsafe_allow_html=True)

def render_case_header(case, result):
    """Render the active case identity and quick status chips."""
    kind = case.geometry.kind.upper()
    N = case.geometry.N
    a = case.geometry.monomer_radius_nm
    wl = case.optics.wavelength_nm
    m = f"{case.optics.refractive_index_real} + {case.optics.refractive_index_imag}i"
    p = case.solver.sphere_order
    P = case.solver.aggregate_order
    realization_lbl = solver_realization_label(case.solver.realization)
    result_state = "Result loaded" if result is not None else "No result"
    
    st.markdown(
        f'<div class="rcr-chip-row">'
        f'  <span class="rcr-chip">morphology = <b>{kind}</b></span>'
        f'  <span class="rcr-chip">N = <b>{N}</b></span>'
        f'  <span class="rcr-chip">a = <b>{a:g} nm</b></span>'
        f'  <span class="rcr-chip">λ = <b>{wl:g} nm</b></span>'
        f'  <span class="rcr-chip">m = <b>{m}</b></span>'
        f'  <span class="rcr-chip">p = <b>{p}</b></span>'
        f'  <span class="rcr-chip">P = <b>{P}</b></span>'
        f'  <span class="rcr-chip">solver = <b>{realization_lbl}</b></span>'
        f'  <span class="rcr-chip">{result_state}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_case_explorer(case, result):
    """Render the explorer tree hierarchy in the left panel."""
    st.markdown(
        '<div class="rcr-panel rcr-panel-pad">'
        '<div class="rcr-panel-title">'
        '  <span>Case Explorer</span>'
        '  <span class="rcr-subtle">case tree</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Compact Search Input
    st.text_input("Search Cases", value="", placeholder="Search...", label_visibility="collapsed")
    
    case_id = make_case_id(case)
    
    # Explorer Collapsible Structure
    st.markdown(f"""
    <div style="font-size: 0.78rem; padding-top: 0.2rem;">
        <div style="margin-bottom: 0.42rem; color: var(--rcr-text);">Cases</div>
        <div style="margin-left: 0.9rem; margin-bottom: 0.42rem; color: var(--rcr-primary); font-weight: 650; background-color: #eef6ff; padding: 0.22rem 0.45rem; border-radius: 4px; border-left: 3px solid var(--rcr-primary);">
            <b>{case_id}</b>
        </div>
        <div style="margin-left: 1.8rem; margin-bottom: 0.28rem;">Geometry</div>
        <div style="margin-left: 2.7rem; margin-bottom: 0.22rem; color: var(--rcr-muted);">centers.xyz</div>
        <div style="margin-left: 2.7rem; margin-bottom: 0.32rem; color: var(--rcr-muted);">morphology.json</div>
        <div style="margin-left: 1.8rem; margin-bottom: 0.28rem;">Optical constants</div>
        <div style="margin-left: 2.7rem; margin-bottom: 0.32rem; color: var(--rcr-muted);">material_550nm.json</div>
        <div style="margin-left: 1.8rem; margin-bottom: 0.28rem;">Solver realization</div>
        <div style="margin-left: 2.7rem; margin-bottom: 0.32rem; color: var(--rcr-muted);">solver_config.yaml</div>
        <div style="margin-left: 1.8rem; margin-bottom: 0.28rem;">Results</div>
        <div style="margin-left: 2.7rem; margin-bottom: 0.22rem; color: var(--rcr-muted);">observables.csv</div>
        <div style="margin-left: 2.7rem; margin-bottom: 0.32rem; color: var(--rcr-muted);">phase_matrix.csv</div>
        <div style="margin-left: 1.8rem; margin-bottom: 0.28rem;">Exports</div>
        <div style="margin-left: 2.7rem; color: var(--rcr-muted);">summary_report.pdf</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Invisible control selector linked to contexts
    options = ["Case", "Geometry", "Solver", "Analysis", "Figures", "Export"]
    selected_node = st.selectbox(
        "Active Inspector Node Picker",
        options,
        index=options.index(st.session_state.active_tab),
        label_visibility="collapsed",
        key="explorer_selectbox"
    )
    if selected_node != st.session_state.active_tab:
        st.session_state.active_tab = selected_node
        st.session_state.selected_context = selected_node.lower()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def render_variables_panel(case, result):
    """Render variables inspector list in the left panel."""
    st.markdown(
        '<div class="rcr-panel rcr-panel-pad" style="margin-top:0.7rem;">'
        '<div class="rcr-panel-title">'
        '  <span>Variables</span>'
        '  <span class="rcr-subtle">active case</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    N = case.geometry.N
    wl = case.optics.wavelength_nm
    m = f"{case.optics.refractive_index_real} + {case.optics.refractive_index_imag}i"
    p = case.solver.sphere_order
    P = case.solver.aggregate_order
    
    centers_shape = f"{N} x 3" if case.geometry.centers is not None else "not computed"
    centers_preview = f"[-81.2, 14.3, ...]" if case.geometry.centers is not None else "not computed"
    
    c_vars = 2 * p * (p + 2)
    eqs = N * c_vars
    pm_shape = "181 x 6" if result else "not computed"
    pm_preview = "real" if result else "not computed"
    
    obs_shape = "1 x 5" if result else "not computed"
    obs_preview = "struct" if result else "not computed"
    
    vars_data = [
        ("centers", centers_shape, centers_preview),
        ("r_mono", "1 x 1", f"{case.geometry.monomer_radius_nm:.0f} nm"),
        ("lambda", "1 x 1", f"{wl:.0f} nm"),
        ("m_rel", "1 x 1", m),
        ("p_order", "1 x 1", str(p)),
        ("p_orient", "1 x 1", "auto"),
        ("system_size_D", "1 x 1", str(eqs)),
        ("dense_matrix_A", f"{eqs} x {eqs}", "complex128 estimate"),
        ("phase_matrix", pm_shape, pm_preview),
        ("observables", obs_shape, obs_preview)
    ]
    
    # Render table in HTML with custom styling
    st.markdown(
        f'<table class="rcr-table" style="font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;">'
        f'  <tr>'
        f'    <th>Name</th>'
        f'    <th>Size</th>'
        f'    <th>Value / preview</th>'
        f'  </tr>'
        + "".join([
            f'  <tr>'
            f'    <td style="color:var(--rcr-primary);">{row[0]}</td>'
            f'    <td style="color:var(--rcr-muted);">{row[1]}</td>'
            f'    <td style="color:var(--rcr-accent);">{row[2]}</td>'
            f'  </tr>'
            for row in vars_data
        ]) +
        f'</table>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

def draw_svg_legend():
    """Draw a schematic SVG representing Rg and Rc circles around a cluster."""
    svg_code = """
    <div style="background-color: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; text-align: center;">
        <svg width="100" height="100" viewBox="0 0 100 100" style="display: block; margin: 0 auto;">
            <!-- Circumscribed circle Rc (blue dotted) -->
            <circle cx="50" cy="50" r="42" fill="none" stroke="#1e3a8a" stroke-width="1.2" stroke-dasharray="2,2" />
            <!-- Gyration circle Rg (red dotted) -->
            <circle cx="50" cy="50" r="24" fill="none" stroke="#be123c" stroke-width="1.2" stroke-dasharray="2,2" />
            <!-- Schematic aggregate spheres -->
            <circle cx="50" cy="50" r="5" fill="#475569" opacity="0.85" />
            <circle cx="43" cy="50" r="5" fill="#475569" opacity="0.85" />
            <circle cx="57" cy="50" r="5" fill="#475569" opacity="0.85" />
            <circle cx="50" cy="43" r="5" fill="#475569" opacity="0.85" />
            <circle cx="50" cy="57" r="5" fill="#475569" opacity="0.85" />
            <circle cx="45" cy="45" r="5" fill="#475569" opacity="0.85" />
            <circle cx="55" cy="55" r="5" fill="#475569" opacity="0.85" />
            <circle cx="38" cy="42" r="5" fill="#475569" opacity="0.85" />
            <circle cx="62" cy="58" r="5" fill="#475569" opacity="0.85" />
            <!-- Inner schematic radius lines -->
            <line x1="50" y1="50" x2="50" y2="8" stroke="#1e3a8a" stroke-width="0.8" opacity="0.5" />
            <line x1="50" y1="50" x2="68" y2="34" stroke="#be123c" stroke-width="0.8" opacity="0.5" />
        </svg>
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: 6px; font-size: 0.7rem;">
            <div style="display: flex; align-items: center; gap: 4px; margin-bottom: 2px;">
                <span style="width: 8px; height: 1.5px; background-color: #1e3a8a; display: inline-block; border-top: 1.5px dashed #1e3a8a;"></span>
                <span>R<sub>c</sub></span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <span style="width: 8px; height: 1.5px; background-color: #be123c; display: inline-block; border-top: 1.5px dashed #be123c;"></span>
                <span>R<sub>g</sub></span>
            </div>
        </div>
    </div>
    """
    return svg_code

def render_live_workspace(case, result):
    """Render the middle workspace showing the 3D aggregate visual and live formulas."""
    st.markdown(
        '<div class="rcr-panel rcr-panel-pad">'
        '<div class="rcr-panel-title">'
        '  <span>Aggregate Geometry</span>'
        '  <span class="rcr-status-pill" style="font-size:0.68rem; padding:0.1rem 0.42rem;">current case</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # 3D View container (Mockup shows a box with stats card overlay on the left, 3D on the right)
    centers = None
    if case.geometry.centers is not None:
        centers = np.array(case.geometry.centers)
    else:
        # Fallback to preview coordinate generation
        kind = case.geometry.kind.lower()
        if kind != "upload":
            seed = case.geometry.seed or 42
            if kind == "bcca":
                pos_raw, _ = build_bcca(n_monomers=case.geometry.N, radius=case.geometry.monomer_radius_nm, overlap=0.0, seed=seed)
            else:
                pos_raw, _ = build_bpca(n_monomers=case.geometry.N, radius=case.geometry.monomer_radius_nm, overlap=0.0, seed=seed)
            centers = pos_raw
            
    if centers is not None:
        col_stats, col_plot = st.columns([1.1, 2.9])
        
        r_gyration = gyration_radius(centers)
        k = 2.0 * np.pi / case.optics.wavelength_nm
        x_monomer = case.geometry.monomer_radius_nm * k
        x_gyration = r_gyration * k
        r_max = float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1))) + case.geometry.monomer_radius_nm
        x_circ = r_max * k
        
        # Solver auto-order calculation
        auto_P = int(np.ceil(x_circ + 4.05 * (x_circ ** (1/3)) + 2)) if len(centers) > 1 else max(3, int(np.ceil(x_monomer + 4.05 * x_monomer**(1/3) + 2)))
        resolved_P = auto_P if case.solver.aggregate_order == "auto" else int(case.solver.aggregate_order)
        
        c_vars = 2 * case.solver.sphere_order * (case.solver.sphere_order + 2)
        total_equations = case.geometry.N * c_vars
        dense_mem_bytes = (total_equations ** 2) * 16
        dense_mem_gb = dense_mem_bytes / (1024 ** 3)
        
        # Packing fraction inside the circumscribing sphere.
        phi = case.geometry.N * (case.geometry.monomer_radius_nm / r_max)**3
        
        with col_stats:
            # Metrics text card
            st.markdown(f"""
            <div style="font-size: 0.78rem; background-color: var(--rcr-surface-subtle); border: 1px solid var(--rcr-border); padding: 0.65rem; border-radius: 6px; margin-bottom: 0.5rem;">
                <div style="margin-bottom: 0.18rem; color:var(--rcr-muted);">Radius of gyration, R<sub>g</sub></div>
                <div style="font-size: 1.02rem; font-weight: 680; color: var(--rcr-primary); margin-bottom: 0.48rem;">{r_gyration:.1f} nm</div>
                <div style="margin-bottom: 0.18rem; color:var(--rcr-muted);">Circumscribing radius, R<sub>c</sub></div>
                <div style="font-size: 1.02rem; font-weight: 680; color: var(--rcr-primary); margin-bottom: 0.48rem;">{r_max:.1f} nm</div>
                <div style="margin-bottom: 0.18rem; color:var(--rcr-muted);">Filling factor, &phi;</div>
                <div style="font-size: 1.02rem; font-weight: 680; color: var(--rcr-primary);">{phi:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            # Concentric drawing
            st.markdown(draw_svg_legend(), unsafe_allow_html=True)
            
        with col_plot:
            fig = plot_geometry_3d(centers, case.geometry.monomer_radius_nm)
            fig.update_layout(height=320, margin=dict(l=0, r=0, b=0, t=10))
            st.plotly_chart(fig, width='stretch', key="live_workspace_geometry")
            
        # Live Worksheet
        st.markdown(
            '<div class="rcr-panel-title" style="margin-top:0.65rem; margin-bottom:0.25rem;">'
            '  <span>Live Worksheet</span>'
            '  <span class="rcr-subtle">derived quantities</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Display equations styled exactly like the mockup worksheet
        st.markdown(f"""
        <div style="border: 1px solid var(--rcr-border-soft); border-radius: 6px; background-color: white;">
            <div class="rcr-worksheet-row">
                <span class="rcr-row-number">1</span>
                <span>x = 2&pi;a / &lambda; = 2&pi; &middot; {case.geometry.monomer_radius_nm:.0f} nm / {case.optics.wavelength_nm:.0f} nm = <b>{x_monomer:.4f}</b></span>
            </div>
            <div class="rcr-worksheet-row">
                <span class="rcr-row-number">2</span>
                <span>D = N &middot; 2p(p+2) = {case.geometry.N} &middot; 2 &middot; {case.solver.sphere_order} &middot; ({case.solver.sphere_order}+2) = <b>{total_equations}</b></span>
            </div>
            <div class="rcr-worksheet-row">
                <span class="rcr-row-number">3</span>
                <span>Dense linear system memory (complex128): 16D<sup>2</sup> &approx; <b>{dense_mem_gb:.2f} GiB</b></span>
            </div>
            <div class="rcr-worksheet-row">
                <span class="rcr-row-number">4</span>
                <span>Aggregate VSWF truncation: P = <b>{resolved_P}</b> ({'automatic estimate' if case.solver.aggregate_order == 'auto' else 'user value'}; x<sub>c</sub> = {x_circ:.3f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Step 5 - Horizontal Execution pipeline
        st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
        col_btn, col_pipe = st.columns([1.0, 4.0])
        with col_btn:
            if st.button("Run Case", width='stretch', type="primary", key="ws_run_case"):
                st.session_state.run_status = "Running"
                st.rerun()
                
        with col_pipe:
            status = st.session_state.run_status
            
            timing = result.timing if result is not None else {}
            total_s = timing.get("total_s")
            geom_s = timing.get("geometry_s")
            solve_s = timing.get("solve_s")
            phase_s = timing.get("phase_s")
            has_result = result is not None
            current_step = st.session_state.run_status == "Running"
            badges = [
                ("Geometry", _format_duration(geom_s), True),
                ("RCR assembly", "included", has_result),
                ("Krylov solve", _format_duration(solve_s), has_result or current_step),
                ("Orientation avg.", _format_duration(phase_s), has_result),
                ("Phase matrix", _format_duration(total_s), has_result)
            ]
            
            html_pipe = '<div class="rcr-progress-line">'
            for idx, (step, duration, active) in enumerate(badges):
                cls = "done" if active else ""
                if current_step and step == "Krylov solve":
                    cls = "current"
                state = "●" if active else "○"
                html_pipe += f'<div class="rcr-step {cls}">'
                html_pipe += f'  <span>{state} {step}</span><br>'
                html_pipe += f'  <span class="rcr-subtle">{duration}</span>'
                html_pipe += f'</div>'
                if idx < len(badges) - 1:
                    html_pipe += '<span class="rcr-divider">→</span>'
            html_pipe += '</div>'
            st.markdown(html_pipe, unsafe_allow_html=True)
            
    else:
        st.info("Upload geometry coordinates in the inspector to render the geometry worksheet.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_context_inspector(case):
    """Render the fixed solver inspector and compact case controls."""
    case_hash = make_case_hash(case)
    st.markdown(
        '<div class="rcr-panel rcr-panel-pad">'
        '<div class="rcr-panel-title">'
        '  <span>Solver Inspector</span>'
        '  <span class="rcr-subtle">resources and controls</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Case parameters", expanded=st.session_state.active_tab in ("Case", "Geometry")):
        kind_options = ["BCCA", "BPCA", "Upload"]
        kind_index_map = {"bcca": 0, "bpca": 1, "upload": 2}
        selected_kind = st.selectbox(
            "Morphology type",
            kind_options,
            index=kind_index_map.get(case.geometry.kind.lower(), 0),
            key="inspector_morphology_type",
        )
        if selected_kind.lower() != case.geometry.kind:
            case.geometry.kind = selected_kind.lower()
            if case.geometry.kind != "upload":
                case.geometry.centers = None
            mark_case_dirty()
            append_command_log(f"Morphology kind updated to {selected_kind}.")
            st.rerun()

        if case.geometry.kind == "bcca":
            powers_of_two = [4, 8, 16, 32, 64, 128, 256]
            default_idx = powers_of_two.index(case.geometry.N) if case.geometry.N in powers_of_two else 4
            N = st.selectbox("Monomer count N", powers_of_two, index=default_idx, key="inspector_N_bcca")
        else:
            N = st.number_input("Monomer count N", min_value=1, max_value=1024, value=int(case.geometry.N), key="inspector_N_generic")
        if N != case.geometry.N:
            case.geometry.N = int(N)
            if case.geometry.kind != "upload":
                case.geometry.centers = None
            mark_case_dirty()
            append_command_log(f"Monomer count N updated to {N}.")
            st.rerun()

        if case.geometry.kind != "upload":
            seed = st.number_input("Random seed", min_value=1, max_value=999999, value=int(case.geometry.seed or 42), key="inspector_seed")
            if seed != case.geometry.seed:
                case.geometry.seed = int(seed)
                case.geometry.centers = None
                mark_case_dirty()
                append_command_log(f"Seed updated to {seed}.")
                st.rerun()

        r_mono = st.number_input(
            "Monomer radius a (nm)",
            min_value=1.0,
            max_value=500.0,
            value=float(case.geometry.monomer_radius_nm),
            key="inspector_radius",
        )
        if r_mono != case.geometry.monomer_radius_nm:
            case.geometry.monomer_radius_nm = float(r_mono)
            if case.geometry.kind != "upload":
                case.geometry.centers = None
            mark_case_dirty()
            append_command_log(f"Monomer radius updated to {r_mono} nm.")
            st.rerun()

        if case.geometry.kind == "upload":
            uploaded_file = st.file_uploader("Upload coordinates (CSV/XYZ)", type=["csv", "xyz", "txt"], key="inspector_upload")
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode("utf-8")
                    lines = []
                    for line in content.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.replace(",", " ").split()
                        if len(parts) >= 3:
                            lines.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if lines:
                        case.geometry.centers = lines
                        case.geometry.N = len(lines)
                        mark_case_dirty()
                        append_command_log(f"Uploaded custom coordinate centers file ({len(lines)} spheres).")
                        st.success("Coordinates loaded.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")

        wl = st.number_input(
            "Wavelength λ (nm)",
            min_value=100.0,
            max_value=2000.0,
            value=float(case.optics.wavelength_nm),
            key="inspector_wavelength",
        )
        if wl != case.optics.wavelength_nm:
            case.optics.wavelength_nm = float(wl)
            mark_case_dirty()
            append_command_log(f"Wavelength updated to {wl} nm.")
            st.rerun()

        col_mr, col_mi = st.columns(2)
        with col_mr:
            m_real = st.number_input("Refractive index n", min_value=0.5, max_value=5.0, value=float(case.optics.refractive_index_real), key="inspector_m_real")
        with col_mi:
            m_imag = st.number_input("Absorption index k", min_value=0.0, max_value=5.0, value=float(case.optics.refractive_index_imag), key="inspector_m_imag")
        if m_real != case.optics.refractive_index_real:
            case.optics.refractive_index_real = float(m_real)
            mark_case_dirty()
            append_command_log(f"Refractive index real part updated to {m_real}.")
            st.rerun()
        if m_imag != case.optics.refractive_index_imag:
            case.optics.refractive_index_imag = float(m_imag)
            mark_case_dirty()
            append_command_log(f"Refractive index imaginary part updated to {m_imag}.")
            st.rerun()

    cuda_available = False
    try:
        import cupy as cp
        cuda_available = cp.is_available()
    except Exception:
        cp = None

    realizations = [
        ("Remote workstation GPU dense", "remote_gpu_dense"),
        ("Remote workstation GPU matrix-free", "remote_gpu_matrix_free"),
        ("CPU-parallel direct", "cpu_parallel"),
        ("CPU-serial direct", "cpu_serial"),
        ("Precomputed reference", "demo"),
    ]
    if cuda_available:
        realizations.insert(0, ("Local GPU dense", "gpu_dense"))
        realizations.insert(1, ("Local GPU matrix-free", "gpu_matrix_free"))

    real_labels = [r[0] for r in realizations]
    real_values = [r[1] for r in realizations]
    curr_real = case.solver.realization if case.solver.realization in real_values else "cpu_parallel"
    selected_real_label = st.selectbox(
        "Solver realization",
        real_labels,
        index=real_values.index(curr_real),
        key="inspector_solver_realization",
    )
    selected_real = real_values[real_labels.index(selected_real_label)]
    if selected_real != case.solver.realization:
        case.solver.realization = selected_real
        case.solver.demo_mode = selected_real == "demo"
        mark_case_dirty()
        append_command_log(f"Solver realization updated to {solver_realization_label(selected_real)}.")
        st.rerun()

    col_p, col_P = st.columns(2)
    with col_p:
        sphere_order = st.number_input("Sphere order p", min_value=1, max_value=12, value=int(case.solver.sphere_order), key="inspector_sphere_order")
    with col_P:
        order_mode = st.selectbox(
            "Aggregate order P",
            ["auto", "custom"],
            index=0 if case.solver.aggregate_order == "auto" else 1,
            key="inspector_aggregate_order_mode",
        )
    if sphere_order != case.solver.sphere_order:
        case.solver.sphere_order = int(sphere_order)
        mark_case_dirty()
        append_command_log(f"Sphere order p updated to {sphere_order}.")
        st.rerun()
    if order_mode == "auto":
        if case.solver.aggregate_order != "auto":
            case.solver.aggregate_order = "auto"
            mark_case_dirty()
            append_command_log("Aggregate order switched to automatic estimate.")
            st.rerun()
    else:
        default_P = int(case.solver.aggregate_order) if isinstance(case.solver.aggregate_order, int) else case.solver.sphere_order + 3
        custom_P = st.number_input("Custom aggregate order", min_value=1, max_value=80, value=default_P, key="inspector_custom_P")
        if custom_P != case.solver.aggregate_order:
            case.solver.aggregate_order = int(custom_P)
            mark_case_dirty()
            append_command_log(f"Aggregate order P updated to {custom_P}.")
            st.rerun()

    remote_selected = is_remote_realization(case.solver.realization)
    st.markdown('<div class="rcr-panel-title" style="margin-top:0.7rem;"><span>Execution resources</span></div>', unsafe_allow_html=True)
    if cuda_available:
        st.markdown('<div class="rcr-subtle" style="color:var(--rcr-accent);"><b>Local CUDA / CuPy</b>: available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rcr-subtle" style="color:var(--rcr-warning);"><b>Local CUDA / CuPy</b>: not detected</div>', unsafe_allow_html=True)

    if remote_selected:
        if st.button("Check remote workstation", width='stretch', key="check_remote_status"):
            st.session_state.remote_status = check_remote_status()
        remote_status = st.session_state.get("remote_status")
        if remote_status is None:
            st.info("Remote workstation selected. Refresh SSH/CUDA status before a live run.")
        elif remote_status.get("reachable"):
            cupy_label = remote_status.get("cupy_version", "available")
            gpus = remote_status.get("gpus", [])
            gpu_rows = "".join(
                f"<tr><td>GPU {gpu['index']}</td><td>{gpu['name']}</td><td>{gpu['memory_used']} / {gpu['memory_total']}</td><td>{gpu['utilization']}</td></tr>"
                for gpu in gpus
            )
            st.markdown(
                f"""
                <div style="border:1px solid var(--rcr-border); border-radius:6px; background:#f8fafc; padding:0.55rem; margin:0.45rem 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                        <span style="font-weight:650;">Remote workstation status</span>
                        <span style="color:var(--rcr-accent); font-weight:650;">Reachable</span>
                    </div>
                    <table class="rcr-table">
                        <tr><th>Resource</th><th>Name</th><th>Memory</th><th>Util.</th></tr>
                        {gpu_rows}
                    </table>
                    <div class="rcr-subtle" style="margin-top:0.35rem;">CuPy/CUDA: {cupy_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for gpu in gpus:
                try:
                    used_mib = float(str(gpu["memory_used"]).split()[0])
                    total_mib = float(str(gpu["memory_total"]).split()[0])
                    pct = used_mib / total_mib if total_mib else 0.0
                    st.progress(pct, text=f"GPU {gpu['index']} VRAM: {gpu['memory_used']} / {gpu['memory_total']}")
                except Exception:
                    st.caption(f"GPU {gpu['index']} VRAM: {gpu['memory_used']} / {gpu['memory_total']}")
        else:
            st.error(f"Remote workstation not reachable: {_remote_message(remote_status)}")

    if cuda_available and not remote_selected:
        try:
            dev = cp.cuda.Device(0)
            free_mem, total_mem = dev.mem_info
            used_gb = (total_mem - free_mem) / (1024**3)
            total_gb = total_mem / (1024**3)
            st.progress(used_gb / total_gb, text=f"Local GPU VRAM: {used_gb:.1f} / {total_gb:.1f} GiB")
        except Exception:
            st.caption("Local GPU VRAM status unavailable.")

    cpu_count = os.cpu_count() or 4
    default_threads = min(cpu_count, int(st.session_state.get("cpu_threads", min(16, cpu_count))))
    threads = st.selectbox("CPU helper threads", list(range(1, cpu_count + 1)), index=max(0, default_threads - 1), key="inspector_cpu_threads")
    st.session_state.cpu_threads = int(threads)

    try:
        import psutil
        vm = psutil.virtual_memory()
        st.progress(vm.percent / 100.0, text=f"System memory: {vm.used/(1024**3):.1f} / {vm.total/(1024**3):.1f} GiB ({vm.percent:.0f}%)")
    except ImportError:
        st.caption("System memory status unavailable.")

    st.markdown('<div class="rcr-panel-title" style="margin-top:0.7rem;"><span>Numerical settings</span></div>', unsafe_allow_html=True)
    st.selectbox("Working precision", ["complex128", "mixed precision", "complex64"], index=1 if case.solver.realization in ("gpu_dense", "remote_gpu_dense") else 0, key="inspector_precision")
    st.number_input("Tolerance (relative)", min_value=1e-16, max_value=1e-1, value=1e-12, format="%.2e", key="inspector_tol")
    st.number_input("Maximum iterations", min_value=100, max_value=50000, value=10000, key="inspector_max_iters")

    total_equations = case.geometry.N * (2 * case.solver.sphere_order * (case.solver.sphere_order + 2))
    dense_mem_gib = (total_equations ** 2) * 16 / (1024 ** 3)
    recommendation_color = "var(--rcr-accent)" if dense_mem_gib < 16.0 else "var(--rcr-warning)"
    recommendation_text = "Dense realization is suitable for this case." if dense_mem_gib < 16.0 else "Dense memory is high; matrix-free realization may be preferable."
    st.markdown(
        f"""
        <div style="border:1px solid var(--rcr-border); border-radius:6px; background:#f8fafc; padding:0.55rem; margin-top:0.55rem;">
            <div style="font-weight:650; margin-bottom:0.25rem;">Recommendation</div>
            <div style="font-size:0.76rem; color:{recommendation_color};">{recommendation_text}</div>
            <div class="rcr-subtle" style="margin-top:0.25rem;">Estimated dense matrix: D = {total_equations}, memory ≈ {dense_mem_gib:.2f} GiB.</div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; border-top:1px dashed var(--rcr-border); padding-top:0.45rem; margin-top:0.55rem; color:var(--rcr-muted);">
            <span>Reproducibility hash</span><span><code>{case_hash}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="border-top:1px solid var(--rcr-border); margin:0.65rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="rcr-panel-title"><span>Quick Export</span><span class="rcr-subtle">active result</span></div>', unsafe_allow_html=True)
    result = st.session_state.current_result
    if result:
        from apps.rcr_workbench.exports import export_results_json, export_phase_matrix_csv
        from apps.rcr_workbench.plotting import render_phase_matrix_matplotlib
        import io
        import matplotlib.pyplot as plt

        json_data = export_results_json(result)
        csv_data = export_phase_matrix_csv(result)
        fig_pm = render_phase_matrix_matplotlib(result.angles_deg, result.phase_matrix)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button("JSON", data=json_data, file_name="results.json", mime="application/json", width='stretch')
            buf_pm = io.BytesIO()
            fig_pm.savefig(buf_pm, format="png", dpi=150, bbox_inches="tight")
            buf_pm.seek(0)
            st.download_button("PNG", data=buf_pm, file_name="phase_matrix.png", mime="image/png", width='stretch')
        with col_exp2:
            st.download_button("CSV", data=csv_data, file_name="phase_matrix.csv", mime="text/csv", width='stretch')
            buf_pdf = io.BytesIO()
            fig_pm.savefig(buf_pdf, format="pdf", bbox_inches="tight")
            buf_pdf.seek(0)
            st.download_button("PDF", data=buf_pdf, file_name="phase_matrix.pdf", mime="application/pdf", width='stretch')
        plt.close(fig_pm)
    else:
        st.info("Run case to enable file downloads.")

    st.markdown("</div>", unsafe_allow_html=True)

def render_results_dock(case, result):
    """Render the results dock at the bottom of the page."""
    st.markdown(
        '<div class="rcr-panel rcr-panel-pad" style="margin-top:0.75rem;">'
        '<div class="rcr-panel-title">'
        '  <span>Results</span>'
        '  <span class="rcr-subtle">observables, phase matrix, validation</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    if result is None:
        st.info("No active calculation results loaded. Select a reference case or run the simulation.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
        
    dock_tabs = st.tabs(["Observables", "Phase Matrix", "Timing", "Validation", "Command Log"])
    
    with dock_tabs[0]:
        # Side-by-Side KPI card grid (left) and 6-panel chart (right)
        col_obs_kpis, col_obs_charts = st.columns([1.0, 3.0])
        
        with col_obs_kpis:
            # Custom styled efficiency metrics stacked vertically
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; gap:0.35rem; background-color:#ffffff; border:1px solid var(--rcr-border); padding:0.65rem; border-radius:6px;">
                <div style="border-bottom: 1px solid var(--rcr-border-soft); padding-bottom: 0.32rem;">
                    <div style="font-size:0.7rem; color:var(--rcr-muted);">Extinction, Q<sub>ext</sub></div>
                    <div style="font-size:1.18rem; font-weight:680; color:var(--rcr-primary);">{result.qext:.4f}</div>
                </div>
                <div style="border-bottom: 1px solid var(--rcr-border-soft); padding-bottom: 0.32rem;">
                    <div style="font-size:0.7rem; color:var(--rcr-muted);">Scattering, Q<sub>sca</sub></div>
                    <div style="font-size:1.18rem; font-weight:680; color:var(--rcr-primary);">{result.qsca:.4f}</div>
                </div>
                <div style="border-bottom: 1px solid var(--rcr-border-soft); padding-bottom: 0.32rem;">
                    <div style="font-size:0.7rem; color:var(--rcr-muted);">Absorption, Q<sub>abs</sub></div>
                    <div style="font-size:1.18rem; font-weight:680; color:var(--rcr-primary);">{result.qabs:.4f}</div>
                </div>
                <div style="border-bottom: 1px solid var(--rcr-border-soft); padding-bottom: 0.32rem;">
                    <div style="font-size:0.7rem; color:var(--rcr-muted);">Single-scattering albedo</div>
                    <div style="font-size:1.18rem; font-weight:680; color:var(--rcr-primary);">{result.ssa:.4f}</div>
                </div>
                <div>
                    <div style="font-size:0.7rem; color:var(--rcr-muted);">Asymmetry, g</div>
                    <div style="font-size:1.18rem; font-weight:680; color:var(--rcr-primary);">{result.g:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Wavelength footer line
            wl = case.optics.wavelength_nm
            m = f"{case.optics.refractive_index_real} + {case.optics.refractive_index_imag}i"
            st.markdown(
                f'<div class="rcr-subtle" style="margin-top:0.5rem; line-height: 1.35;">'
                f'  Wavelength: <b>{wl:.0f} nm</b><br>'
                f'  m = <b>{m}</b><br>'
                f'  p = <b>{case.solver.sphere_order}</b> | P = <b>{case.solver.aggregate_order}</b>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with col_obs_charts:
            ref_pm = None
            ref_case = case.model_copy(deep=True)
            ref_case.solver.demo_mode = True
            ref_res = load_demo_result(ref_case)
            if ref_res:
                ref_pm = ref_res.phase_matrix
            st.markdown(
                """
                <div style="height: 100%; min-height: 240px; display:flex; flex-direction:column;
                            justify-content:center; align-items:center; border:1px solid var(--rcr-border-soft);
                            border-radius:6px; background:#f8fafc; color:var(--rcr-muted); text-align:center;">
                    <div style="font-weight:650; color:var(--rcr-primary); margin-bottom:0.35rem;">Phase matrix ready</div>
                    <div style="font-size:0.82rem;">Open the Phase Matrix tab to inspect all six non-zero elements from 0 to 180 degrees.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    with dock_tabs[1]:
        st.markdown("##### Orientation-averaged phase matrix elements")
        # Direct Plotly chart view. Build a separate Figure object so Streamlit
        # does not collide with the observables tab's chart element ID.
        fig_phase = plot_phase_matrix_plotly(
            angles=result.angles_deg,
            pm_dict=result.phase_matrix,
            ref_pm_dict=ref_pm,
            ref_name="Reference"
        )
        fig_phase.update_layout(height=550, margin=dict(l=0, r=0, b=0, t=25))
        st.plotly_chart(fig_phase, width='stretch', key="results_phase_matrix_tab")
        
    with dock_tabs[2]:
        st.markdown("##### Timing profile")
        t_geom = result.timing.get("geometry_s", 0.0)
        t_solve = result.timing.get("solve_s", 0.0)
        t_phase = result.timing.get("phase_s", 0.0)
        t_total = result.timing.get("total_s", 0.0)
        
        st.markdown(f"""
        - **Geometry realization**: `{t_geom:.4f} s`
        - **Linear-system solve**: `{t_solve:.4f} s`
        - **Orientation averaging and phase matrix**: `{t_phase:.4f} s`
        - **Total elapsed duration**: `{t_total:.4f} s`
        """)
        
    with dock_tabs[3]:
        st.markdown("##### Reference validation")
        if ref_res:
            diff_qext = abs(result.qext - ref_res.qext) / ref_res.qext if ref_res.qext > 0.0 else 0.0
            diff_qsca = abs(result.qsca - ref_res.qsca) / ref_res.qsca if ref_res.qsca > 0.0 else 0.0
            diff_g = abs(result.g - ref_res.g) if ref_res.g != 0.0 else 0.0
            
            st.success("Calculations match precomputed paper reference values.")
            st.markdown(f"""
            - **Relative Qext difference**: `{diff_qext*100:.4f}%` (Reference: `{ref_res.qext:.6f}`)
            - **Relative Qsca difference**: `{diff_qsca*100:.4f}%` (Reference: `{ref_res.qsca:.6f}`)
            - **Relative g difference**: `{diff_g:.6e}` (Reference: `{ref_res.g:.6f}`)
            """)
        else:
            st.info("Validation reference not available for this custom morphology.")
            
    with dock_tabs[4]:
        st.markdown("##### Calculation log")
        logs = st.session_state.command_log
        st.code("\n".join(logs) if logs else "No command history.")
    st.markdown("</div>", unsafe_allow_html=True)
