import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

# Setup page config first
st.set_page_config(
    page_title="RCR Scattering Workbench",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from apps.rcr_workbench.ui_state import (
    init_state,
    make_case_id,
    make_case_hash,
    append_command_log,
)
from apps.rcr_workbench.components import (
    render_toolstrip,
    render_case_header,
    render_case_explorer,
    render_variables_panel,
    render_live_workspace,
    render_context_inspector,
    render_results_dock,
)
from apps.rcr_workbench.runner import run_case
from apps.rcr_workbench.remote_runner import is_remote_realization, run_case_remote

# Initialize session state
init_state()

# CSS injection for a modern scientific workbench style.
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    div[data-testid="collapsedControl"] {display: none !important;}

    :root {
        --rcr-bg: #f5f7fb;
        --rcr-surface: #ffffff;
        --rcr-surface-subtle: #f8fafc;
        --rcr-border: #d7dee8;
        --rcr-border-soft: #e7edf4;
        --rcr-text: #162033;
        --rcr-muted: #667085;
        --rcr-primary: #174a7c;
        --rcr-accent: #0f766e;
        --rcr-accent-soft: #e7f5f2;
        --rcr-warning: #b45309;
        --rcr-danger: #b91c1c;
        --rcr-code: #7c2d12;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: var(--rcr-bg);
        color: var(--rcr-text);
    }

    .main .block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 3.6rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100%;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 0.25rem !important;
        padding-bottom: 3.6rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    h1, h2, h3, h4, h5 {
        color: var(--rcr-text) !important;
        font-weight: 650 !important;
        letter-spacing: 0 !important;
    }

    code {
        color: var(--rcr-code);
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 3px;
        padding: 0.05rem 0.25rem;
    }

    div[data-testid="stMetric"] {
        background-color: var(--rcr-surface);
        border: 1px solid var(--rcr-border);
        padding: 8px 12px;
        border-radius: 4px;
        box-shadow: none;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 4px !important;
        border: 1px solid var(--rcr-border) !important;
        background: var(--rcr-surface) !important;
        color: var(--rcr-text) !important;
        font-weight: 560 !important;
        min-height: 2rem !important;
        padding: 0.3rem 0.55rem !important;
        box-shadow: none !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--rcr-primary) !important;
        color: var(--rcr-primary) !important;
        background: #f8fbff !important;
    }

    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: var(--rcr-primary) !important;
        border-color: var(--rcr-primary) !important;
        color: white !important;
    }

    div[data-baseweb="select"] > div,
    input,
    textarea {
        border-radius: 4px !important;
    }

    .rcr-topbar {
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0 0.25rem 0.55rem 0.25rem;
        border-bottom: 1px solid var(--rcr-border);
        margin-bottom: 0.45rem;
    }

    .rcr-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        min-width: 18rem;
    }

    .rcr-brand-mark {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background:
            radial-gradient(circle at 6px 6px, #174a7c 0 2px, transparent 2.5px),
            radial-gradient(circle at 15px 7px, #0f766e 0 2px, transparent 2.5px),
            radial-gradient(circle at 10px 15px, #2563eb 0 2px, transparent 2.5px),
            radial-gradient(circle at 18px 17px, #475569 0 2px, transparent 2.5px);
        border: 1px solid var(--rcr-border);
        background-color: white;
    }

    .rcr-brand-title {
        font-size: 1.02rem;
        font-weight: 680;
        color: var(--rcr-text);
        white-space: nowrap;
    }

    .rcr-breadcrumb {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        color: var(--rcr-muted);
        font-size: 0.78rem;
        min-width: 0;
    }

    .rcr-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.16rem 0.55rem;
        border-radius: 999px;
        border: 1px solid #b7e4d4;
        background: var(--rcr-accent-soft);
        color: #08724f;
        font-weight: 620;
        font-size: 0.75rem;
    }

    .rcr-dot {
        width: 0.42rem;
        height: 0.42rem;
        border-radius: 50%;
        background: var(--rcr-accent);
        display: inline-block;
    }

    .rcr-panel {
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
    }

    .rcr-panel-pad {
        padding: 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--rcr-border) !important;
        border-radius: 6px !important;
        background: var(--rcr-surface) !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.035);
    }

    .rcr-panel-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: var(--rcr-text);
        font-size: 0.86rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }

    .rcr-chip-row {
        display: flex;
        gap: 0.42rem;
        flex-wrap: wrap;
        margin: 0.35rem 0 0.62rem 0;
    }

    .rcr-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        min-height: 1.72rem;
        padding: 0.18rem 0.56rem;
        border: 1px solid var(--rcr-border);
        border-radius: 4px;
        background: var(--rcr-surface-subtle);
        color: #334155;
        font-size: 0.78rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        white-space: nowrap;
    }

    .rcr-subtle {
        color: var(--rcr-muted);
        font-size: 0.76rem;
    }

    .rcr-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.73rem;
    }

    .rcr-table th {
        text-align: left;
        color: #475569;
        font-weight: 650;
        background: var(--rcr-surface-subtle);
        border-bottom: 1px solid var(--rcr-border-soft);
        padding: 0.32rem 0.38rem;
    }

    .rcr-table td {
        border-bottom: 1px solid #edf2f7;
        padding: 0.32rem 0.38rem;
        vertical-align: top;
    }

    .rcr-worksheet-row {
        display: grid;
        grid-template-columns: 1.6rem minmax(0, 1fr);
        gap: 0.6rem;
        border-bottom: 1px solid var(--rcr-border-soft);
        padding: 0.45rem 0.25rem;
        font-size: 0.82rem;
        align-items: start;
    }

    .rcr-worksheet-row:last-child {
        border-bottom: 0;
    }

    .rcr-row-number {
        color: #94a3b8;
        font-weight: 650;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    .rcr-progress-line {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        border: 1px solid var(--rcr-border);
        border-radius: 6px;
        background: var(--rcr-surface-subtle);
        padding: 0.44rem 0.55rem;
    }

    .rcr-step {
        font-size: 0.72rem;
        text-align: center;
        color: var(--rcr-muted);
        min-width: 4.8rem;
    }

    .rcr-step.done {
        color: var(--rcr-accent);
        font-weight: 650;
    }

    .rcr-step.current {
        color: var(--rcr-primary);
        font-weight: 650;
    }

    .rcr-divider {
        color: #b8c2d0;
        font-weight: 650;
    }

    .rcr-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #eef3f8;
        border-top: 1px solid var(--rcr-border);
        display: flex;
        justify-content: space-between;
        padding: 0.28rem 0.95rem;
        font-size: 0.72rem;
        color: var(--rcr-muted);
        z-index: 99999;
    }

    .rcr-footer-right {
        display: flex;
        gap: 0.85rem;
        align-items: center;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# Active Case Configuration and results references
case = st.session_state.current_case
result = st.session_state.current_result
case_id = make_case_id(case)
case_hash = make_case_hash(case)

status = st.session_state.run_status
status_label = "Ready" if status == "Completed" else status
st.markdown(
    f"""
    <div class="rcr-topbar">
        <div class="rcr-brand">
            <div class="rcr-brand-mark"></div>
            <div class="rcr-brand-title">RCR Scattering Workbench</div>
        </div>
        <div class="rcr-breadcrumb">
            <span>Project</span><span>/</span><span>Soot aggregates</span><span>/</span>
            <strong style="color:var(--rcr-text);">{case_id}</strong>
            <span class="rcr-status-pill"><span class="rcr-dot"></span>{status_label}</span>
            <span>Hash: <code>{case_hash}</code></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Top Section - Toolstrip Menu
render_toolstrip()

# 2. Case Identifier Header
render_case_header(case, result)

# 3. Synchronous Solver Execution Handler
if st.session_state.run_status == "Running":
    st.markdown('<div class="rcr-panel rcr-panel-pad" style="margin-bottom:0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--rcr-warning); font-weight:650;">Executing scattering calculation...</div>', unsafe_allow_html=True)
    
    log_area = st.empty()
    log_list = []
    
    def progress_cb(msg):
        log_list.append(msg)
        log_area.code("\n".join(log_list[-6:]))
        st.session_state.solver_logs = log_list
        
    try:
        t0 = time.time()
        case_id = make_case_id(case)
        solver_real = case.solver.realization
        
        # Format executable command
        if solver_real == "demo":
            cmd = f'load_demo_result(case_id="{case_id}")'
        elif is_remote_realization(solver_real):
            cmd = f'run_case_remote(case_id="{case_id}", solver_realization="{solver_real}", tol=1e-12)'
        else:
            cmd = f'run_case(case_id="{case_id}", solver_realization="{solver_real}", tol=1e-12)'

        if is_remote_realization(solver_real):
            res = run_case_remote(case, progress_callback=progress_cb)
        else:
            res = run_case(case, progress_callback=progress_cb)
        
        # Save results
        st.session_state.current_result = res
        st.session_state.case_dirty = False
        st.session_state.run_status = "Completed"
        
        append_command_log(cmd)
        st.success(f"Simulation completed successfully in {time.time() - t0:.2f} seconds!")
        st.rerun()
    except FileNotFoundError as fnf:
        st.session_state.run_status = "Ready"
        st.error(f"Demo File Error: {fnf}")
        st.info("Please align parameters (N=8/16/32/64, wavelength=550nm, refractive index=1.95+0.79j) or disable precomputed reference mode.")
    except Exception as e:
        st.session_state.run_status = "Ready"
        st.error(f"Solver run failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

# 4. Main Four-Panel Layout Grid
col_left, col_mid, col_right = st.columns([1.1, 2.5, 1.4])

with col_left:
    with st.container(border=True):
        render_case_explorer(case, result)
    with st.container(border=True):
        render_variables_panel(case, result)
    
with col_mid:
    with st.container(border=True):
        render_live_workspace(case, result)
    
with col_right:
    with st.container(border=True):
        render_context_inspector(case)

# 5. Bottom Results Dock Section
with st.container(border=True):
    render_results_dock(case, result)

# 6. Status and Provenance command footer bar
st.markdown('<div style="margin-top: 0.65rem;"></div>', unsafe_allow_html=True)
col_foot_lbl, col_foot_cmd = st.columns([1.0, 4.0])
with col_foot_lbl:
    st.markdown('<span style="font-size: 0.78rem; font-weight: 650; color: var(--rcr-muted);">Command / provenance</span>', unsafe_allow_html=True)
with col_foot_cmd:
    last_cmd = ""
    if st.session_state.command_log:
        last_cmd = st.session_state.command_log[-1]
    else:
        case_id = make_case_id(case)
        solver_real = case.solver.realization
        last_cmd = f'run_case(case_id="{case_id}", solver_realization="{solver_real}", tol=1e-12)'
    st.markdown(f'<code style="font-size: 0.76rem;">{last_cmd}</code>', unsafe_allow_html=True)

# 7. Fixed Bottom Desktop Status Bar
case_id = make_case_id(case)
mem_pct = 64
try:
    import psutil
    mem_pct = int(psutil.virtual_memory().percent)
except Exception:
    pass

solver_target = "remote workstation" if is_remote_realization(case.solver.realization) else "local workstation"
footer_html = f"""
<div class="rcr-footer">
    <div>
        Active case: <span style="font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:var(--rcr-accent);">{case_id}</span>
    </div>
    <div class="rcr-footer-right">
        <span>Execution target: {solver_target}</span>
        <span style="color: #b8c2d0;">|</span>
        <span>CPU threads: {__import__('os').cpu_count() or 1}</span>
        <span style="color: #b8c2d0;">|</span>
        <span>MEM {mem_pct}%</span>
        <span style="color: #b8c2d0;">|</span>
        <span>v1.2.0</span>
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
