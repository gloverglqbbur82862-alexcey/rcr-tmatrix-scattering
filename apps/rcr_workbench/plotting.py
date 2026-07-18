import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

def plot_geometry_3d(centers: np.ndarray, radius_nm: float) -> go.Figure:
    """Create a 3D scatter plot of the aggregate sphere centers colored by depth (Z)."""
    if centers is None or len(centers) == 0:
        fig = go.Figure()
        fig.update_layout(title="No Geometry Loaded")
        return fig
        
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    
    # Compute circumscribed radius for axis scaling
    max_range = max(
        np.max(x) - np.min(x) if len(x) > 1 else 1.0,
        np.max(y) - np.min(y) if len(y) > 1 else 1.0,
        np.max(z) - np.min(z) if len(z) > 1 else 1.0
    )
    mid_x = (np.max(x) + np.min(x)) / 2 if len(x) > 0 else 0
    mid_y = (np.max(y) + np.min(y)) / 2 if len(y) > 0 else 0
    mid_z = (np.max(z) + np.min(z)) / 2 if len(z) > 0 else 0
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=10,
            color=z,
            colorscale='Viridis',
            colorbar=dict(title="Z depth (nm)", thickness=15),
            opacity=0.9,
            line=dict(color='white', width=1)
        ),
        name="Monomer center",
        hovertemplate='X: %{x:.2f} nm<br>Y: %{y:.2f} nm<br>Z: %{z:.2f} nm<extra></extra>'
    )])
    
    # Equal aspect ratio layout
    fig.update_layout(
        title="Aggregate Geometry 3D View",
        scene=dict(
            xaxis=dict(title='X (nm)', range=[mid_x - max_range/2 - radius_nm, mid_x + max_range/2 + radius_nm]),
            yaxis=dict(title='Y (nm)', range=[mid_y - max_range/2 - radius_nm, mid_y + max_range/2 + radius_nm]),
            zaxis=dict(title='Z (nm)', range=[mid_z - max_range/2 - radius_nm, mid_z + max_range/2 + radius_nm]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_white"
    )
    return fig

def plot_phase_matrix_plotly(
    angles: List[float], 
    pm_dict: Dict[str, List[float]], 
    ref_pm_dict: Optional[Dict[str, List[float]]] = None,
    ref_name: str = "Reference"
) -> go.Figure:
    """Create a 6-panel Plotly layout of Mueller matrix elements with optional reference markers."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "F11 (Phase Function)",
            "F12 / F11 (Polarization Ratio)",
            "F22 / F11 (Depolarization Ratio)",
            "F33 / F11",
            "F34 / F11",
            "F44 / F11"
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    elements = [
        ("F11", 1, 1, '#1e3a8a'),
        ("F12_over_F11", 1, 2, '#1e3a8a'),
        ("F22_over_F11", 2, 1, '#1e3a8a'),
        ("F33_over_F11", 2, 2, '#1e3a8a'),
        ("F34_over_F11", 3, 1, '#1e3a8a'),
        ("F44_over_F11", 3, 2, '#1e3a8a')
    ]
    
    for key, row, col, color in elements:
        # Main trace (lines)
        if key in pm_dict and len(pm_dict[key]) > 0:
            fig.add_trace(go.Scatter(
                x=angles, y=pm_dict[key], 
                name=f"Current {key}", mode='lines', 
                line=dict(color=color, width=2.0)
            ), row=row, col=col)
            
        # Reference trace (scatter markers)
        if ref_pm_dict and key in ref_pm_dict and len(ref_pm_dict[key]) > 0:
            # Subsample angles/markers for visual clarity (every 5th angle)
            angles_sub = angles[::5]
            ref_sub = ref_pm_dict[key][::5]
            fig.add_trace(go.Scatter(
                x=angles_sub, y=ref_sub, 
                name=f"{ref_name} {key}", mode='markers',
                marker=dict(color='black', size=5, symbol='circle-open')
            ), row=row, col=col)
            
        # Set ranges and logs
        if key == "F11":
            fig.update_yaxes(type="log", row=row, col=col)
        elif key == "F22_over_F11":
            fig.update_yaxes(range=[-0.05, 1.05], row=row, col=col)
        else:
            fig.update_yaxes(range=[-1.05, 1.05], row=row, col=col)
            
        fig.update_xaxes(title_text="Angle (deg)", row=row, col=col)
        
    fig.update_layout(
        height=950,
        showlegend=False,
        title_text="Orientation-Averaged Phase Matrix Elements",
        template="plotly_white"
    )
    return fig

def render_phase_matrix_matplotlib(
    angles: List[float], 
    pm_dict: Dict[str, List[float]], 
    ref_pm_dict: Optional[Dict[str, List[float]]] = None,
    ref_name: str = "Reference"
) -> plt.Figure:
    """Render a publication-ready Matplotlib figure of the phase matrix elements."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 13), dpi=300)
    
    elements = [
        ("F11", 0, 0, 'royalblue', "F11 (Phase Function)", True, None),
        ("F12_over_F11", 0, 1, 'firebrick', "F12 / F11 (Polarization)", False, (-1.05, 1.05)),
        ("F22_over_F11", 1, 0, 'forestgreen', "F22 / F11 (Depolarization)", False, (-0.05, 1.05)),
        ("F33_over_F11", 1, 1, 'darkorange', "F33 / F11", False, (-1.05, 1.05)),
        ("F34_over_F11", 2, 0, 'purple', "F34 / F11", False, (-1.05, 1.05)),
        ("F44_over_F11", 2, 1, 'teal', "F44 / F11", False, (-1.05, 1.05))
    ]
    
    for key, row, col, color, title, is_log, ylim in elements:
        ax = axes[row, col]
        
        # Plot main curve
        if key in pm_dict and len(pm_dict[key]) > 0:
            ax.plot(angles, pm_dict[key], color=color, linewidth=2, label="Current Result")
            
        # Plot reference markers
        if ref_pm_dict and key in ref_pm_dict and len(ref_pm_dict[key]) > 0:
            angles_sub = angles[::6]
            ref_sub = ref_pm_dict[key][::6]
            ax.scatter(angles_sub, ref_sub, facecolors='none', edgecolors='black', s=15, label=ref_name, zorder=3)
            
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Scattering Angle (deg)", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        
        if is_log:
            ax.set_yscale('log')
        if ylim:
            ax.set_ylim(ylim)
            
        if ref_pm_dict:
            ax.legend(fontsize=8, loc="best")
            
    plt.tight_layout()
    return fig

def render_geometry_matplotlib(centers: np.ndarray, radius_nm: float) -> plt.Figure:
    """Render a 3D Matplotlib figure of the aggregate geometry for export."""
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=60, edgecolors='black', alpha=0.8)
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_title('Aggregate Geometry 3D View', fontsize=14, fontweight='bold')
    
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cb.set_label('Z depth (nm)', fontsize=10)
    
    # Calculate limits for equal aspect ratio
    max_range = max(
        np.max(x) - np.min(x) if len(x) > 1 else 1.0,
        np.max(y) - np.min(y) if len(y) > 1 else 1.0,
        np.max(z) - np.min(z) if len(z) > 1 else 1.0
    )
    mid_x = (np.max(x) + np.min(x)) / 2 if len(x) > 0 else 0
    mid_y = (np.max(y) + np.min(y)) / 2 if len(y) > 0 else 0
    mid_z = (np.max(z) + np.min(z)) / 2 if len(z) > 0 else 0
    
    ax.set_xlim(mid_x - max_range/2 - radius_nm, mid_x + max_range/2 + radius_nm)
    ax.set_ylim(mid_y - max_range/2 - radius_nm, mid_y + max_range/2 + radius_nm)
    ax.set_zlim(mid_z - max_range/2 - radius_nm, mid_z + max_range/2 + radius_nm)
    
    plt.tight_layout()
    return fig
