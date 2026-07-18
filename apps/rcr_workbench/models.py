from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union


SOLVER_REALIZATION_LABELS = {
    "cpu_parallel": "CPU-parallel direct",
    "cpu_serial": "CPU-serial direct",
    "gpu_matrix_free": "GPU matrix-free direct",
    "gpu_dense": "GPU dense preassembled",
    "remote_gpu_matrix_free": "Remote workstation GPU matrix-free direct",
    "remote_gpu_dense": "Remote workstation GPU dense preassembled",
    "demo": "precomputed reference",
}


def solver_realization_label(realization: str) -> str:
    """Return a reader-facing label for a solver realization identifier."""
    return SOLVER_REALIZATION_LABELS.get(realization, realization.replace("_", " "))

class GeometryConfig(BaseModel):
    kind: str  # "bcca", "bpca", "upload"
    N: int
    monomer_radius_nm: float
    seed: Optional[int] = None
    centers: Optional[List[List[float]]] = None

class OpticalConfig(BaseModel):
    wavelength_nm: float
    refractive_index_real: float
    refractive_index_imag: float

class SolverConfig(BaseModel):
    realization: str  # "cpu_parallel", "gpu_matrix_free", "gpu_dense", "demo"
    sphere_order: int
    aggregate_order: Union[int, str]  # e.g., 6 or "auto"
    demo_mode: bool = False

class CaseConfig(BaseModel):
    geometry: GeometryConfig
    optics: OpticalConfig
    solver: SolverConfig

class SolverResult(BaseModel):
    qext: float
    qsca: float
    qabs: float
    ssa: float
    g: float
    angles_deg: List[float]
    phase_matrix: Dict[str, List[float]]  # Dict containing F11, F12_over_F11, etc.
    timing: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
