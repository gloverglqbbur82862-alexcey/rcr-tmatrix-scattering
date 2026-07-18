"""SSH-based remote execution support for the RCR Workbench."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional

from apps.rcr_workbench.models import CaseConfig, SolverResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_HOST = os.environ.get("RCR_REMOTE_HOST", "")
DEFAULT_REMOTE_REPO = os.environ.get("RCR_REMOTE_REPO", "")
DEFAULT_REMOTE_PYTHON = os.environ.get("RCR_REMOTE_PYTHON", "python")
REMOTE_WORK_DIR = ".workbench_remote"

REMOTE_REALIZATION_MAP = {
    "remote_gpu_dense": "gpu_dense",
    "remote_gpu_matrix_free": "gpu_matrix_free",
}


def is_remote_realization(realization: str) -> bool:
    """Return True if a public solver realization should execute remotely."""
    return realization in REMOTE_REALIZATION_MAP


def local_realization_for_remote(realization: str) -> str:
    """Map public remote solver realization names to remote-local names."""
    try:
        return REMOTE_REALIZATION_MAP[realization]
    except KeyError as exc:
        raise ValueError(f"Unknown remote solver realization: {realization}") from exc


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _remote_cmd(remote_repo: str, command: str) -> str:
    return f'cd /d {remote_repo} && {command}'


def _require_remote_config(host: str, remote_repo: str) -> None:
    if not host or not remote_repo:
        raise RuntimeError(
            "Remote execution requires RCR_REMOTE_HOST and RCR_REMOTE_REPO "
            "to be set in the environment."
        )


def sync_workbench_modules(
    host: str = DEFAULT_REMOTE_HOST,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Copy the workbench app modules to the remote project checkout."""
    _require_remote_config(host, remote_repo)
    if progress_callback:
        progress_callback("Synchronizing workbench modules to remote workstation...")

    mkdir_result = _run(["ssh", host, _remote_cmd(remote_repo, f"if not exist {REMOTE_WORK_DIR} mkdir {REMOTE_WORK_DIR}")])
    if mkdir_result.returncode != 0:
        raise RuntimeError(f"Failed to prepare remote work directory:\n{mkdir_result.stderr or mkdir_result.stdout}")

    apps_src = PROJECT_ROOT / "apps"
    scp_result = _run(["scp", "-r", str(apps_src), f"{host}:{remote_repo}/"], timeout=180)
    if scp_result.returncode != 0:
        raise RuntimeError(f"Failed to synchronize workbench modules:\n{scp_result.stderr or scp_result.stdout}")


def check_remote_status(
    host: str = DEFAULT_REMOTE_HOST,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    remote_python: str = DEFAULT_REMOTE_PYTHON,
    timeout: int = 12,
) -> dict:
    """Probe SSH, CuPy, and GPU availability on the remote workstation."""
    _require_remote_config(host, remote_repo)
    status: dict = {
        "host": host,
        "reachable": False,
        "cupy_available": False,
        "gpus": [],
        "message": "",
    }

    query = (
        f"{remote_python} -c \"import cupy as cp; "
        "print('cupy=' + cp.__version__); "
        "print('devices=' + str(cp.cuda.runtime.getDeviceCount()))\" "
        "&& nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader"
    )
    result = _run(["ssh", host, _remote_cmd(remote_repo, query)], timeout=timeout)
    if result.returncode != 0:
        status["message"] = (result.stderr or result.stdout).strip()
        return status

    status["reachable"] = True
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("cupy="):
            status["cupy_available"] = True
            status["cupy_version"] = line.split("=", 1)[1]
        elif line.startswith("devices="):
            try:
                status["device_count"] = int(line.split("=", 1)[1])
            except ValueError:
                status["device_count"] = line.split("=", 1)[1]
        elif "," in line and line[0].isdigit():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                status["gpus"].append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total": parts[2],
                        "memory_used": parts[3],
                        "utilization": parts[4],
                    }
                )
    status["message"] = "Remote workstation is ready."
    return status


def run_case_remote(
    case: CaseConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
    host: str = DEFAULT_REMOTE_HOST,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    remote_python: str = DEFAULT_REMOTE_PYTHON,
    sync_modules: bool = True,
) -> SolverResult:
    """Execute a workbench case on a remote workstation and return the result."""
    _require_remote_config(host, remote_repo)
    public_realization = case.solver.realization
    local_realization = local_realization_for_remote(public_realization)

    if not shutil.which("ssh") or not shutil.which("scp"):
        raise RuntimeError("Remote execution requires local ssh and scp commands.")

    job_id = uuid.uuid4().hex[:12]
    remote_case = f"{REMOTE_WORK_DIR}/case_{job_id}.json"
    remote_result = f"{REMOTE_WORK_DIR}/result_{job_id}.json"

    remote_case_config = case.model_copy(deep=True)
    remote_case_config.solver.realization = local_realization
    remote_case_config.solver.demo_mode = False

    with tempfile.TemporaryDirectory(prefix="rcr_remote_") as tmpdir:
        local_case_path = Path(tmpdir) / "case.json"
        local_result_path = Path(tmpdir) / "result.json"
        local_case_path.write_text(remote_case_config.model_dump_json(indent=2), encoding="utf-8")

        if sync_modules:
            sync_workbench_modules(host=host, remote_repo=remote_repo, progress_callback=progress_callback)

        if progress_callback:
            progress_callback("Uploading case configuration to remote workstation...")
        upload_result = _run(["scp", str(local_case_path), f"{host}:{remote_repo}/{remote_case}"])
        if upload_result.returncode != 0:
            raise RuntimeError(f"Failed to upload remote case:\n{upload_result.stderr or upload_result.stdout}")

        if progress_callback:
            progress_callback(f"Running solver on remote workstation ({public_realization})...")
        job_command = (
            f"{remote_python} apps/rcr_workbench/remote_job.py "
            f"--case-json {remote_case} "
            f"--output-json {remote_result} "
            f"--local-realization {local_realization} "
            f"--public-realization {public_realization} "
            f"--host-label remote_workstation"
        )
        run_result = _run(["ssh", host, _remote_cmd(remote_repo, job_command)], timeout=7200)
        if run_result.returncode != 0:
            raise RuntimeError(f"Remote solver failed:\nSTDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}")

        if progress_callback:
            progress_callback("Downloading remote solver result...")
        download_result = _run(["scp", f"{host}:{remote_repo}/{remote_result}", str(local_result_path)])
        if download_result.returncode != 0:
            raise RuntimeError(f"Failed to download remote result:\n{download_result.stderr or download_result.stdout}")

        result = SolverResult.model_validate_json(local_result_path.read_text(encoding="utf-8"))

    return result
