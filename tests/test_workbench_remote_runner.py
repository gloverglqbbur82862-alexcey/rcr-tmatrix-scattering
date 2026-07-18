import subprocess

import pytest

from apps.rcr_workbench.remote_runner import (
    check_remote_status,
    is_remote_realization,
    local_realization_for_remote,
)


def test_remote_realization_mapping():
    assert is_remote_realization("remote_gpu_dense") is True
    assert is_remote_realization("remote_gpu_matrix_free") is True
    assert is_remote_realization("gpu_dense") is False
    assert local_realization_for_remote("remote_gpu_dense") == "gpu_dense"
    assert local_realization_for_remote("remote_gpu_matrix_free") == "gpu_matrix_free"

    with pytest.raises(ValueError):
        local_realization_for_remote("cpu_parallel")


def test_check_remote_status_parses_gpu_query(monkeypatch):
    stdout = "\n".join(
        [
            "cupy=14.0.1",
            "devices=2",
            "0, NVIDIA GeForce RTX 3090, 24576 MiB, 125 MiB, 0 %",
            "1, NVIDIA GeForce RTX 3090, 24576 MiB, 198 MiB, 1 %",
        ]
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("apps.rcr_workbench.remote_runner._run", fake_run)

    status = check_remote_status()

    assert status["reachable"] is True
    assert status["cupy_available"] is True
    assert status["cupy_version"] == "14.0.1"
    assert status["device_count"] == 2
    assert len(status["gpus"]) == 2
    assert status["gpus"][0]["name"] == "NVIDIA GeForce RTX 3090"
