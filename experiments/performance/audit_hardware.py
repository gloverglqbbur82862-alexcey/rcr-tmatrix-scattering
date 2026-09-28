#!/usr/bin/env python3
"""Hardware and Environment Audit Script.

Gathers detailed specifications of the CPU, memory, OS, Python packages,
and CUDA/GPU availability, then writes them to hardware_audit.json.
"""

import os
import sys
import json
import platform
import subprocess

def get_cpu_model():
    """Returns the CPU model name in a cross-platform way."""
    system = platform.system()
    if system == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            try:
                # Fallback for Apple Silicon
                brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand"]).decode().strip()
                return brand
            except Exception:
                return "Unknown Apple CPU"
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return model.strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"

def get_cpu_cores():
    """Returns physical and logical core counts using cross-platform methods."""
    logical = os.cpu_count() or 1
    physical = logical
    
    # Try using psutil if available
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        # Fallback for Darwin/Linux without psutil
        system = platform.system()
        if system == "Darwin":
            try:
                physical = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
            except Exception:
                pass
        elif system == "Linux":
            try:
                # Count distinct core ids
                core_ids = set()
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("core id") or line.startswith("cpu cores"):
                            core_ids.add(line.split(":")[1].strip())
                if core_ids:
                    physical = len(core_ids)
            except Exception:
                pass
    return {"physical": physical, "logical": logical}

def get_total_ram():
    """Returns total system memory in GB."""
    system = platform.system()
    if system == "Darwin":
        try:
            bytes_mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            return bytes_mem / (1024 ** 3)
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb_mem = int(line.split()[1])
                        return kb_mem / (1024 ** 2)
        except Exception:
            pass
    # Fallback to psutil
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    return float("nan")

def audit():
    audit_data = {}

    # 1. System info
    audit_data["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine()
    }

    # 2. CPU info
    cores = get_cpu_cores()
    audit_data["cpu"] = {
        "model": get_cpu_model(),
        "physical_cores": cores["physical"],
        "logical_cores": cores["logical"],
        "ram_gb": get_total_ram()
    }

    # 3. Python and library versions
    audit_data["software"] = {
        "python_version": platform.python_version(),
        "numpy_version": "Not Installed",
        "scipy_version": "Not Installed",
        "numba_version": "Not Installed"
    }

    try:
        import numpy as np
        audit_data["software"]["numpy_version"] = np.__version__
    except ImportError:
        pass

    try:
        import scipy as sp
        audit_data["software"]["scipy_version"] = sp.__version__
    except ImportError:
        pass

    try:
        import numba
        audit_data["software"]["numba_version"] = numba.__version__
    except ImportError:
        pass

    # 4. GPU / CUDA Audit
    cuda_available = False
    gpu_info = []
    numba_cuda_available = False
    cupy_available = False
    cupy_device_info = "Not Installed"

    # Test Numba CUDA
    try:
        from numba import cuda
        numba_cuda_available = cuda.is_available()
        if numba_cuda_available:
            cuda_available = True
            for g in cuda.gpus:
                gpu_info.append({
                    "model": g.name.decode("utf-8") if isinstance(g.name, bytes) else str(g.name),
                    "id": g.id
                })
    except Exception as e:
        gpu_info.append({"error": f"Numba CUDA check failed: {e}"})

    # Test CuPy
    try:
        import cupy
        cupy_available = True
        try:
            device = cupy.cuda.Device(0)
            cupy_device_info = {
                "id": device.id,
                "compute_capability": device.compute_capability,
                "mem_info": device.mem_info  # (free, total) bytes
            }
        except Exception as e:
            cupy_device_info = f"CuPy installed but failed to query device: {e}"
    except ImportError:
        cupy_device_info = "CuPy Not Installed"

    audit_data["cuda"] = {
        "cuda_available": cuda_available,
        "numba_cuda_available": numba_cuda_available,
        "cupy_available": cupy_available,
        "cupy_device_info": cupy_device_info,
        "gpus": gpu_info
    }

    # Write to JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hardware_audit.json")
    
    with open(out_path, "w") as f:
        json.dump(audit_data, f, indent=4)

    print(f"Hardware audit completed successfully.")
    print(f"Results written to: {out_path}")
    print(json.dumps(audit_data, indent=2))

if __name__ == "__main__":
    audit()
