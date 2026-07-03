"""Collect a hardware/architecture profile for tagging pilot results."""
import platform
import socket
import multiprocessing
import subprocess
import sys


def _safe(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def get_hw_info():
    system = platform.system()
    machine = platform.machine()  # 'arm64' on Apple Silicon, 'x86_64' on Intel

    cpu_model = ""
    if system == "Darwin":
        cpu_model = _safe("sysctl -n machdep.cpu.brand_string")
    elif system == "Linux":
        cpu_model = _safe("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip()
    elif system == "Windows":
        cpu_model = platform.processor()

    # Detect vector instruction support (matters a lot for HE)
    features = ""
    if system == "Darwin":
        features = _safe("sysctl -n machdep.cpu.features") + " " + \
                   _safe("sysctl -n machdep.cpu.leaf7_features")
    elif system == "Linux":
        features = _safe("grep -m1 'flags' /proc/cpuinfo")

    has_avx512 = "avx512" in features.lower()
    has_avx2 = "avx2" in features.lower()
    is_apple_silicon = (system == "Darwin" and machine == "arm64")

    # Total RAM
    ram_gb = None
    if system == "Darwin":
        mem = _safe("sysctl -n hw.memsize")
        if mem.isdigit():
            ram_gb = round(int(mem) / (1024**3), 1)
    elif system == "Linux":
        mem = _safe("grep MemTotal /proc/meminfo | awk '{print $2}'")
        if mem.isdigit():
            ram_gb = round(int(mem) / (1024**2), 1)

    arch_label = "apple_silicon" if is_apple_silicon else \
                 "x86_64" if machine == "x86_64" else machine

    return {
        "hostname": socket.gethostname(),
        "system": system,
        "machine": machine,
        "arch_label": arch_label,
        "cpu_model": cpu_model,
        "physical_cores": multiprocessing.cpu_count(),
        "ram_gb": ram_gb,
        "is_apple_silicon": is_apple_silicon,
        "has_avx512": has_avx512,
        "has_avx2": has_avx2,
        "python_version": sys.version.split()[0],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_hw_info(), indent=2))
