import time
import glob
import os
from datetime import datetime
import pynvml

def read_int(path):
    with open(path, "r") as f:
        return int(f.read().strip())

def find_rapl_paths():
    base = "/sys/class/powercap"
    # Prefer package domain if present.
    package_energy = os.path.join(base, "intel-rapl:0", "energy_uj")
    package_max = os.path.join(base, "intel-rapl:0", "max_energy_range_uj")
    if os.path.exists(package_energy) and os.path.exists(package_max):
        return package_energy, package_max
    # Fall back to the first available domain.
    energy_files = glob.glob(os.path.join(base, "intel-rapl:*", "energy_uj"))
    if not energy_files:
        raise RuntimeError("No RAPL energy file found. Are you on an Intel CPU?")
    energy_path = energy_files[0]
    max_path = os.path.join(os.path.dirname(energy_path), "max_energy_range_uj")
    if not os.path.exists(max_path):
        raise RuntimeError(f"Missing max energy range file for {energy_path}")
    return energy_path, max_path

def read_rapl_energy(energy_path):
    """Reads energy in joules from RAPL sysfs."""
    energy_uj = read_int(energy_path)
    return energy_uj / 1_000_000  # convert microjoules to joules

def average_cpu_gpu_power(duration_sec=60, sample_interval=1.0, output_dir=None):
    """Logs CPU package + GPU power over a window. Writes per-sample CSV and summary txt."""
    # CPU (RAPL) setup.
    energy_path, max_path = find_rapl_paths()
    max_energy_j = read_int(max_path) / 1_000_000

    # GPU (NVML) setup. Read-only handle on GPU 0.
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    # Output paths.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"cpu_gpu_power_{timestamp}.csv")
    txt_path = os.path.join(output_dir, f"cpu_gpu_power_{timestamp}.txt")

    start_energy = read_rapl_energy(energy_path)
    start_time = time.monotonic()
    end_energy = start_energy
    gpu_samples = []

    # Per-sample CSV. Flushed each row so a Ctrl+C still leaves usable data.
    with open(csv_path, "w") as f:
        f.write("timestamp_iso,cpu_energy_j,gpu_power_w,gpu_util_pct\n")
        while (time.monotonic() - start_time) < duration_sec:
            now_iso = datetime.utcnow().isoformat() + "Z"
            cpu_e = read_rapl_energy(energy_path)
            gpu_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            gpu_u = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            f.write(f"{now_iso},{cpu_e},{gpu_w},{gpu_u}\n")
            f.flush()
            end_energy = cpu_e
            gpu_samples.append(gpu_w)
            time.sleep(sample_interval)

    pynvml.nvmlShutdown()

    # Aggregate.
    actual_duration = time.monotonic() - start_time
    delta = end_energy - start_energy
    if delta < 0:
        # Counter wrapped.
        delta += max_energy_j
    cpu_avg_power = delta / actual_duration
    gpu_avg_power = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0.0
    gpu_total_energy = gpu_avg_power * actual_duration  # rectangle approx; refine from CSV if needed

    print(f"GPU: {gpu_name}")
    print(f"Duration: {actual_duration:.1f}s")
    print(f"CPU avg power: {cpu_avg_power:.2f} W  (CPU energy: {delta:.1f} J)")
    print(f"GPU avg power: {gpu_avg_power:.2f} W  (GPU energy: {gpu_total_energy:.1f} J)")

    with open(txt_path, "w") as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"duration_sec: {actual_duration}\n")
        f.write(f"start_cpu_energy_j: {start_energy}\n")
        f.write(f"end_cpu_energy_j: {end_energy}\n")
        f.write(f"cpu_total_energy_j: {delta}\n")
        f.write(f"cpu_avg_power_w: {cpu_avg_power}\n")
        f.write(f"gpu_name: {gpu_name}\n")
        f.write(f"gpu_total_energy_j: {gpu_total_energy}\n")
        f.write(f"gpu_avg_power_w: {gpu_avg_power}\n")
        f.write(f"n_gpu_samples: {len(gpu_samples)}\n")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {txt_path}")

if __name__ == "__main__":
    average_cpu_gpu_power(1500)  # 25-minute test