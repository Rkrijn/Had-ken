import time
import subprocess
import psutil
import statistics

def get_gpu_util():
    try:
        # vraag alleen het gpu-util veld op in nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def monitor(duration=30, interval=1.0):
    cpu_samples = []
    gpu_samples = []

    print(f"[INFO] meten gedurende {duration} sec (interval {interval} sec)…")
    t0 = time.time()
    while time.time() - t0 < duration:
        cpu = psutil.cpu_percent(interval=None)
        gpu = get_gpu_util()
        cpu_samples.append(cpu)
        if gpu is not None:
            gpu_samples.append(gpu)
            print(f"CPU {cpu:5.1f}%   GPU {gpu:5.1f}%")
        else:
            print(f"CPU {cpu:5.1f}%   GPU n/a")

        time.sleep(interval)

    print("\n--- GEMIDDELDES ---")
    print(f"CPU: {statistics.mean(cpu_samples):.1f}% (n={len(cpu_samples)})")
    if gpu_samples:
        print(f"GPU: {statistics.mean(gpu_samples):.1f}% (n={len(gpu_samples)})")
    else:
        print("GPU: niet gemeten (nvidia-smi niet gevonden)")

if __name__ == "__main__":
    # pas duration/interval aan naar wens
    monitor(duration=180, interval=1.0)