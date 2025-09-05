"""
benchmark_sweep.py
Automatisch meerdere PPO-configs testen en CPU/GPU/FPS loggen.
"""

import argparse
import time
import psutil
import subprocess
import statistics
import csv
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor

import retro
import gymnasium as gym
from gymnasium.wrappers import TimeLimit


# -------------------------
# Env helpers
# -------------------------
def make_retro(game="Airstriker-Genesis", state=None, max_episode_steps=None):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    if max_episode_steps:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def make_train_env(game, state, scenario=None):
    def _thunk():
        return make_retro(game=game, state=state)
    return _thunk


# -------------------------
# GPU util helper
# -------------------------
def get_gpu_util():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# -------------------------
# Benchmark run
# -------------------------
def run_benchmark(game, n_envs, n_steps, batch_size, n_epochs, duration=60):
    print(f"\n[INFO] Benchmark: envs={n_envs}, n_steps={n_steps}, batch={batch_size}, epochs={n_epochs}, duration={duration}s")

    env = SubprocVecEnv([make_train_env(game, state=None)] * n_envs, start_method="forkserver")
    env = VecFrameStack(env, 4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        "CnnPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=2.5e-4,
        verbose=0,
        device=device,
    )

    # Optioneel compile
    if hasattr(torch, "compile"):
        try:
            model.policy = torch.compile(model.policy, mode="max-autotune")
        except Exception:
            pass

    cpu_samples, gpu_samples, mem_samples = [], [], []
    t_start = time.time()
    steps_done = 0

    while time.time() - t_start < duration:
        model.learn(total_timesteps=n_steps * n_envs, reset_num_timesteps=False)
        steps_done += n_steps * n_envs

        cpu = psutil.cpu_percent(interval=None)
        gpu = get_gpu_util()
        mem = psutil.virtual_memory().percent
        cpu_samples.append(cpu)
        if gpu is not None:
            gpu_samples.append(gpu)
        mem_samples.append(mem)

    results = {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "cpu_avg": statistics.mean(cpu_samples),
        "gpu_avg": statistics.mean(gpu_samples) if gpu_samples else 0,
        "mem_avg": statistics.mean(mem_samples),
        "steps_done": steps_done,
        "fps": steps_done / duration,
    }

    env.close()
    return results


# -------------------------
# Main sweep
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="StreetFighterIISpecialChampionEdition-Genesis")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--out", default="benchmark_results.csv")
    args = parser.parse_args()

    # vaste hyperparams behalve n_envs
    n_steps = 2048
    batch_size = 2048
    n_epochs = 3

    env_variants = [8, 16, 24, 32, 40, 46]

    results = []
    for n_envs in env_variants:
        res = run_benchmark(args.game, n_envs, n_steps, batch_size, n_epochs, args.duration)
        results.append(res)

    # schrijf CSV
    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] Alle resultaten opgeslagen in {out_path.absolute()}")


if __name__ == "__main__":
    main()