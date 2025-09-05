# =========================
# Headless & speed env-vars (MOETEN vóór imports)
# =========================
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # voorkom venster (SDL)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # geen X/GL-venster nodig
os.environ.setdefault("DISPLAY", "")                # expliciet geen display
os.environ.setdefault("WAYLAND_DISPLAY", "")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1") # force software GL init
os.environ.setdefault("OMP_NUM_THREADS", "1")       # minder thread-contentie
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecMonitor,
)
from stable_baselines3.common.callbacks import CheckpointCallback

import torch
import retro


# -------------------------
# Torch / GPU tuning
# -------------------------
torch.set_num_threads(1)  # CPU vrijhouden voor env-subprocessen
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # snellere convs bij vaste inputgrootte


# -------------------------
# Wrappers
# -------------------------
class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0.0
        ob = None
        info = {}
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac

            # Probeer rendering te vermijden
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)

            totrew += rew
            if terminated or truncated:
                break

        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=None, **kwargs):
    if state is None:
        state = retro.State.DEFAULT

    # Geen render/record args → zo snel mogelijk
    env = retro.make(game, state, **kwargs)

    # Hard headless/no-render: audio uit, fast-forward/no-throttle indien beschikbaar
    try:
        env.unwrapped.em.set_audio(False)
    except Exception:
        pass
    # verschillende cores hebben verschillende API's; probeer ze allemaal
    for attr in ("set_fastforward", "set_speed", "set_no_throttle"):
        try:
            fn = getattr(env.unwrapped.em, attr)
            try:
                fn(True)          # bool API
            except TypeError:
                fn(1_000_000.0)   # factor API
        except Exception:
            pass
    # render volledig noop maken
    try:
        env.unwrapped.viewer = None
    except Exception:
        pass
    try:
        env.render = lambda *a, **k: None
    except Exception:
        pass

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    DeepMind-achtige preprocessing:
      - WarpFrame: 84x84, grayscale
      - ClipRewardEnv: clip {-1,0,+1}
      (geen per-env Monitor; we gebruiken VecMonitor op de hele vecenv)
    """
    env = WarpFrame(env)        # (84, 84, 1) HWC
    env = ClipRewardEnv(env)
    return env


def make_train_env(game, state, scenario, frame_skip, stickprob):
    def _thunk():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = StochasticFrameSkip(env, n=frame_skip, stickprob=stickprob)
        env = wrap_deepmind_retro(env)
        return env
    return _thunk


def latest_checkpoint(save_dir: Path):
    ckpts = sorted(Path(save_dir).glob("ckpt_*.zip"))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)

    # Logging & checkpoints
    parser.add_argument("--logdir", default="logs", help="TensorBoard log directory")
    parser.add_argument("--save-dir", default="runs/sb3", help="Directory voor checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000, help="Save elke N timesteps (globaal)")

    # Snelheidsknoppen
    parser.add_argument("--n-envs", type=int, default=24, help="Aantal parallelle envs (CPU bound)")
    parser.add_argument("--frame-skip", type=int, default=8, help="Frames per agent-actie")
    parser.add_argument("--stickprob", type=float, default=0.25, help="Kans vorige actie te blijven gebruiken")

    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Hervat vanaf laatste checkpoint indien aanwezig")
    args = parser.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # VecEnv
    # -------------------------
    train_vec = SubprocVecEnv([
        make_train_env(args.game, args.state, args.scenario, args.frame_skip, args.stickprob)
    ] * args.n_envs,
    start_method="forkserver"
    )

    # (84,84,1)*4 -> (84,84,4) -> (4,84,84)
    train_vec = VecFrameStack(train_vec, n_stack=4)
    train_vec = VecTransposeImage(train_vec)
    train_vec = VecMonitor(train_vec)  # één monitor rond de hele vecenv

    # -------------------------
    # Model (auto-resume)
    # -------------------------
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        ckpt = latest_checkpoint(Path(args.save_dir))
        if ckpt is not None:
            print(f"[INFO] Hervatten vanaf checkpoint: {ckpt}")
            model = PPO.load(ckpt, env=train_vec, seed=args.seed, device=device)
        else:
            print("[INFO] Geen checkpoint gevonden, start nieuwe training.")

    if model is None:
        model = PPO(
            policy="CnnPolicy",
            env=train_vec,
            # Grotere rollouts → minder overhead, betere GPU-batching
            n_steps=128,         # was 256
            batch_size=32,      # pas aan als VRAM krap is (512/256)
            n_epochs=4,          # was 4
            learning_rate=lambda f: f * 2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.logdir,
            device=device,
        )
        # Optioneel: PyTorch 2 compile (kan extra GPU-snelheid geven)
        if hasattr(torch, "compile"):
            try:
                model.policy = torch.compile(model.policy, mode="max-autotune")
                print("[INFO] torch.compile geactiveerd")
            except Exception as e:
                print("[WARN] torch.compile uitgeschakeld:", e)

    # -------------------------
    # Checkpoints
    # -------------------------
    save_freq_env_steps = max(args.checkpoint_freq // args.n_envs, 1)
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq_env_steps,
        save_path=args.save_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # -------------------------
    # Train
    # -------------------------
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=1,
        tb_log_name=f"{args.game}_ppo",
        callback=ckpt_cb,
        reset_num_timesteps=not args.resume,
        progress_bar=False,
    )

    # Eindsave
    model.save(str(Path(args.save_dir) / "final_model"))


if __name__ == "__main__":
    main()


#SDL_VIDEODRIVER=dummy PYOPENGL_PLATFORM=egl DISPLAY= WAYLAND_DISPLAY= LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ulimit -n 8192; python3 test.py --game "StreetFighterIISpecialChampionEdition-Genesis" --logdir logs/sf2 --save-dir runs/sf2 --checkpoint-freq 50000 --n-envs 46 --frame-skip 4 --total-timesteps 10000000000 --resume
