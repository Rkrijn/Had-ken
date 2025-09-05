import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import retro


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
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    DeepMind-achtige preprocessing:
      - WarpFrame: 84x84, grayscale
      - ClipRewardEnv: beloningen clippen op {-1, 0, +1}
      - Monitor: episode stats voor TensorBoard
    """
    env = WarpFrame(env)        # (84, 84, 1) HWC
    env = ClipRewardEnv(env)
    env = Monitor(env)          # log ep_rew_mean/length
    return env


def make_train_env(game, state, scenario):
    def _thunk():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env
    return _thunk


def make_eval_env(game, state, scenario):
    """
    Eval-omgeving moet IDENTIEK zijn qua preprocessing en shape:
    DummyVecEnv(1) -> VecFrameStack(4) -> VecTransposeImage (C,H,W)
    """
    def _thunk():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env

    vec = DummyVecEnv([_thunk])
    vec = VecFrameStack(vec, n_stack=4)     # (84,84,4)
    vec = VecTransposeImage(vec)            # (4,84,84)
    return vec


def latest_checkpoint(save_dir: Path):
    ckpts = sorted(Path(save_dir).glob("ckpt_*.zip"))
    return ckpts[-1] if ckpts else None
    
env = make_retro(game="StreetFighterIISpecialChampionEdition-Genesis", state=retro.State.DEFAULT)
env = wrap_deepmind_retro(env)
obs, info = env.reset()
for t in range(5000):
    action = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(action)
    if t % 50 == 0 or terminated or truncated:
        print(info.keys(), info)  # kijk welke velden er zijn (round, wins, health, stage, etc.)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
