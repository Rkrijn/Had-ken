import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import retro

from test import make_retro, wrap_deepmind_retro  # hergebruik wrappers
##python3 play.py --game="StreetFighterIISpecialChampionEdition-Genesis" \
##  --model runs/sf2/ckpt_2000000.zip
######

def make_play_env(game, state, scenario):
    def _thunk():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env
    vec = DummyVecEnv([_thunk])
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="StreetFighterIISpecialChampionEdition-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--model", required=True, help="Pad naar .zip (bv. runs/sf2/ckpt_2000000.zip)")
    args = parser.parse_args()

    env = make_play_env(args.game, args.state, args.scenario)
    model = PPO.load(args.model, env=env, device="auto")

    obs = env.reset()
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
