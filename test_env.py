from multiprocessing_env import SubprocVecEnv
import torch

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class testEnv(gym.Env):
    envCounter = 0
    metadata = {'render.modes': ['human']}

    def __init__(self):
        testEnv.envCounter += 1
        print("envs: ", testEnv.envCounter)
        pass

    def step(self, action):
        print("step ", action, testEnv.envCounter)
        return 0, 0, 0, 0

    def reset(self):
        print("reset ", testEnv.envCounter)
        return 0

    def render(self, mode='human', close=False):
        pass


def make_env():
    def _thunk():
        env = testEnv()
        return env

    return _thunk


if __name__ == "__main__":
    """<h2>Use CUDA</h2>"""

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    """<h2>Create Environments</h2>"""

    num_envs = 16
    env_name = "CartPole-v0"

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state = envs.reset()
    print(state)
    action = torch.Tensor(16)
    next_state, reward, done, _ = envs.step(action.cpu().numpy())
