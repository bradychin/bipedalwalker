# --------- Import Libraries ---------#
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.config import ENVIRONMENT

# --------- Environment functions ---------#
def create_env(render_mode='rgb_array'):
    return gym.make(ENVIRONMENT['environment_id'], render_mode=render_mode)

def make_vec_env():
    return DummyVecEnv([lambda: create_env(render_mode='rgb_array')])