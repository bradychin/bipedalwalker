# --------- Import Libraries ---------#
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

from utils.config import ENVIRONMENT

# --------- Environment functions ---------#
def create_env(render_mode='rgb_array'):
    return gym.make(ENVIRONMENT['environment_id'], render_mode=render_mode)

def make_vec_env(n_envs=1):
    """Create vectorized environment using stable-baselines3 utility"""
    return sb3_make_vec_env(ENVIRONMENT['environment_id'],
                            n_envs=n_envs,
                            env_kwargs={'render_mode': 'rgb_array'})