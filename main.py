# --------- Import libraries ---------#
import torch as th
from stable_baselines3 import PPO

# --------- Import scripts ---------#
from src.environment import make_vec_env, create_env
from src.trainer import train_agent
from src.evaluate import evaluate_agent
from src.demo import demo_agent
from utils.config import ENVIRONMENT, TRAINING

# --------- Main function ---------#
def main():
    # --------- Setup ---------#
    env = make_vec_env()
    eval_env = create_env(render_mode='rgb_array')

    # Neural network architecture for the policy
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
                         net_arch=TRAINING['policy_net'])

    # Create the PPO agent
    model = PPO('MlpPolicy',
                env,
                learning_rate=TRAINING['learning_rate'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=TRAINING['tensorboard_log'])

    # --------- Run ---------#
    print(f'Using environment {ENVIRONMENT["environment_id"]}')
    model = train_agent(model, eval_env)
    evaluate_agent(model, eval_env)
    demo_agent(model)

    env.close()
    eval_env.close()
    print('done')

if __name__ == '__main__':
    main()















