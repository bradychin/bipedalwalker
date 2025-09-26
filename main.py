# --------- Third-party imports ---------#
import torch as th
from stable_baselines3 import PPO

# --------- Local imports ---------#
from src.environment import make_vec_env, create_env
from src.trainer import train_agent
from src.evaluate import evaluate_agent
from src.demo import demo_agent
from utils.config import TRAINING, PATHS
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Main function ---------#
def main():
    logger.info('Starting PPO training pipeline.')
    # --------- Setup ---------#
    logger.info('Creating environment.')
    env = make_vec_env()
    eval_env = create_env(render_mode='rgb_array')
    logger.info('Environment created.')

    # Neural network architecture for the policy
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
                         net_arch=TRAINING['policy_net'])

    # Create the PPO agent
    model = PPO('MlpPolicy',
                env,
                learning_rate=TRAINING['learning_rate'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=PATHS['tensorboard_log'])

    # --------- Run ---------#
    # Training
    model = train_agent(model, eval_env)

    # Evaluate and demo
    evaluate_agent(model, eval_env)
    demo_agent(model)

    # Close environment
    env.close()
    eval_env.close()
    logger.info('Training pipeline completed successfully.')

if __name__ == '__main__':
    main()