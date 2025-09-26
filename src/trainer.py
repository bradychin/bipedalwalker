# --------- Standard library imports ---------#
import os

# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO

# --------- Local imports ---------#
from utils.config import TRAINING, PATHS
from utils.functions import add_timestamp
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Callbacks ---------#
def create_training_callbacks(eval_env):
    # Callback to stop training when target reward is reached
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=TRAINING['target_score'],
                                                  verbose=1)
    # Evaluation callback
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=TRAINING['eval_freq'],
                                 deterministic=True,
                                 render=False,
                                 verbose=1,
                                 best_model_save_path=PATHS['best_model_path'])

    return eval_callback

# --------- Training ---------#
def train_agent(model, eval_env):
    logger.info(f'Training target: {TRAINING["target_score"]} reward.')
    logger.info(f'Max timesteps: {TRAINING["max_timesteps"]}')
    eval_callback = create_training_callbacks(eval_env)
    try:
        # Train the agent
        logger.info('Starting training...')
        model.learn(total_timesteps=TRAINING['max_timesteps'],
                    callback=eval_callback)
        logger.info('Training completed!')
    except KeyboardInterrupt:
        logger.warning('\nTraining interrupted by user.')
        logger.info('Saving interrupted model...')
        model.save(PATHS['tensorboard_log'])
    except Exception as e:
        logger.error(f'Training failed: {str(e)}')
        return

    best_model_path = os.path.join(PATHS['best_model_path'], 'best_model.zip')
    if os.path.exists(best_model_path):
        logger.info('Using best model...')
        best_model = PPO.load(best_model_path)
        add_timestamp(PATHS['best_model_path'], PATHS['tensorboard_log'])
        return best_model
    else:
        logger.warning('Best model not found. Using final training model')
        return model
