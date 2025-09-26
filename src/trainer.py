# --------- Import libraries ---------#
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO

# --------- Import scripts ---------#
from utils.config import TRAINING, PATHS
from utils.functions import add_timestamp

# --------- Training ---------#
def train_agent(model, eval_env):
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

    try:
        # Train the agent
        model.learn(total_timesteps=TRAINING['max_timesteps'],
                    callback=eval_callback)
        print("Training completed!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        model.save('ppo_walker_interrupted_model')

    best_model_path = os.path.join(PATHS['best_model_path'], 'best_model.zip')
    if os.path.exists(best_model_path):
        print('Loading best model...')
        best_model = PPO.load(best_model_path)
        add_timestamp(PATHS['best_model_path'], PATHS['tensorboard_log'])
        return best_model
    else:
        print('Best model not found. Returning final training model')
        return model
