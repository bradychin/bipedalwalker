import gymnasium as gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from utils.config import ENVIRONMENT, TRAINING

# -------------------------------- Environment setup -------------------------------- #
def create_env(render_mode='rgb_array'):
    return gym.make(ENVIRONMENT['environment_id'], render_mode=render_mode)
# Create vectorized environment
env = DummyVecEnv([lambda: create_env(render_mode='rgb_array')])

# Separate environment for evaluation
eval_env = create_env(render_mode='rgb_array')

# -------------------------------- Training configuration -------------------------------- #
print(f"Using environment: BipedalWalker-v3")
print("This is a 2D bipedal robot that learns to walk.")

# Neural network architecture for the policy
policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
                     net_arch=TRAINING['policy_net'])

# -------------------------------- Agent / Policy creation -------------------------------- #
# Create the PPO agent
model = PPO('MlpPolicy',
            env,
            learning_rate=TRAINING['learning_rate'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=TRAINING['tensorboard_log'])

# -------------------------------- Training -------------------------------- #
print("Starting training...")
print(f"Target score: {TRAINING['target_score']}")
print("The bipedal walker will learn to walk forward efficiently!")

def train_agent(model, eval_env, max_timesteps, target_score):
    # Callback to stop training when target reward is reached
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=target_score, verbose=1)
    # Evaluation callback
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=TRAINING['eval_freq'],
                                 deterministic=True,
                                 render=False,
                                 verbose=1,
                                 best_model_save_path=TRAINING['best_model_path'])

    try:
        # Train the agent
        model.learn(total_timesteps=max_timesteps,  # 1M steps should be enough for BipedalWalker
                    callback=eval_callback)
        print("Training completed!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        model.save('ppo_walker_interrupted_model')

    return model

# -------------------------------- Evaluation -------------------------------- #
def evaluate_agent(model, eval_env, n_episodes=10):
    print("\nFinal evaluation...")
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=n_episodes,
                                              deterministic=True)

    print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward

# -------------------------------- Demonstration -------------------------------- #
def demo_agent(model, max_steps=2000):
    # Demonstrate the trained agent
    print("\nDemonstrating trained walker...")
    demo_env = create_env(render_mode='human')
    obs, _ = demo_env.reset()
    total_reward = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        total_reward += reward
        demo_env.render()

        if terminated or truncated:
            print(f"Episode finished after {step} steps with total reward: {total_reward:.2f}")
            obs, _ = demo_env.reset()
            total_reward = 0

    demo_env.close()

# -------------------------------- Demonstration -------------------------------- #
if __name__ == '__main__':
    model = train_agent(model, eval_env, TRAINING['max_timesteps'], TRAINING['target_score'])
    evaluate_agent(model, eval_env)
    demo_agent(model)

    env.close()
    eval_env.close()
    print('done')



















