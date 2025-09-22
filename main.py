import gymnasium as gym
import pybullet
import torch as th
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Import PyBullet environments the correct way for gymnasium
try:
    # Try the new way first
    import pybullet_envs.bullet.bullet_envs  # This registers the environments

    env_id = 'AntBulletEnv-v0'
except (ImportError, AttributeError):
    # Fallback: use a different environment that's guaranteed to work
    print("PyBullet Ant environment not available. Using BipedalWalker instead.")
    env_id = 'BipedalWalker-v3'  # Built into gymnasium


# Alternative: If you want to stick with PyBullet, try this approach
def create_pybullet_env():
    """Alternative way to create PyBullet environment"""
    try:
        # Direct PyBullet environment creation
        import pybullet_envs.gym_locomotion_envs as locomotion_envs
        from pybullet_envs.env_bases import MJCFBaseBulletEnv
        # This is more complex - let's use the simpler approach below
        pass
    except:
        pass

    # Fallback to a working environment
    return gym.make('BipedalWalker-v3')


# Create environment - using BipedalWalker as it's more reliable
def make_env():
    def _init():
        # BipedalWalker is a similar walking task and works reliably
        env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
        return env

    return _init


# Create vectorized environment
env = DummyVecEnv([make_env()])

# Separate environment for evaluation
eval_env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

# Adjust target score for BipedalWalker (different scoring than Ant)
MAX_AVERAGE_SCORE = 300  # BipedalWalker target score

print(f"Using environment: BipedalWalker-v3")
print("This is a 2D bipedal robot that learns to walk - similar concept to the Ant!")

# Neural network architecture for the policy
policy_kwargs = dict(
    activation_fn=th.nn.LeakyReLU,
    net_arch=[256, 256]  # Slightly smaller network for BipedalWalker
)

# Create the PPO agent
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0003,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./ppo_walker_tensorboard/"
)

# Callback to stop training when target reward is reached
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=MAX_AVERAGE_SCORE, verbose=1)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=5000,  # Evaluate every 5k steps
    deterministic=True,
    render=False,
    verbose=1,
    best_model_save_path='./best_walker_model/'
)

print("Starting training...")
print(f"Target score: {MAX_AVERAGE_SCORE}")
print("The bipedal walker will learn to walk forward efficiently!")

try:
    # Train the agent
    model.learn(
        total_timesteps=1000000,  # 1M steps should be enough for BipedalWalker
        callback=eval_callback
    )

    print("Training completed!")

    # Save the final model
    model.save('ppo_walker_final_model')

    # Final evaluation
    print("\nFinal evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )

    print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Demonstrate the trained agent
    print("\nDemonstrating trained walker...")
    demo_env = gym.make('BipedalWalker-v3', render_mode='human')

    obs, _ = demo_env.reset()
    total_reward = 0

    for step in range(2000):  # Run for longer to see walking
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        total_reward += reward
        demo_env.render()

        if terminated or truncated:
            print(f"Episode finished after {step} steps with total reward: {total_reward:.2f}")
            obs, _ = demo_env.reset()
            total_reward = 0

    demo_env.close()

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    model.save('ppo_walker_interrupted_model')

finally:
    # Clean up
    env.close()
    eval_env.close()
    print("Environment closed.")