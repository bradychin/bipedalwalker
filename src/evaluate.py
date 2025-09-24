# --------- Import Libraries ---------#
from stable_baselines3.common.evaluation import evaluate_policy

# --------- Evaluation ---------#
def evaluate_agent(model, eval_env, n_episodes=10):
    print("\nFinal evaluation...")
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=n_episodes,
                                              deterministic=True)

    print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward