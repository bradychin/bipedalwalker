# --------- Third-party imports ---------#
from stable_baselines3.common.evaluation import evaluate_policy

# --------- Local imports ---------#
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Evaluation ---------#
def evaluate_agent(model, eval_env, n_episodes=10):
    logger.info("Final evaluation...")
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=n_episodes,
                                              deterministic=True)

    logger.info(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward