# --------- Local imports ---------#
from src.environment import create_env
from utils.config import DEMO
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Demonstration ---------#
def demo_agent(model, max_steps=DEMO['max_steps']):
    # Demonstrate the trained agent
    logger.info("Demonstrating trained walker...")
    demo_env = create_env(render_mode='human')

    try:
        obs, _ = demo_env.reset()
        total_reward = 0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            total_reward += reward
            demo_env.render()

            if terminated or truncated:
                logger.info(f"Episode finished after {step} steps with total reward: {total_reward:.2f}")
                obs, _ = demo_env.reset()
                total_reward = 0
    finally:
        demo_env.close()