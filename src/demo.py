# --------- Import scripts ---------#
from environment import create_env

# --------- Demonstration ---------#
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