ENVIRONMENT = {
    'environment_id': 'BipedalWalker-v3',
    'render_mode_train': 'rgb_array',
    'render_mode_demo': 'human'
}

TRAINING = {
    'target_score': 300,
    'max_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'policy_net': [256,256],
    'tensorboard_log': './ppo_walker_tensorboard',
    'eval_freq': 5000,
    'best_model_path': './best_walker_model'
}