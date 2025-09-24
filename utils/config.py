ENVIRONMENT = {
    'environment_id': 'BipedalWalker-v3',
    'render_mode_train': 'rgb_array',
}

TRAINING = {
    'target_score': 300,
    'max_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'policy_net': [256,256],
    'eval_freq': 5000
}

PATHS = {
    'tensorboard_log': './ppo_walker_tensorboard',
    'best_model_path': './best_walker_model'
}