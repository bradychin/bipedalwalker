ENVIRONMENT = {
    'environment_id': 'BipedalWalker-v3'
}

TRAINING = {
    'target_score': 300,
    'max_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'policy_net': [256,256],
    'eval_freq': 5000
}

PATHS = {
    'tensorboard_log': './models/tensorboards',
    'best_model_path': './models/best_models'
}