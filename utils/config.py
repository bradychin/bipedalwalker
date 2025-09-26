ENVIRONMENT = {
    'environment_id': 'BipedalWalker-v3'
}

TRAINING = {
    'target_score': 300,
    'max_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'policy_net': [256,256],
    'eval_freq': 5_000
}

DEMO = {
    'max_steps': 2_000
}

PATHS = {
    'best_model_path': './models/best_models',
    'tensorboard_log': './models/tensorboards',
    'log_path': './log/app.log'
}

FORMAT = {
    'date_time': '%Y.%m.%d_%H-%M-%S'
}