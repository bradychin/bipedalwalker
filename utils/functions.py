# --------- Import libraries ---------#
import os
import shutil
from datetime import datetime

# --------- Timestamp function ---------#
def add_timestamp(best_model_path, tensorboard_path):
    timestamp = datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
    model_file = os.path.join(best_model_path, 'best_model.zip')

    if os.path.exists(model_file):
        timestamped_model = os.path.join(best_model_path, f'{timestamp}_best_model.zip')
        shutil.move(model_file, timestamped_model)

    if os.path.exists(tensorboard_path):
        timestamped_tb_dir = f'{timestamp}_{tensorboard_path}'
        shutil.move(tensorboard_path, timestamped_tb_dir)

    return timestamp