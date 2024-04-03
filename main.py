import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

from Supporting_files.supporting_functions import start_train

config_param_filepath = 'user_config/configuration.yml'
eval_param_filepath = 'user_config/eval_hyperparams.yml'

data_filename = 'data'
image_filename = 'images' 

if __name__ == "__main__":
    start_train(config_param_filepath, eval_param_filepath, data_filename, image_filename)