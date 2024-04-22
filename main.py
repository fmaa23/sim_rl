def add_path_to_system():
    """
    Adds the root directory of the current script to the system path.

    This is particularly useful in situations where the script needs to import
    modules from parent directories not originally in the Python system path, and 
    a more flexible project structure without relying on the user to modify their system path manually.
    """
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))

import os

config_dir = 'user_config'

# Create the file paths using os.path.join
config_param_filepath = os.path.join(config_dir, 'configuration.yml')
eval_param_filepath = os.path.join(config_dir, 'eval_hyperparams.yml')



data_filename = 'output_csv'
image_filename = 'output_plots' 

if __name__ == "__main__":
    add_path_to_system()
    from foundations.supporting_functions import start_train

    start_train(config_param_filepath, eval_param_filepath,save_file = True, data_filename = data_filename, image_filename = image_filename )