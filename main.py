
def add_path_to_system():
    """
    Adds the root directory of the current script to the system path.

    This is particularly useful in situations where the script needs to import
    modules from parent directories not originally in the Python system path,
    facilitating a more flexible project structure without relying on the user
    to modify their environment variables or system path manually.
    """
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))

config_param_filepath = 'user_config/configuration.yml'
eval_param_filepath = 'user_config/eval_hyperparams.yml'

data_filename = 'data'
image_filename = 'images' 

if __name__ == "__main__":
    add_path_to_system()
    from Supporting_files.supporting_functions import start_train

    start_train(config_param_filepath, eval_param_filepath, data_filename, image_filename)