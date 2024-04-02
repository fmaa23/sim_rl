from supporting_files.supporting_functions import start_train

config_param_filepath = 'user_config/configuration.yml'
eval_param_filepath = 'user_config/eval_hyperparams.yml'

data_filename = 'data'
image_filename = 'images' 

if __name__ == "__main__":
    start_train(config_param_filepath, eval_param_filepath, data_filename, image_filename)