from foundations.supporting_functions import start_train, start_tuning
import time

import os

config_dir = 'user_config'

# Create the file paths using os.path.join
config_param_filepath = os.path.join(config_dir, 'configuration.yml')
eval_param_filepath = os.path.join(config_dir, 'eval_hyperparams.yml')
tune_param_filepath = os.path.join(config_dir, 'tuning_hyperparams.yml')


data_filename = 'output_csv'
image_filename = 'output_plots' 

function = 'tune'

if __name__ == "__main__":

    if function == 'tune':
        project_name = 'datasparq'
        wandb_api_key='02bb2e4979e9df3d890f94a917a95344aae652b9'#replace with yours here
        num_runs = 10
        plot_best_param = False
        start_tuning(project_name, num_runs, tune_param_filepath, config_param_filepath, eval_param_filepath,wandb_api_key, plot_best_param)
    
    if function == 'train':
        start_time = time.time()
        start_train(config_param_filepath, eval_param_filepath, data_filename=data_filename, image_filename=image_filename)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The start_train function took {elapsed_time/60} minutes to execute.")