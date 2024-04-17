from foundations.supporting_functions import start_train, start_tuning
import time

config_param_filepath = 'user_config/configuration.yml'
eval_param_filepath = 'user_config/eval_hyperparams.yml'

data_filename = 'output_csv'
image_filename = 'output_plots' 

function = 'train'

if __name__ == "__main__":

    if function == 'tune':
        project_name = 'datasparq'
        num_runs = 10
        tune_param_filepath = 'user_config/tuning_hyperparams.yml'
        plot_best_param = False
        config_param_filepath = 'user_config/configuration.yml'
        eval_param_filepath = 'user_config/eval_hyperparams.yml'
        start_tuning(project_name, num_runs, tune_param_filepath, config_param_filepath, eval_param_filepath, plot_best_param)
    
    if function == 'train':
        start_time = time.time()
        start_train(config_param_filepath, eval_param_filepath, data_filename=data_filename, image_filename=image_filename)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The start_train function took {elapsed_time/60} minutes to execute.")