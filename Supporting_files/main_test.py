from supporting_functions import start_tuning  

project_name = 'datasparq'
num_runs = 10
tune_param_filepath = 'user_config/tuning_hyperparams.yml'
plot_best_param = True
config_param_filepath = 'user_config/configuration.yml'
eval_param_filepath = 'user_config/eval_hyperparams.yml'

start_tuning(project_name, num_runs, tune_param_filepath, config_param_filepath, eval_param_filepath, plot_best_param)