
from supporting_functions import *

# start train

config_param_filepath = 'user_config/configuration.yml'
eval_param_filepath = 'user_config/eval_hyperparams.yml'

data_filename = 'data' # path to save CSVs
image_filename = 'images' # path to save plots

start_train(config_param_filepath, eval_param_filepath, data_filename, image_filename)