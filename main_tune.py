import argparse
from Supporting_files.supporting_functions import start_tuning

# Setup argparse
parser = argparse.ArgumentParser(description='Start the tuning process with configuration and hyperparameters files.')
parser.add_argument('--project_name', type=str, required=True, help='Project name for the tuning session.')
parser.add_argument('--num_runs', type=int, required=True, help='Number of runs for the tuning session.')
parser.add_argument('--tune_param_filepath', type=str, required=True, help='File path to the tuning hyperparameters YAML file.')
parser.add_argument('--plot_best_param', type=bool, default=True, help='Flag to plot best parameters after tuning.')
parser.add_argument('--param_filepath', type=str, default='', help='File path to save the best parameters YAML file (optional).')

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    # Adjust the function call to match the new signature
    start_tuning(args.project_name, args.num_runs, args.tune_param_filepath, args.plot_best_param, args.param_filepath)