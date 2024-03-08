import argparse
from supporting_functions import start_train

# Setup argparse
parser = argparse.ArgumentParser(description='Start the training process with configuration and parameter files.')
parser.add_argument('--config', type=str, required=True, help='File path to the configuration YAML file.')
parser.add_argument('--params', type=str, required=True, help='File path to the hyperparameters YAML file.')

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    start_train(args.config, args.params)