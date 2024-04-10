# Dyna-DDPG Simulation Environment

This document outlines the required parameters for configuring the queue network simulation environment. The simulation leverages a YAML configuration file to define various aspects of the queue network, including lambda lists, service rates, capacities, and queue classes, among others.

## Running the Simulation

### Prerequisites

- Python 3.8 or newer
- Install required Python packages listed in `requirements.txt`.

### Installation of Dependencies

Before running the simulation, you need to install the required Python packages. This can be done by navigating to the project directory in your terminal and running the following command:

```bash
pip install -r requirements.txt
```

### Instructions

#### Training Process
After installing the required dependencies, you can start the training process. Ensure you're still in the directory containing the train_model.py script and run the following command in your terminal:
```bash
python main_eval.py --config /user_config/queueing_configuration.yaml --params /user_config/eval_hyperparameters.yaml
```
This command starts the training process using the specified configuration and hyperparameter YAML files.

#### Tuning Process
To perform hyperparameter tuning, use the main_tune.py script. Execute the following command in the terminal, making sure you are in the directory containing main_tune.py:
```bash
python main_tune.py --project_name DataSparq_Project --num_runs 100 --tune_param_filepath /user_config/tuning_hyperparams.yaml --plot_best_param True --param_filepath /user_config/eval_hyperparameters.yaml
```
Replace YourProjectName with the name of your project, NumberOfRuns with the desired number of tuning iterations, and the file paths with the actual paths to your YAML files. The --plot_best_param flag is optional and can be set to True or False. If you wish to save the best parameters found during the tuning process, provide a path for --param_filepath.

## Queue Network Configuration Parameters Explanations

The simulation environment requires the following parameters to be defined in the configuration file:

- `lambda_list`: A list of arrival rates for each queue in the network.
- `miu_list`: A list of service rates for each service node in the network.
- `active_cap`: The active capacity of the nodes.
- `deactive_t`: The deactivation threshold for the nodes.
- `adjacent_list`: A dictionary defining the adjacency list for the network topology.
- `buffer_size_for_each_queue`: A list defining the buffer size for each queue.
- `transition_proba_all`: A dictionary defining the transition probabilities between nodes.
- `q_classes`: A dictionary specifying the class for each queue.
- `q_args`: A dictionary containing arguments for the queue classes, such as service and arrival functions.
- `edge_list`: A dictionary defining the edges of the network and their properties.

### Example Configuration

```yaml
lambda_list: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
miu_list: [0.5, 1, 1.2, 0.2]
active_cap: 5
deactive_t: 0.12
adjacent_list:
  0: [1]
  1: [2, 3, 4]
  2: [5]
  3: [6, 7]
  4: [8]
  5: [9]
  6: [9]
  7: [9]
  8: [9]
  9: [10]
buffer_size_for_each_queue: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
transition_proba_all:
  0: {1: 1}
  1: {2: 0.33, 3: 0.33, 4: 0.34}
  2: {5: 1}
  3: {6: 0.5, 7: 0.5}
  4: {8: 1}
  5: {9: 1}
  6: {9: 1}
  7: {9: 1}
  8: {9: 1}
  9: {10: 1}
```

## Training Hyperparameters Configuration

The simulation environment also requires specific hyperparameters to be defined for effective learning and simulation. These parameters are defined in a separate file referred to by param_filepath. Below is a detailed explanation of each hyperparameter:

- `num_episodes`: The number of episodes to run the simulation.
- `threshold`: Threshold value for a certain performance metric.
- `num_epochs`: The number of epochs for training.
- `time_steps`: The number of time steps in each episode.
- `target_update_frequency`: Frequency of target model updates.
- `batch_size`: Size of the batch used in training.
- `num_sim`: The number of simulations to run.
- `tau`: Coefficient for soft update of the target parameters.
- `lr`: Learning rate for the optimizer.
- `discount`: Discount factor for future rewards.
- `planning_steps`: The number of planning steps.
- `epsilon`: Initial value for epsilon in the epsilon-greedy strategy.
- `epsilon_f`: Final value for epsilon in the epsilon-greedy strategy after decay.
- `actor_lr`: Learning rate for the actor model.
- `num_sample:` The number of samples for something specific in the simulation.
- `w1`: Weight parameter 1.
- `w2`: Weight parameter 2.
- `epsilon_state_exploration`: Epsilon value for state exploration.

### Example Configuration
```yaml
num_episodes: 10
threshold: 64
num_epochs: 10
time_steps: 10
target_update_frequency: 100
batch_size: 64
num_sim: 5000
tau: 0.001
lr: 0.1
discount: 0.2
planning_steps: 10
epsilon: 0.2
epsilon_f: 0.1
actor_lr: 0.1
num_sample: 50
w1: 0.5
w2: 0.5
epsilon_state_exploration: 1
```

## Network Hidden Layers Configuration

The structure and complexity of the neural networks used in the simulation are defined through hidden layer configurations. Each key represents a different component of the model, with the array defining the size of the hidden layers.

- `actor`: Defines the hidden layers for the actor model [32, 32].
- `critic`: Defines the hidden layers for the critic model [64, 64].
- `reward_model`: Defines the hidden layers for the reward model [64, 64].
- `next_state_model`: Defines the hidden layers for the next state - prediction model [64, 64].

Example Configuration for Hyperparameters and Hidden Layers

Ensure to separate the YAML content with a blank line and indent your YAML block to align with the list content properly.

### Example Configuration

```yaml
actor: [32, 32]
critic: [64, 64]
reward_model: [64, 64]
next_state_model: [64, 64]
```

## Important Features

### State Exploration Feature Introduction

Upon initialization, the `ExploreStateEngine` class loads parameters for exploring states from a YAML file and activates the following features based on the instructions entered by users:

##### 1. Output JSON Files (`output_json_files`)

When enabled, this feature allows the engine to output the rankings of states to JSON files. These files include information about both key states and peripheral states, categorized based on their reward rankings and visit rankings. The output is useful for tracking the performance and decisions of the exploration engine over time.

##### 2. Reset Mechanism (`reset`)

This feature introduces a reset mechanism that, if activated, allows the exploration engine to reset weights. This can be crucial for long-running exploration tasks, where periodic adjustments to exploration parameters can help in navigating different phases of the learning process more effectively.

##### 3. Output Histograms (`output_histogram`)

The `output_histogram` feature, when activated, instructs the engine to generate and save histograms of Q-values and visit counts for the top and least states. These visualizations are valuable for analyzing the distribution and trends of Q-values and visit frequencies, providing insights into the exploration process and the state space's dynamics.

##### 4. Output Coverage Metric (`output_coverage_metric`)

Finally, enabling the `output_coverage_metric` feature allows the engine to calculate and output a coverage metric, which quantifies the diversity and thoroughness of the state space exploration. This metric is derived from the moving average of rewards between the top and least states, offering a measure of how well the exploration strategy covers the potential state space.

Each of these features contributes to a more informative, adaptable, and effective state exploration strategy, enabling the `ExploreStateEngine` to navigate complex state spaces with enhanced efficiency and insight.

#### Example Configuration Under eval_hyperparams.yml
```yaml
state_exploration_param:

  num_sample: 50

  w1: 0.5

  w2: 0.5

  epsilon_state_exploration: 1


  reset_frequency: 2

  num_output: 5

  moa_window: 5

  reset: True

  output_json: True

  output_histogram: True

  output_coverage_metric: True
```