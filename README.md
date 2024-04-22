
# RL-Diven Queueing Network Simulation

This repository implements a Dyna-DDPG (Deep Deterministic Policy Gradient) Reinforcement Learning agent that optimizes routing probabilities to maximize end-to-end delay and throughput in a simulated queueing network.

## Project Structure

- `agents`: Contains the Dyna-DDPG agent implementation and allows the integration of new types of agents for exploring the simulated queueing environment.
- `queue_env`: Defines the simulated queueing environment, utilizing functionalities from the `queueing-tool` package.
- `rl_env`: Hosts the RL environment, which is portable and compatible with different agent types.
- `features`: Includes several utility features:
  - **Breakdown Exploration**: Explores key states versus peripheral states
  - **Blockage Demonstration**: Demonstrates how the agent responds to a server outage by adjusting routing probabilities.
  - **Confidence Evaluation**: Assesses the stability and reliability of the agent across different training setups
  - **Noise Evaluation**: Evaluate the effect of environmental noise on the performance of the agent
  - **Startup Behavior Visualization**: Identifies the burn-in period of the agent
  - **Num Runs**: Manages the number of simulation runs for training and evaluation

## Prerequisites

Before running the simulations, ensure you have the following installed:
- Python 3.8+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- queueing-tool
- wandb
- Ray

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://gitlab.doc.ic.ac.uk/jw923/MScDataSparqProject.git
cd MScDataSparqProject
pip install -r requirements.txt
```

## Step 1: Configuration

### Environment Setup

#### **Queueing Environment Configuration**

The simulation environment requires the following parameters to be defined in the `configuration.yml`.

- `adjacent_list`: A dictionary defining the adjacency list for the network topology.
- `miu_dict`: A dictionary of service rates for each service node in the network.
- `transition_proba_all`: A dictionary defining the transition probabilities between nodes.
- `active_cap`: The active capacity of the nodes from outside the network.
- `deactive_t`: The deactivation threshold for the nodes from outside the network.
- `buffer_size_for_each_queue`: A dictionary that defines the buffer size for each queue.
- `arrival_rate`: A list that defines the arrival rates for all possible entry nodes.
- `max_agents`: A value that defines the maximum number of agents can be accpeted from outside the network for the entry nodes.
- `sim_jobs`: A value that defines the number of jobs being simulated during every simulation.
- `max_arr_rate_list`: A list that defines the maximum arrival rate for all entry queues.
- `entry_nodes`: A list that defines the source and target vertices of each entry node.

   Example:
   ```yaml
   miu_list:  
   1: 0.250
   2: 0.25
   3: 0.01500
   4: 100
   5: 1.20
   6: 0.01000
   7: 10
   8: 0.1000
   9: 0.500

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

   buffer_size_for_each_queue: 
   0: 5000
   1: 5000
   2: 5000
   3: 5000
   4: 5000
   5: 5000
   6: 5000
   7: 5000
   8: 5000
   9: 5000
   10: 5000
   11: 5000
   12: 5000

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
   
   active_cap: 5

   deactive_t: 0.12

   arrival_rate: [0.3]

   max_agents: inf

   sim_jobs: 100

   max_arr_rate_list: [375, 375, 375]

   entry_nodes:
   - [0, 1] 
   ```

#### **RL Environment Parameters**

Set up the RL environment parameters in `rl_env_params.yml`:

- `num_episodes`: The number of episodes to run the simulation.
- `num_epochs`: The number of epochs for training.
- `time_steps`: The number of time steps in each episode.
- `batch_size`: Size of the batch used in training. (Default is equal to time_steps)
- `num_sim`: The number of simulations to run during training.
- `tau`: Coefficient for soft update of the target parameters.
- `actor_lr`: Learning rate for the Actor network optimizer.
- `critic_lr`: Learning rate for the Critic Network optimizer.
- `discount`: Discount factor for future rewards.
- `planning_steps`: The number of steps during planning.
- `planning_std`: Standard deviation of the normal disturbance during planning.
- `actor_network`: Network architecture for actor network.
- `critic_network`: Network architecture for critic network.
- `reward_model`: Network architecture for reward model usd in planning.
` `next_state_model`: Network architecture for next state model used in planning. 

   Example:
   ```yaml
   num_episodes: 5

   threshold: 10

   num_epochs: 100

   time_steps: 30

   batch_size: 30
   
   target_update_frequency: 100
   
   buffer_size: 10000
   
   num_sim: 10
   
   tau: 0.5

   num_train_AC: 10
   
   critic_lr: 0.01

   actor_lr: 0.0001
   
   discount: 0.8
   
   planning_steps: 10
   
   planning_std: 0.1

   actor_network:
   - 64
   - 64
   - 64

   critic:
   - 64
   - 64
   - 64

   reward_model:
   - 32
   - 64
   - 64
   - 32

   next_state_model:
   - 32
   - 64
   - 64
   - 32
   ```

#### **Tuning Configuration**
Set up the hyperparameter tuning ranges in `tuning_params.yml`:

- `lr_min/max`: Min and max ranges of the learning rate being tuned.
- `epochs_list`: A list that defines the range of possible epochs to train reward model and next state model.
- `batch_size`: A list that defines the range of batch sizes to sample from the replay buffer.
- `tau_min/max`: Min and max ranges of the soft update parameters.
- `discount min/max`: Min and max ranges of the discount factor for future rewards.
- `epsilon_min/max`: Min and max ranges of the standard deviation of normal disturbances during planning.
- `planning_steps`: A list that defines possible steps for planning.
- `w1/w2`: Weight parameters that influence the exploration between key and peripheral states.
- `num_episodes`: A list that defines the possible numbers of episodes to train the agents.
- `time_steps`: A list that defines the possible number of time steps during each episode

   Example:
   ```yaml
   lr_max: 0.1
   lr_min: 0.001

   epochs_list:
   - 10
   - 10
   - 10

   batch_size:
   - 16
   - 32
   - 64

   tau_min: 0.0005
   tau_max: 0.002

   discount_min: 0.1
   discount_max: 0.3

   epsilon_min: 0.1
   epsilon_max: 0.3

   planning_steps: 
   - 10

   num_sample: 
   - 50

   w1: 
   - 0.5

   w2: 
   - 0.5

   num_episodes: 
   - 5

   time_steps: 
   - 10
   ```

## Step 2: Running Simulations

### Training Agent
This command starts training the agent within the simulated queueing environment. Results are saved in `/foundations/output_csv` and `/foundations/output_plots`.

```bash
python main_eval.py --config /user_config/queueing_configuration.yaml --params /user_config/eval_hyperparameters.yaml
```

### Hyperparameter Tuning

Below provides users two types of tuning strategies that feature different functionalities.

#### **Wandb Tuning**

A machine learning development platform that allows users to track and visualize varou aspects of their model training process in real-time, including loss and accuracy charts, parameter distributions, gradient histograms and system metrics. To run wandb:

```bash
python main_tune.py --project_name DataSparq_Project --num_runs 100 --tune_param_filepath /user_config/tuning_hyperparams.yaml --plot_best_param True --param_filepath /user_config/eval_hyperparameters.yaml
```

#### **Ray Tuning**

An industry standard tool for distributed hyperparameter tuning which integrates with TensorBoard and extensive analysis libraries. It also allows users to leverage cutting edge optimization algorithms at scale, including Bayesian Optimization, Population Based Training and HyperBand. To run ray tuning:

```bash
python main_tune.py --project_name DataSparq_Project --num_runs 100 --tune_param_filepath /user_config/tuning_hyperparams.yaml --tuner ray --param_filepath /user_config/eval_hyperparameters.yaml
```

## Step 3: Explore Features

### 1. **Explore Breakdown Scenarios**

This feature allows the user to train the agent based on customed exploration preferences between key states and peripheral states using weight parameter `w1_key` and `w2_peripheral`. The purpose of this feature is to enable the agent to not only generate high rewards for key states but also visit all breadown scenarios sufficiently enough. 

Set up the parameters in `user_config\features_params\bloackage_explore_params.yml`:

- `w1_key`: Weight parameter to control favor exploring key states.
- `w2_peripheral`: Weight parameter to control favor exploring peripheral states.
- `reset`: A bool value that controls whether to reset weight parameters during training.
- `reset_frequency`: A value that defines the number of episodes frequency to reset the weight parameters.
- `num_output`: A value that defines the number of top and least reward/visits states to plot in a histogram
- `output_json`: A bool value that determines whether to output the json file of key states and peripheral states
- `output_histogram`: A bool value that determines whether to output the histogram that shows the rewards and visits of the top and least states.
- `output_coverage_metric`: A bool value that determines whether to output the current coverage metric.

   Example:
   ```yaml
   w1: 0.5

   w2: 0.5

   reset: False

   reset_frequency: 2

   num_output: 5

   output_json: True

   output_histogram: False

   output_coverage_metric: True
   ```

To run this feature, navigate to `/features/breakdown_exploration` and run:
   ```bash
   python breakdown_exploration.py
   ```

### 2. **Blockage Demonstrations**

This feature allows the user to test a trained agent's performance on a simulated server blockage queueing environment by visualizing the changes in transition probabilities. The purpose of this feature is to show how effectice the tranied agent is acting on breakdown cases. 

Set up the parameters in `user_config\features_params\blockage_demonstration_params.yml`:

- `num_sim`: Defines the number of jobs to simulate for each time step during training.
- `time_steps`: Defines the number of time steps to perform for each episode.
- `queue_index`: Defines the queue index that record the metrics for.
- `metric`: Defines the metric to be reported for the selected queue.

   Example:
   ```yaml
   num_sim: 100

   time_steps: 100

   queue_index: 2

   metric: throughput
   ```

To use this feature, navigate to `/features/blockage_demonstration` and run:
   ```bash
   python demonstrations.py
   ```

### 3. **Startup Behavior Identification**

This feature allows the user to visualize when the burn-in periods end on the learning curve. 

Set up the parameters in `user_config\features_params\startup_behavior_params.yml`:

- `window_size`: 
- `threshold`:
- `consecutive_points`:

   Example:
   ```yaml
   window_size: 5

   threshold: 0.01

   consecutive_points: 5
   ```

To perform the feature, navigate to `/features/startup_behavior` and run:
   ```bash
   python startup.py
   ```

### 4. **Confidence Evaluation** (need clarification from Josh)

This feature allows the user ______________________ (need clarification about the functionality of this feature)

Set up the parameters in `user_config\features_params\startup_behavior_params.yml`:
(need explanation for each parameter)

- `num_episodes_list`: 
- `timesteps`:

   Example:
   ```yaml
   num_episodes: 
   - 100
   - 200
   - 300
   - 400
   - 500
   - 600
   - 700
   - 800
   - 900
   - 1000

   timesteps: 1000
   ```

To run this feature, navigate to `/features/confidence_evaluation` and run:
   ```bash
   python confidence.py
   ```

### 5. **Num Runs** (need clarification from Fatima)

This feature allows the user (need clarification of the functionality of this feature).

Set up the parameters in `user_config\features_params\startup_behavior_params.yml`:
(fillout the explanation for each parameter)

- `confidence_level`: 
- `desired_error`:
- `num_runs`:
- `time_steps`:
- `num_sim`:

   Example:
   ```yaml
   confidence_level: 0.95

   desired_error: 1

   num_runs: 10

   time_steps: 100

   nums_sim: 100
   ```

To run this feature, navigate to `/features/num_runs` and run:
   ```bash
   python runs.py
   ```

### 6. **Noise Evaluation** (Need clarification from Jevon)

This feature allows the user to evaluate the effect of environmental noise on the performance of the agent. (how is env noise defined here?) 

Set up the parameters in `user_config\features_params\startup_behavior_params.yml`:
(complete explanation for each parameter):

- `param_1`:
- `param_2`:

To run the feature, navigate to `/features/noise_evaluation` and run:
   ```bash
   python noise_evaluation.py
   ```

## Contribution

Contributions are welcome. Please create a pull request or issue to discuss proposed changes or report bugs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
