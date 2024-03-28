import os

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


from .ddpg import DDPGAgent
from .RL_Environment import RLEnv
import torch
import numpy as np
import wandb
import yaml
import os
from .base_functions import *

def load_tuning_config(tune_param_filepath):

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one directory to the MScDataSparqProject directory
    project_dir = os.path.dirname(script_dir)

    # Build the path to the configuration file
    abs_file_path = os.path.join(project_dir, tune_param_filepath)

    with open(abs_file_path, 'r') as tune_params_file:
        tune_params = yaml.load(tune_params_file, Loader=yaml.FullLoader)

    config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': tune_params['learning_rate_min'],
                'max': tune_params['learning_rate_max']
            },
            'epochs': {
                'values': tune_params['epochs_list']
            },
            'batch_size': {
                'values': tune_params['batch_size']
            }
        }
    }

    return config

def init_wandb(project_name, tune_param_filepath, config_param_filepath, opt_target = 'reward', num_runs = 100):
    # initialize W&B
    wandb.login()

    # initial project
    wandb.init(project = project_name)

    # read hyperparameter files
    tuning_config = load_tuning_config(tune_param_filepath)
    env_config = load_config(config_param_filepath)
    sweep_id = wandb.sweep(tuning_config, project=project_name)


    def wandb_train():
        with wandb.init() as run:
            queue_env = create_queueing_env(config_param_filepath)
            env = create_RL_env(queue_env, env_config)

            config = run.config
            num_sample = config['num_sample']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            w1 = config['w1']
            w2 = config['w2']
            epsilon_state_exploration = config['epsilon_state_exploration']
            num_episodes = config['num_episodes']
            batch_size = config['batch_size']
            num_epochs = config['epochs_list']
            time_steps = config['time_steps']
            target_update_frequency = config['target_update_frequency']

            reward_list = []
            action_dict = {}

            n_states = len(env.get_state())
            n_actions = len(env.get_state()) - 2
            agent = DDPGAgent(n_states, n_actions, config.parameters)

            agent.train()
            for episode in range(num_episodes):
                env.reset()
                state = env.explore_state(agent, env, num_sample, device, w1, w2, epsilon_state_exploration)
                t = 0
                while t < time_steps:

                    if type(state) == np.ndarray:
                        state = torch.from_numpy(state).to(device)
                    action = agent.select_action(state).to(device)

                    action_list = action.cpu().numpy().tolist()
                    for index, value in enumerate(action_list):
                        node_list = action_dict.setdefault(index, [])
                        node_list.append(value)
                        action_dict[index] = node_list

                    next_state, _ = env.get_next_state(action)
                    next_state = torch.tensor(next_state).float().to(device)
                    reward = env.get_reward()

                    reward_list.append(reward)
                    experience = (state, action, reward, next_state)
                    agent.store_experience(experience)

                    if agent.buffer.current_size > batch_size:
                        agent.fit_model(batch_size=batch_size, threshold=batch_size, epochs=num_epochs)
                        batch = agent.buffer.sample(batch_size=batch_size)
                        agent.update_critic_network(batch)
                        agent.update_actor_network(batch)
                        agent.plan(batch)

                    t += 1
                    state = next_state

                    if t % target_update_frequency == 0:
                        agent.soft_update(network="critic")
                        agent.soft_update(network="actor")
                wandb.log({"episode": episode, opt_target: np.mean(reward_list)})
    wandb.agent(sweep_id, wandb_train, count=num_runs)

def get_best_param(project_name, opt_target = 'reward'):

    api = wandb.Api()
    sweep = api.sweep(project_name)

    # Get all runs in the sweep
    runs = sorted(sweep.runs,
                  key=lambda r: r.summary.get(opt_target, float("inf")),
                  reverse=True)  # Set reverse=False for metrics where lower is better

    # Retrieve the best run
    best_run = runs[0]

    print("Best Hyperparameters:")
    for key, value in best_run.config.items():
        print(f"{key}: {value}")

    best_hyperparameters = best_run.config

    return best_hyperparameters

if __name__ == "__main__":
    init_wandb(project_name = 'datasparq_sample_run', tune_param_filepath = 'file_path')