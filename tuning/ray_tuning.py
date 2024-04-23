import os
from tqdm import tqdm
# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


from agents.ddpg_agent import DDPGAgent
from rl_env.RL_Environment import RLEnv
import torch
import numpy as np
import wandb
import yaml
import os
import random 
from foundations.core_functions import *
from queue_env.queue_base_functions import * 

import ray
from ray import train as train_ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import argparse

global rewards_list
rewards_list = []

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
            },
            'tau': {
                'min': tune_params['tau_min'],
                'max': tune_params['tau_max']
            },
            'discount':{
                'min':tune_params['discount_min'],
                'max':tune_params['discount_max']
            },
            'epsilon':{
                'min': tune_params['epsilon_min'],
                'max': tune_params['epsilon_max']
            },
            'planning_steps': {
                'values': tune_params['planning_steps']
            },
            'num_sample': {
                'values': tune_params['num_sample']
            },
            'w1': {
                'values': tune_params['w1']
            },
            'w2': {
                'values': tune_params['w2']
            },
            'epsilon_state_exploration':{
                'values': tune_params['epsilon_state_exploration']
            },
            'num_episodes':{
                'values': tune_params['num_episodes']
            },
            'time_steps':{
                'values':tune_params['time_steps']
            }
        }
    }

    return config

def get_agent_parameters(config):
    params = {}
    params['tau'] = config['tau']
    params['learning_rate'] = config['learning_rate']
    params['discount'] = config['discount']
    params['epsilon'] = config['epsilon']
    params['planning_steps'] = config['planning_steps']

    return params

def train(config, 
          eval_param_filepath='user_config/eval_hyperparams.yml'): 
     
    environment = create_simulation_env(params={'num_sim':5000}, config_file='user_config/configuration.yml')
   
    n_states = len(environment.get_state())
    n_actions = environment.net.num_nodes - environment.num_nullnodes

    agent_params = get_agent_parameters(config)
    _, hidden = load_hyperparams(eval_param_filepath)
    
    agent = DDPGAgent(n_states, n_actions, hidden=hidden, params=agent_params)
    env=environment

    num_episodes = config['n_episodes']
    max_episode_length = config['max_episode_length']
    batch_size_buffer_sampling = config['batch_size_buffer_sampling']
    threshold = config['batch_size']
    batch_size = config['batch_size']
    epochs = config['epochs']
    target_update_frequency = config['target_update_frequency']
    
    reward_list = []
    action_dict = {}

    agent.train()
    for episode in range(num_episodes):        
        env.reset()
        state = env.explore_state(agent, env.qn_net, episode)
        t = 0
        while t < max_episode_length:
            if type(state) == np.ndarray:
                state = torch.from_numpy(state).to(device)
            action = agent.select_action(state).to(device)

            action_list = action.cpu().numpy().tolist()
            for index, value in enumerate(action_list):
                 node_list = action_dict.setdefault(index, [])
                 node_list.append(value)
                 action_dict[index] = node_list
                                          # line 5
            
            next_state = env.get_next_state(action)                       
            next_state = torch.tensor(next_state).float()
            reward = env.get_reward()
            rewards_list.append(reward)

            # Use tune.report to log the metric for hyperparameter optimization
            ray.train.report({"reward": reward})                              
            experience = (state, action, reward, next_state)
            agent.store_experience(experience)                                     

            if agent.buffer.current_size > batch_size:
                agent.fit_model(batch_size=batch_size, epochs=epochs)
                batch = agent.buffer.sample(batch_size=batch_size)
                agent.update_critic_network(batch)
                agent.update_actor_network(batch)
                agent.plan(batch)                                 


            t += 1
            state = next_state

            if t % target_update_frequency == 0:
                agent.soft_update(network="critic")
                agent.soft_update(network="actor")
        return {"reward": np.mean(np.array(rewards_list))}

def ray_tune():

    config = load_tuning_config(tune_param_filepath='user_config/tuning_hyperparams.yml')
    
    hyperparam_mutations = {
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "train_batch_size": lambda: random.randint(8,64),
        "n_episodes": lambda: random.randint(10, 50),
        "threshold":  lambda: random.randint(8, 64),
        "max_episode_length":  lambda: random.randint(8, 64),
        "batch_size_buffer_sampling": lambda: random.randint(8, 64),
        "batch_size": lambda: random.randint(8, 64),
        "epochs": lambda: random.randint(8, 64)
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25, 
        hyperparam_mutations=hyperparam_mutations, 
        custom_explore_fn=train,
    )

    param_space={

            "num_workers": 10, 
            "num_cpus": 1,  # number of CPUs to use per trial
            "num_gpus": 0,  # number of GPUs to use per trial

            # These params are tuned from a fixed starting value.
            "learning_rate": 1e-4,
            # These params start off randomly drawn from a set.
            "tau": tune.uniform(config['parameters']['tau']['min'],config['parameters']['tau']['max']),
            "discount" : tune.uniform(config['parameters']['discount']['min'],config['parameters']['discount']['max']),
            "epsilon":tune.uniform(config['parameters']['epsilon']['min'],config['parameters']['epsilon']['max']),
            "planning_steps":config['parameters']['planning_steps']['values'],
            "n_episodes": tune.choice(config['parameters']['num_episodes']['values']),
            "threshold": tune.choice(config['parameters']['batch_size']['values']),
            "max_episode_length": tune.choice(config['parameters']['time_steps']['values']),
            "batch_size_buffer_sampling":tune.choice(config['parameters']['batch_size']['values']),
            "batch_size":tune.choice(config['parameters']['batch_size']['values']),
            "epochs":tune.choice(config['parameters']['epochs']['values']),

        }
    tuner = tune.Tuner(
        tune.with_resources(train, resources={"gpu": 0.5, "cpu":0.5}),
        tune_config=tune.TuneConfig(
        metric="reward",
        mode="max",
        num_samples=10,
        scheduler=pbt,
        ),
        param_space=param_space,
        run_config=train_ray.RunConfig()
        )

    # Start Tuning
    results = tuner.fit()

    return results