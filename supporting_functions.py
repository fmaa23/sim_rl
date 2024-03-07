
import torch
import numpy as np 
import pandas as pd
import queueing_tool as qt 
import numpy as np

from RL_Environment import RLEnv
from ddpg import DDPGAgent
from State_Exploration import *
from queueing_network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_filepath):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    - config_filepath (str): The file path to the configuration YAML file.

    Returns:
    - dict: A dictionary containing the configuration parameters.
    """

    config_params = (open(config_filepath, 'r'))
    config_dictionary = yaml.load(config_params, Loader=yaml.FullLoader)
    return config_dictionary

def load_hyperparams(param_filepath):
    """
    Load hyperparameters from a YAML file.

    Parameters:
    - param_filepath (str): The file path to the hyperparameters YAML file.

    Returns:
    - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
    """

    params_file = (open(param_filepath, 'r'))
    parameter_dictionary = yaml.load(params_file, Loader=yaml.FullLoader)
    params = parameter_dictionary['params']
    hidden = parameter_dictionary['hidden']
    

    # since we do not have the above now, we will hard-code for now
    params = {
            'num_episodes': 10,
            'threshold': 64,
            'num_epochs':10,
            'time_steps':10,
            'target_update_frequency':100,
            'batch_size': 64,
            'num_sim': 5000,
            'tau': 0.001,
            'lr': 0.1,
            'discount': 0.2,
            'planning_steps': 10,
            'epsilon': 0.2,
            'epsilon_f': 0.1,
            "actor_lr":0.1,
            'num_sample': 50,
            'w1':0.5,
            'w2':0.5,
            'epsilon_state_exploration':1
            }
    
    hidden = {
        'actor': [32, 32],
        'critic': [64, 64],
        'reward_model': [64, 64],
        'next_state_model': [64, 64]
    }

    return params, hidden

def create_params(config_file):
    """
    Generate parameters for the queueing environment based on a configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Multiple return values including lists and dictionaries essential for creating the queueing environment.
    """
    config_params = load_config(config_file)

    if False:
        def ser_f(t):
            return t + np.exp(config_params['sef_rate_first_node'])

        def rate(t):

            return 25 + 350 * np.sin(np.pi * t / 2)**2

        def arr(t):
            return qt.poisson_random_measure(t, rate, 375)

        def get_service_time(miu_list):
            # compute the time of an agent’s service time from service rate
            services_f = []
            for miu in miu_list:
                def ser_f(t):
                    return t + np.exp(miu)
                services_f.append(ser_f)
            return services_f
        
        lamda_list = config_params['lambda_list']
        miu_list = config_params['miu_list']
        active_cap = config_params['active_cap']
        deactive_t = config_params['deactive_t']
        adjacent_list = config_params['adjacent_list']
        buffer_size_for_each_queue = config_params['buffer_size']
        transition_proba_all = config_params['transition_proba']
        services_f = get_service_time(miu_list)
        
        q_classes = config_params['q_classes']
        q_args = config_params['q_args']

        edge_list = config_params['edge_list']

    # hard code for now
    def ser_f(t):
        return t + np.exp(0.2 / 2.1)

    def rate(t):

        return 25 + 350 * np.sin(np.pi * t / 2)**2

    def arr(t):
        return qt.poisson_random_measure(t, rate, 375)

    def get_service_time(miu_list):
        # compute the time of an agent’s service time from service rate
        services_f = []
        for miu in miu_list:
            def ser_f(t):
                return t + np.exp(1.2)
            services_f.append(ser_f)
        return services_f
    lamda_list = [0.2, 0.2, 0.2, 0.2,0.2, 0.2, 0.2, 0.2]
    miu_list = [0.5, 1, 1.2, 0.2]
    active_cap = 5
    deactive_t = 0.12
    adjacent_list = {0: [1], 1:[2,3,4], 2:[5],3: [6,7],4:[8], 5:[9], 6:[9], 7:[9], 8:[9], 9:[10]}
    buffer_size_for_each_queue = [10, 10, 10, 10,10, 10, 10, 10, 10, 10, 10, 10]
    transition_proba_all = {0:{1:1}, 1:{2:0.33,3:0.33,4:0.34}, 2:{5:1}, 3:{6:0.5, 7:0.5},4:{8:1}, 5:{9:1}, 6:{9:1}, 7:{9:1}, 8:{9:1}, 9:{10:1}}
    services_f = get_service_time(miu_list)
    
    q_classes = {0: qt.NullQueue, 1: qt.LossQueue, 2: qt.LossQueue, 3:qt.LossQueue, 4:qt.LossQueue, 5:qt.LossQueue}
    q_args = {1: {
        'arrival_f': arr,
        'service_f': ser_f,
    },
    2:{
        'service_f': services_f[0],
        'qbuffer':20,
    },
    3:{
        'service_f': services_f[1],
        'qbuffer':20,
    },
    4:{
        'service_f': services_f[2],
        'qbuffer':20
    },
    5:{
        'service_f': services_f[3],
        'qbuffer':20
    }}

    edge_list = {0:{1:1}, 1: {k: 1 for k in range(2, 5)}, 2:{5:2}, 3:{6:3, 7:4},4:{8:5}, 5:{9:2}, 6:{9:4}, 7:{9:3}, 8:{9:5}, 9:{10:0}}
    
    return lamda_list, miu_list, q_classes, q_args, adjacent_list, edge_list, transition_proba_all

def create_queueing_env(config_file):
    """
    Create and configure a queueing environment based on a given configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Queue_network: An instance of the queueing environment.
    """
    lamda_list, miu_list, q_classes, q_args, \
        adjacent_lits, edge_list, transition_proba_all = create_params(config_file)
    
    q_net = Queue_network()
    q_net.process_input(lamda_list, miu_list, q_classes, q_args, adjacent_list, 
                        edge_list, transition_proba_all)
    q_net.create_env()
    return q_net

def create_RL_env(q_net, params, hidden):
    """
    Create a reinforcement learning environment.

    Parameters:
    - q_net (Queue_network): The queueing network environment.
    - params (dict): Parameters for the RL environment.
    - hidden (dict): Hidden layer configurations.

    Returns:
    - RLEnv: An instance of the RL environment.
    """
    env = RLEnv(q_net, n = params['num_sim'])
    return env

def create_simulation_env(params, hidden, config_file):
    """
    Create a simulation environment for reinforcement learning based on given parameters and a configuration file.

    Parameters:
    - params (dict): Hyperparameters for the simulation.
    - hidden (dict): Hidden layer configurations.
    - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".

    Returns:
    - RLEnv: The RL environment ready for simulation.
    """
    q_net = create_queueing_env(config_file)
    RL_env = create_RL_env(q_net, params, hidden)

    return RL_env

def get_param_for_state_exploration(params):
    """
    Extract parameters necessary for state exploration.

    Parameters:
    - params (dict): Hyperparameters including those needed for state exploration.

    Returns:
    - tuple: A tuple containing parameters specific to state exploration.
    """
    num_sample = params['num_sample']
    device = device
    w1 = params['w1']
    w2 = params['w2']
    epsilon = params['epsilon_state_exploration']

    return num_sample, device, w1, w2, epsilon

def get_params_for_train(params):
    """
    Extract parameters necessary for training.

    Parameters:
    - params (dict): Hyperparameters including those needed for training.

    Returns:
    - tuple: A tuple containing parameters specific to training.
    """
    num_episodes = params['num_episodes']
    threshold = params['threshold']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    time_steps = params['time_steps']
    target_update_frequency = params['target_update_frequency']

    return num_episodes, threshold, batch_size, num_epochs, time_steps, target_update_frequency

def train(params, agent, env):
    """
    Conduct training sessions for a given agent and environment.

    Parameters:
    - params (dict): Hyperparameters for training.
    - agent: The agent to be trained.
    - env: The environment in which the agent operates.

    Returns:
    - Multiple values including lists that track various metrics through training.
    """
    next_state_list_all = []
    rewards_list_all = [] 
    critic_loss_list_all = []
    actor_loss_list_all = []
    reward_list = []
    actor_gradient_list_all = []
    action_dict = {}

    num_sample, device, w1, w2, epsilon_state_exploration = get_param_for_state_exploration(params)
    num_episodes, threshold, batch_size, num_epochs, time_steps, target_update_frequency = get_params_for_train(params)

    agent.train()
    for episode in range(num_episodes):
        print(f"-----------------episode {episode}------------------------")
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
                             
            next_state, transition_probas = env.get_next_state(action)    
            next_state = torch.tensor(next_state).float().to(device)
            reward = env.get_reward()
     
            reward_list.append(reward)                               
            experience = (state, action, reward, next_state)        
            agent.store_experience(experience)                             
           
            actor_loss_list= []
            critic_loss_list = []
            actor_gradient_list = []

            if agent.buffer.current_size > threshold:

                reward_loss_list, next_state_loss_list = agent.fit_model(batch_size=threshold, threshold=threshold, epochs=num_epochs)
                next_state_list_all += next_state_loss_list
                rewards_list_all += reward_loss_list
                
                batch = agent.buffer.sample(batch_size=threshold)
                critic_loss = agent.update_critic_network(batch)                   
                actor_loss, gradient_dict = agent.update_actor_network(batch)              
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                agent.plan(batch)

            actor_loss_list_all += actor_loss_list
            critic_loss_list_all += critic_loss_list
            actor_gradient_list_all += actor_gradient_list
        
            t += 1
            state = next_state

            if t%target_update_frequency == 0:
                agent.soft_update(network="critic")
                agent.soft_update(network="actor")
    
    return rewards_list_all, next_state_list_all, critic_loss_list_all,\
          actor_loss_list_all, reward_list, action_dict, gradient_dict, transition_probas

def create_ddpg_agent(environment, params, hidden):
    """
    Create a DDPG (Deep Deterministic Policy Gradient) agent.

    Parameters:
    - environment: The environment in which the agent will be trained.
    - params (dict): Hyperparameters for the DDPG agent.
    - hidden (dict): Hidden layer configurations.

    Returns:
    - DDPGAgent: An instance of the DDPG agent.
    """
    n_states = environment.net.num_edges - 1
    n_actions = len(environment.get_state())-2
    agent = DDPGAgent(n_states, n_actions, hidden, params)
    return agent

def save_all(rewards_list_all, next_state_list_all, \
        critic_loss_list_all, actor_loss_list_all, \
        reward_list, action_dict, gradient_dict, \
        transition_probas):
    """
    Save all relevant data from the training process.

    Parameters:
    - rewards_list_all (list): All rewards obtained.
    - next_state_list_all (list): All next states encountered.
    - critic_loss_list_all (list): All critic loss values.
    - actor_loss_list_all (list): All actor loss values.
    - reward_list (list): List of rewards.
    - action_dict (dict): Dictionary of actions taken.
    - gradient_dict (dict): Dictionary of gradients.
    - transition_probas (list): List of transition probabilities.

    This function also saves data to various files for further analysis.
    """
    pd.DataFrame(reward_list).to_csv('reward_new.csv')
    pd.DataFrame(actor_loss_list_all).to_csv("actor_loss_new.csv")
    pd.DataFrame(critic_loss_list_all).to_csv("critic_loss_new.csv")
    pd.DataFrame(next_state_list_all).to_csv("next_state_model_loss_new.csv")
    pd.DataFrame(rewards_list_all).to_csv("reward_model_loss_new.csv")
    pd.DataFrame(action_dict).to_csv("action_dict.csv")
    pd.DataFrame(transition_probas).to_csv("transition_proba.csv")

    import json
    # Specify the filename
    filename = 'gradient_dict.json'

    # Write the dictionary to a file as JSON
    with open(filename, 'w') as f:
        json.dump(gradient_dict, f)

def start_train(config_file, param_file, save_file = True):
    """
    Start the training process for a reinforcement learning environment and agent.

    Parameters:
    - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".
    - param_file (str, optional): The file path to the hyperparameters file. Defaults to "hyperparameter_file.yaml".
    - save_file (bool, optional): Flag indicating whether to save the training results to files. Defaults to True.

    This function orchestrates the loading of configurations, creation of environments and agents, and the training process.
    """

    params, hidden = load_hyperparams(param_file)

    sim_environment = create_simulation_env(params, hidden, param_file)
    agent = create_ddpg_agent(sim_environment, params, hidden)

    rewards_list_all, next_state_list_all, \
    critic_loss_list_all, actor_loss_list_all, \
    reward_list, action_dict, gradient_dict, \
    transition_probas = train(params, agent, sim_environment)

    if save_file:
        save_all(rewards_list_all, next_state_list_all, \
        critic_loss_list_all, actor_loss_list_all, \
        reward_list, action_dict, gradient_dict, \
        transition_probas)
