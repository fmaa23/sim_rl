
import torch
import numpy as np 
import pandas as pd
import queueing_tool as qt 
import numpy as np
import os

from RL_Environment import RLEnv

from ddpg import DDPGAgent
from State_Exploration import *
from queueing_network import *
from wandb_tuning import *
from plot_datasparq import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(env_param_filepath):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    - config_filepath (str): The file path to the configuration YAML file.

    Returns:
    - dict: A dictionary containing the configuration parameters.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one directory to the MScDataSparqProject directory
    project_dir = os.path.dirname(script_dir)

    # Build the path to the configuration file
    abs_file_path = os.path.join(project_dir, env_param_filepath)

    with open(abs_file_path, 'r') as env_param_file:
        config_params = yaml.load(env_param_file, Loader=yaml.FullLoader)

    return config_params

def load_hyperparams(eval_param_filepath):
    """
    Load hyperparameters from a YAML file.

    Parameters:
    - param_filepath (str): The file path to the hyperparameters YAML file.

    Returns:
    - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one directory to the MScDataSparqProject directory
    project_dir = os.path.dirname(script_dir)

    # Build the path to the configuration file
    abs_file_path = os.path.join(project_dir, eval_param_filepath)
    
    with open(abs_file_path, 'r') as env_param_file:
        parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)
    params = parameter_dictionary['params']
    hidden = parameter_dictionary['hidden']

    return params, hidden

def create_queueing_env(config_file):
    """
    Create and configure a queueing environment based on a given configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Queue_network: An instance of the queueing environment.
    """
    arrival_rate, miu_list, q_classes, q_args, \
        adjacent_list, edge_list, transition_proba_all = create_params(config_file)
    
    q_net = Queue_network()
    q_net.process_input(arrival_rate, miu_list, q_classes, q_args, adjacent_list, 
                        edge_list, transition_proba_all)
    q_net.create_env()
    return q_net

def get_num_connections(adjacent_list):
    # did not account for Nullqueue
    num_connection = 0
    exit_nodes = []
    for start_node in adjacent_list.keys():
        end_node_list = adjacent_list[start_node]
        for end_node in end_node_list:
            if end_node not in list(adjacent_list.keys()):
                exit_nodes.append(end_node)
            num_connection += 1
    
    return num_connection, exit_nodes

def make_edge_list(adjacent_list, exit_nodes):
    edge_list = {}
    edge_type = 1
    for start_node in adjacent_list.keys():
        end_node_list = adjacent_list[start_node]
        
        connection_dict = {}
        for end_node in end_node_list:
            if end_node not in exit_nodes:
                connection_dict[end_node] = edge_type
                edge_type += 1
            else:
                connection_dict[end_node] = 0
        
        edge_list[start_node] = connection_dict
    
    return edge_list

def get_connection_info(adjacent_list):
    connection_info = {}
    for start_node in adjacent_list.keys():
        for end_node in adjacent_list[start_node]:
            connect_start_node_list = connection_info.setdefault(end_node, [])
            connect_start_node_list.append(start_node)
            connection_info[end_node] = connect_start_node_list
    
    return connection_info


def make_unique_edge_type(adjacent_list, edge_list):
    # dictionary where keys are node, values are the edge type
    connection_info = get_connection_info(adjacent_list)
    edge_type_info = {}
    for end_node in connection_info.keys():
        start_node_list = connection_info[end_node]
        edge_type_list = []
        for start_node in start_node_list:
            edge_type = edge_list[start_node][end_node]
            edge_type_list.append(edge_type)
        edge_type_info[end_node] = edge_type_list
    
    return edge_type_info


def create_params(config_file):
    """
    Generate parameters for the queueing environment based on a configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Multiple return values including lists and dictionaries essential for creating the queueing environment.
    """
    
    config_params = load_config(config_file)

    miu_dict = config_params['miu_list']
    adjacent_list = config_params['adjacent_list']
    num_connections, exit_nodes = get_num_connections(adjacent_list)
    q_classes = create_q_classes(num_connections)
    edge_list = make_edge_list(adjacent_list, exit_nodes) 
    edge_type_info = make_unique_edge_type(adjacent_list, edge_list)

    buffer_size_for_each_queue = config_params['buffer_size_for_each_queue']
    q_args = create_q_args(edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes)

    arrival_rate = config_params['arrival_rate']
    
    transition_proba_all = config_params['transition_proba_all']

    # active_cap = config_params['active_cap']
    # deactive_t = config_params['deactive_cap']

    return arrival_rate, miu_dict, q_classes, q_args, adjacent_list, edge_list, transition_proba_all

def create_q_classes(num_queues):

    q_classes = {}
    q_classes[0] = qt.NullQueue
    for i in range(1, num_queues + 1):
        q_classes[i] = qt.LossQueue
    return q_classes

def create_q_args(edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes):

    def rate(t):
        return 25 + 350 * np.sin(np.pi * t / 2)**2

    def arr(t):
        return qt.poisson_random_measure(t, rate, config_params['arrival_rate'])

    def get_service_time(edge_type_info, miu_dict, exit_nodes):
        # need to make into a user-prompted way to ask for service rate where each end node corresponds to a distinctive service rate
        # the convert into each edge correponds to a distinctive service rate
        # save for buffer size
        services_f = {}
        for end_node in edge_type_info.keys():
            if end_node not in exit_nodes:
                service_rate = miu_dict[end_node]

                for edge_type in edge_type_info[end_node]:
                    def ser_f(t):
                        return t + np.exp(service_rate)
                    
                    services_f[edge_type] = ser_f

        return services_f

    services_f = get_service_time(edge_type_info, miu_dict, exit_nodes)
    
    q_args = {}
    for end_node in edge_type_info.keys():
        corresponding_edge_types = edge_type_info[end_node]
        for edge_type in corresponding_edge_types:

            if edge_type != 0:
                if edge_type == 1:
                    q_args[edge_type] = {
                    'arrival_f': arr,
                    'service_f': services_f[edge_type],
                    'qbuffer': buffer_size_for_each_queue[edge_type]
                    }
                else:
                    q_args[edge_type] = {
                    'service_f': services_f[edge_type],
                    'qbuffer':buffer_size_for_each_queue[edge_type],
                    }
    
    return q_args

def create_RL_env(q_net, params):
    """
    Create a reinforcement learning environment.

    Parameters:
    - q_net (Queue_network): The queueing network environment.
    - params (dict): Parameters for the RL environment.
    - hidden (dict): Hidden layer configurations.

    Returns:
    - RLEnv: An instance of the RL environment.
    """
    env = RLEnv(q_net, num_sim = params['num_sim'])
    return env

def create_simulation_env(params, config_file):
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
    RL_env = create_RL_env(q_net, params)

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
    device_here = device
    w1 = params['w1']
    w2 = params['w2']
    epsilon = params['epsilon_state_exploration']

    return num_sample, device_here, w1, w2, epsilon

def get_params_for_train(params):
    """
    Extract parameters necessary for training.

    Parameters:
    - params (dict): Hyperparameters including those needed for training.

    Returns:
    - tuple: A tuple containing parameters specific to training.
    """
    num_episodes = params['num_episodes']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    time_steps = params['time_steps']
    target_update_frequency = params['target_update_frequency']
    threshold = params['threshold']

    return num_episodes, batch_size, num_epochs, time_steps, target_update_frequency, threshold

def train(params, agent, env, best_params = None):
    """
    Conduct training sessions for a given agent and environment.

    Parameters:
    - params (dict): Hyperparameters for training.
    - agent: The agent to be trained.
    - env: The environment in which the agent operates.

    Returns:
    - Multiple values including lists that track various metrics through training.
    """

    if best_params is not None:
        for key in params.keys():
            if key not in best_params.keys():
                best_params[key] = params[key]
    
        params = best_params

    next_state_list_all = []
    rewards_list_all = [] 
    critic_loss_list_all = []
    actor_loss_list_all = []
    reward_list = []
    actor_gradient_list_all = []
    action_dict = {}
    gradient_dict = {}
    transition_probas = {}

    num_sample, device, w1, w2, epsilon_state_exploration = get_param_for_state_exploration(params)
    num_episodes, batch_size, num_epochs, time_steps, target_update_frequency, threshold = get_params_for_train(params)

    agent.train()
    for episode in range(num_episodes):
        print(f"-----------------episode {episode}------------------------")
        env.reset()
        state = env.explore_state(agent, env, num_sample, device, w1, w2, epsilon_state_exploration)
        t = 0

        actor_loss_list= []
        critic_loss_list = []
        actor_gradient_list = []
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

            t += 1
            state = next_state

            if t%target_update_frequency == 0:
                agent.soft_update(network="critic")
                agent.soft_update(network="actor")

        actor_loss_list_all += actor_loss_list
        critic_loss_list_all += critic_loss_list
        actor_gradient_list_all += actor_gradient_list
    
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
    n_states = environment.net.num_edges - environment.num_nullnodes
    n_actions = len(environment.get_state()) - environment.num_entrynodes
    agent = DDPGAgent(n_states, n_actions, hidden, params)
    return agent

def save_all(rewards_list_all, next_state_list_all, \
        critic_loss_list_all, actor_loss_list_all, \
        reward_list, action_dict, gradient_dict, \
        transition_probas, base_path = None):
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

    # Create the directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    pd.DataFrame(reward_list).to_csv(base_path + '/reward.csv')
    pd.DataFrame(actor_loss_list_all).to_csv(base_path + '/actor_loss.csv')
    pd.DataFrame(critic_loss_list_all).to_csv(base_path + '/critic_loss.csv')
    pd.DataFrame(next_state_list_all).to_csv(base_path + '/next_state_model_loss.csv')
    pd.DataFrame(rewards_list_all).to_csv(base_path + '/reward_model_loss.csv')
    pd.DataFrame(action_dict).to_csv(base_path + '/action_dict.csv')
    pd.DataFrame(transition_probas).to_csv(base_path + '/transition_proba.csv')

    import json
    # Specify the filename
    filename = base_path + '/gradient_dict.json'

    # Write the dictionary to a file as JSON
    with open(filename, 'w') as f:
        json.dump(gradient_dict, f)

def start_train(config_file, param_file, save_file = True, data_filename = 'data', image_filename = 'images'):
    """
    Start the training process for a reinforcement learning environment and agent.

    Parameters:
    - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".
    - param_file (str, optional): The file path to the hyperparameters file. Defaults to "hyperparameter_file.yaml".
    - save_file (bool, optional): Flag indicating whether to save the training results to files. Defaults to True.

    This function orchestrates the loading of configurations, creation of environments and agents, and the training process.
    """

    params, hidden = load_hyperparams(param_file)

    sim_environment = create_simulation_env(params, config_file)
    agent = create_ddpg_agent(sim_environment, params, hidden)

    rewards_list_all, next_state_list_all, \
    critic_loss_list_all, actor_loss_list_all, \
    reward_list, action_dict, gradient_dict, \
    transition_probas = train(params, agent, sim_environment)

    csv_filepath = os.getcwd() + '\\' + data_filename
    image_filepath = os.getcwd() + '\\' + image_filename
    if save_file:

        save_all(rewards_list_all, next_state_list_all, \
        critic_loss_list_all, actor_loss_list_all, \
        reward_list, action_dict, gradient_dict, \
        transition_probas, base_path=csv_filepath)
    
    if plot:

        plot(csv_filepath, image_filepath)

def plot_best(data_filepath, images_filepath):
    plot(data_filepath, images_filepath)

def start_tuning(project_name, num_runs, tune_param_filepath, config_param_filepath, eval_param_filepath, 
                 plot_best_param = True, 
                 data_filename = 'data',
                 image_filename = 'images'):

    init_wandb(project_name, tune_param_filepath, config_param_filepath, eval_param_filepath, num_runs = num_runs, opt_target = 'reward')

    if plot_best_param:
        api = wandb.Api(api_key = '02bb2e4979e9df3d890f94a917a95344aae652b9') # replace your api key
        runs = api.runs("yolanda_wang_bu/datasparq")
        best_run = None
        best_metric = None # Assuming higher is better; initialize appropriately based on your metric

        for run in runs:
            # Make sure the metric is reported for the run
            if "reward" in run.summary:
                metric_value = run.summary["reward"]
                
                if best_run is None or metric_value > best_metric:
                    best_metric = metric_value
                    best_run = run

        if best_run:
            print(f"Best run ID: {best_run.id}")
            print(f"Best {best_metric} = {best_metric}")
            print("Best parameters:")
            for key, value in best_run.config.items():
                print(f"{key}: {value}")
        else:
            print("No runs found or metric not reported.")

        best_params = best_run.config

        params, hidden = load_hyperparams(eval_param_filepath)
        sim_environment = create_simulation_env(params, config_param_filepath)
        agent = create_ddpg_agent(sim_environment, params, hidden)
        
        csv_filepath = os.getcwd() + '\\' + data_filename
        image_filepath = os.getcwd() + '\\' + image_filename

        plot_best(csv_filepath, image_filepath)

        rewards_list_all, next_state_list_all, \
            critic_loss_list_all, actor_loss_list_all, \
            reward_list, action_dict, gradient_dict, \
            transition_probas = train(params, agent, sim_environment, best_params = best_params)

        

        save_all(rewards_list_all, next_state_list_all, \
                 critic_loss_list_all, actor_loss_list_all, \
                 reward_list, action_dict, gradient_dict, \
                 transition_probas, base_path=csv_filepath)

        image_filepath = os.getcwd() + '\\' + image_filename
        plot_best(image_filepath)