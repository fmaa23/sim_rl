import torch
import numpy as np 
import pandas as pd
import numpy as np
import os
import yaml

from agents.ddpg_agent import DDPGAgent
from queue_env.queueing_network import *
from tuning.wandb_tuning import *
from tuning.ray_tuning import *
from foundations.plot_datasparq import *
from tqdm import tqdm
from queueing_tool.network.queue_network import QueueNetwork
from queueing_tool.queues.queue_servers import *

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
    with open(abs_file_path, 'r') as param_file:
        config_params = yaml.load(param_file, Loader=yaml.FullLoader)
    
    # Convert lists to tuples
    try:
        config_params['entry_nodes'] = [tuple(node) for node in config_params['entry_nodes']]

        for node, value in config_params['miu_list'].items():
            if value == 'inf':
                config_params['miu_list'][node] = float('inf')
    except:
        pass

    return config_params

def load_hyperparams(eval_param_filepath):
    """
    Load hyperparameters from a YAML file.

    Parameters:
    - param_filepath (str): The file path to the hyperparameters YAML file.

    Returns:
    - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    abs_file_path = os.path.join(project_dir, eval_param_filepath)
    
    with open(abs_file_path, 'r') as env_param_file:
        parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)

    params = parameter_dictionary['rl_params']
    hidden = parameter_dictionary['network_params']

    return params, hidden

def get_num_connections(adjacent_list):
    """
    Calculates the total number of connections and identifies exit nodes within the adjacency list of a network.

    Parameters:
    - adjacent_list (dict): A dictionary where keys are start nodes and values are lists of end nodes they connect to.

    Returns:
    - tuple: A tuple containing the total number of connections (int) and a list of exit nodes ([]).
    """
    exit_nodes = []

    for start_node, end_node_list in adjacent_list.items():

        for end_node in end_node_list:
            if end_node not in list(adjacent_list.keys()):
                if end_node not in exit_nodes:
                    exit_nodes.append(end_node)

    return exit_nodes

def make_edge_list(adjacent_list, exit_nodes):
    """
    Creates an edge list with types for each connection based on the adjacency list and identified exit nodes.

    Parameters:
    - adjacent_list (dict): A dictionary representing the network's adjacency list.
    - exit_nodes (list): A list of nodes identified as exit points in the network.

    Returns:
    - dict: A dict
    ionary representing the edge list, where keys are start nodes, and values are dictionaries of end nodes with their edge types.
    """
    edge_list = {}
    edge_type = 1

    for start_node, end_node_list in adjacent_list.items():        
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
    """
    Generates a dictionary mapping each node to a list of nodes that connect to it.

    Parameters:
    - adjacent_list (dict): A dictionary representing the network's adjacency list.

    Returns:
    - dict: A dictionary where keys are end nodes, and values are lists of start nodes that connect to these end nodes.
    """
    connection_info = {}

    for start_node, end_node_list in adjacent_list.items():

        for end_node in end_node_list:
            connect_start_node_list = connection_info.setdefault(end_node, [])
            connect_start_node_list.append(start_node)
            connection_info[end_node] = connect_start_node_list

    return connection_info

def make_unique_edge_type(adjacent_list, edge_list):
    """
    Assigns a unique edge type to connections between nodes based on the adjacency and edge lists.

    Parameters:
    - adjacent_list (dict): A dictionary representing the network's adjacency list.
    - edge_list (dict): A dictionary representing the network's edge list, indicating connections between nodes.

    Returns:
    - dict: A dictionary where keys are node identifiers, and values are lists of unique edge types for edges ending at that node.
    """

    connection_info = get_connection_info(adjacent_list)
    edge_type_info = {}

    for end_node in connection_info.keys():
        start_node_list = connection_info[end_node]
        edge_type_list = []

        for start_node in start_node_list:
            edge_type = edge_list[start_node][end_node]
            edge_type_list.append(edge_type)

        edge_type_info[end_node] = edge_type_list
    return edge_type_info # keys are target node_id, values are the edge_types


def create_params(config_file, disrupt_case = False, disrupt = False, queue_index = 2):
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
    max_agents = config_params['max_agents']
    sim_jobs = config_params['sim_jobs']
    entry_nodes = config_params['entry_nodes']
    exit_nodes = get_num_connections(adjacent_list)
    edge_list = make_edge_list(adjacent_list, exit_nodes)

    for source_node in edge_list.keys():
        for target_node in edge_list[source_node]:
            queue_type = edge_list[source_node][target_node]

            if queue_type == queue_index:
                deactivate_node = target_node + 1
    
    if disrupt_case:
        std = 0.1
        miu_dict = {key: abs(np.random.normal(scale=std)) for key in miu_dict.keys()}

        if disrupt:
            miu_dict[deactivate_node] = float('inf')
    else:
        miu_dict = config_params['miu_list']

    q_classes = create_q_classes(edge_list)
    edge_type_info = make_unique_edge_type(adjacent_list, edge_list)
    buffer_size_for_each_queue = config_params['buffer_size_for_each_queue']
    q_args = create_q_args(edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes, edge_list, q_classes)
    arrival_rate = config_params['arrival_rate']  
    transition_proba_all = config_params['transition_proba_all']
    return arrival_rate, miu_dict, q_classes, q_args, adjacent_list, edge_list, transition_proba_all, max_agents, sim_jobs, entry_nodes

def get_entry_nodes(config_file): 
    config_params = load_config(config_file)
    entry_nodes = [tuple(entry_node) for entry_node in config_params['entry_nodes']]
    return entry_nodes
    
def create_q_classes(edge_list):
    """
    Creates a dictionary mapping queue identifiers to their corresponding queue class.

    Parameters:
    - num_queues (int): The number of queues to create classes for, excluding the null queue.

    Returns:
    - dict: A dictionary where keys are queue identifiers (starting from 1) and values are queue class types.
    """
    q_classes = {}

    for start_node,end_nodes_dict in edge_list.items():

        for end_node in end_nodes_dict.keys():
            edge_index = end_nodes_dict[end_node]
            if end_node in edge_list.keys():
                q_classes[edge_index] = LossQueue
            else:
                q_classes[edge_index] = NullQueue

    return q_classes

def get_node_tuple_from_edgetype(edge_list):
    node_tuple_dict = {}

    for source_node, endnode_type_dict in edge_list.items():

        for end_node, edge_type in endnode_type_dict.items():
            if edge_type in node_tuple_dict:
                node_tuple_dict[edge_type].append((source_node, end_node))
            else:
                node_tuple_dict[edge_type] = [(source_node, end_node)]

    return node_tuple_dict


def get_node_id(edge_type, edge_type_info):
    for node in edge_type_info.keys():
        if edge_type in edge_type_info[node]:
            return node

def create_q_args(edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes, edge_list, q_classes):
    """
    Constructs arguments for queue initialization based on the network configuration.

    Parameters:
    - edge_type_info (dict): Information about edge types for each node.
    - config_params (dict): Configuration parameters including service rates and buffer sizes.
    - miu_dict (dict): A dictionary mapping nodes to their service rates.
    - buffer_size_for_each_queue (dict): A dictionary mapping queue identifiers to their buffer sizes.
    - exit_nodes (list): A list of nodes identified as exit points in the network.

    Returns:
    - dict: A dictionary of queue arguments where keys are queue identifiers, and values are dictionaries of arguments needed for initializing each queue.
    """

    q_args = {}
    edge_type_lists = []

    for key in edge_type_info.keys():
        if key not in exit_nodes:
            values = edge_type_info[key]
            edge_type_lists += values

    node_tuple_by_edgetype=get_node_tuple_from_edgetype(edge_list)
    entry_node_encountered = 0
    env_entry_nodes = [tuple(item) for item in config_params['entry_nodes']]

    for edge_type in edge_type_lists:
        queue_type = q_classes[edge_type]
        node_id = get_node_id(edge_type, edge_type_info) 
        service_rate = miu_dict[node_id]
        if queue_type == LossQueue:
            if node_tuple_by_edgetype[edge_type][0] in config_params['entry_nodes']:
                max_arrival_rate = config_params['arrival_rate'][entry_node_encountered]
                rate = lambda t: 0.1*(max_arrival_rate) + (1-0.1)*(max_arrival_rate) * np.sin(np.pi * t / 2)**2
                q_args[edge_type] = {
                'arrival_f': lambda t, rate=rate: poisson_random_measure(t, rate, max_arrival_rate),
                'service_f': lambda t, en=node_id:t+np.exp(miu_dict[en]),
                'qbuffer': buffer_size_for_each_queue[edge_type],
                'service_rate': service_rate,
                'active_cap': float('inf'), 
                'active_status' : True
                }
                entry_node_encountered+=1
            else:
                q_args[edge_type] = {
                'service_f': lambda t, en=node_id:t+np.exp(miu_dict[en]),
                'qbuffer':buffer_size_for_each_queue[edge_type],
                'service_rate': service_rate,
                'active_cap':float('inf'),
                'active_status' : False
                }
    return q_args

def create_queueing_env(config_file, disrupt_case = False, disrupt = False, queue_index = 2):
    """
    Create and configure a queueing environment based on a given configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Queue_network: An instance of the queueing environment.
    """
    arrival_rate, miu_dict, q_classes, q_args, adjacent_list, edge_list, \
    transition_proba_all, max_agents, sim_jobs, entry_nodes = create_params(config_file, 
                                                                                             disrupt_case = disrupt_case, 
                                                                                             disrupt = disrupt, 
                                                                                             queue_index = queue_index)
    

    q_net = Queue_network()
    q_net.process_input(arrival_rate, miu_dict, q_classes, q_args, adjacent_list, 
                        edge_list, transition_proba_all, max_agents, sim_jobs)
    q_net.create_env()
    return q_net

def create_RL_env(q_net, params, entry_nodes):
    """
    Create a reinforcement learning environment.

    Parameters:
    - q_net (Queue_network): The queueing network environment.
    - params (dict): Parameters for the RL environment.
    - hidden (dict): Hidden layer configurations.

    Returns:
    - RLEnv: An instance of the RL environment.
    """
    env = RLEnv(q_net, num_sim = params['num_sim'], entry_nodes=entry_nodes)
    return env

def create_simulation_env(params, config_file, disrupt_case = False, disrupt = False, queue_index = 2):
    """
    Create a simulation environment for reinforcement learning based on given parameters and a configuration file.

    Parameters:
    - params (dict): Hyperparameters for the simulation.
    - hidden (dict): Hidden layer configurations.
    - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".

    Returns:
    - RLEnv: The RL environment ready for simulation.
    """
    q_net = create_queueing_env(config_file, disrupt_case, disrupt = disrupt, queue_index = queue_index)
    entry_nodes = get_entry_nodes(config_file)
    RL_env = create_RL_env(q_net, params, entry_nodes)

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
    reset = params['reset']
    if reset == False:
        reset_frequency = None
    else:
        reset_frequency = params['reset_frequency']

    return num_sample, device_here, w1, w2, epsilon, reset, reset_frequency

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
    num_train_AC = params['num_train_AC']
    return num_episodes, batch_size, num_epochs, time_steps, target_update_frequency, num_train_AC

def init_transition_proba(env):
    transition_proba = {}
    adjacent_lists = env.qn_net.adja_list 

    for start_node in adjacent_lists.keys():
        if len(adjacent_lists[start_node]) > 1:
            transition_proba[start_node] = {}

    return transition_proba

def update_transition_probas(transition_probas, env):
    for start_node in transition_probas.keys():
        next_nodes = env.transition_proba[start_node].keys()
        next_proba_dict = transition_probas[start_node]

        for next_node in next_nodes:
            proba_list = next_proba_dict.setdefault(next_node, []) 
            if len(proba_list) == 0:
                proba_list.append(env.qn_net.transition_proba[start_node][next_node])
            proba_list.append(env.transition_proba[start_node][next_node])
            next_proba_dict[next_node] = proba_list

        transition_probas[start_node] = next_proba_dict
    return transition_probas

def convert_format(state):
    initial_states = {}

    for index, num in enumerate(state):
        initial_states[index] = num
    return initial_states

def save_agent(agent): 
    """
    Saves the trained RL agent to a file.

    This function creates a directory named 'Agent' in the current working directory if it doesn't exist,
    and saves the given agent model to a file named 'tensor.pt' within this directory.
    """
    base_path = os.getcwd() + "MScDataSparqProject\\"
    agent_dir = os.path.join(base_path, 'agents')

    # Create the directory if it does not exist
    if not os.path.exists(agent_dir):
        os.makedirs(agent_dir)
        print(f"Directory created at {agent_dir}")
    file_path = os.path.join(agent_dir, 'trained_agent.pt')
    torch.save(agent, file_path)
    print(f"Agent saved successfully at {file_path}")

def train(params, agent, env, best_params = None, blockage_qn_net = None):
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

    next_state_model_list_all = []
    reward_model_list_all =[]
    gradient_dict_all = {}
    action_dict = {}
    gradient_dict_all = {}
    transition_probas = init_transition_proba(env)
    actor_loss_list= []
    critic_loss_list = []
    reward_list = []
    reward_by_episode = {}
    num_episodes, _, num_epochs, time_steps, _, num_train_AC = get_params_for_train(params)
    latest_transition_proba = None

    for episode in tqdm(range(num_episodes), desc="Episode Progress"): 
        agent.train()
        if blockage_qn_net is None:
            env.reset()
        else:
            env.reset(blockage_qn_net)

        if latest_transition_proba is not None:
            env.net.set_transitions(latest_transition_proba)

        env.simulate()
        update = 0
        reward_list = []

        for _ in tqdm(range(time_steps), desc="Time Steps Progress"): 

            state = env.get_state()
            
            state_tensor = torch.tensor(state)
            action = agent.select_action(state_tensor).to(device) 
            action_list = action.cpu().numpy().tolist()

            for index, value in enumerate(action_list):
                node_list = action_dict.setdefault(index, [])
                node_list.append(value)
                action_dict[index] = node_list

            next_state_tensor = torch.tensor(env.get_next_state(action)).float().to(device)
            reward = env.get_reward()
            reward_list.append(reward)                               
            experience = (state_tensor, action, reward, next_state_tensor) 
            agent.store_experience(experience)                             

        reward_model_loss_list, next_state_loss_list = agent.fit_model(batch_size=time_steps, epochs=num_epochs)
        next_state_model_list_all += next_state_loss_list
        reward_model_list_all += reward_model_loss_list
        transition_probas = update_transition_probas(transition_probas, env)

        for _ in tqdm(range(num_train_AC), desc="Train Agent"): 

            batch = agent.buffer.sample(batch_size=time_steps)
            critic_loss = agent.update_critic_network(batch)                   
            actor_loss, gradient_dict = agent.update_actor_network(batch)    
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)

        agent.plan(batch)
        agent.soft_update(network="critic")
        agent.soft_update(network="actor")
        gradient_dict_all[update] = gradient_dict
        agent.buffer.clear()
        reward_by_episode[episode] = reward_list 
        latest_transition_proba = env.transition_proba
    
    save_agent(agent)
    return next_state_model_list_all, critic_loss_list, actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas

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
    n_states = (environment.net.num_edges - environment.num_nullnodes)
    n_actions = environment.net.num_nodes
    agent = DDPGAgent(n_states, n_actions, hidden, params, device)
    return agent

def get_transition_proba_df(transition_probas):
    flatten_dict = {}

    for start_node in transition_probas.keys():
        end_nodes = list(transition_probas[start_node].keys())

        for end_node in end_nodes:
            flatten_dict[end_node] = transition_probas[start_node][end_node]

    df_transition_proba = pd.DataFrame(flatten_dict)
    return df_transition_proba


def save_all(next_state_model_list_all, critic_loss_list,\
          actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas, base_path = None):
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
    import json
    # Create the directory if it doesn't exist
    if base_path is None:
        base_path = os.getcwd()
        
    output_dir = os.path.join(base_path, "foundations", "output_csv")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(actor_loss_list).to_csv(os.path.join(output_dir, 'actor_loss.csv'), index=False)
    pd.DataFrame(critic_loss_list).to_csv(os.path.join(output_dir, 'critic_loss.csv'), index=False)
    pd.DataFrame(next_state_model_list_all).to_csv(os.path.join(output_dir, 'next_state_model_loss.csv'), index=False)
    pd.DataFrame(action_dict).to_csv(os.path.join(output_dir, 'action_dict.csv'), index=False)

    df_transition = get_transition_proba_df(transition_probas)
    df_transition.to_csv(os.path.join(output_dir, 'transition_proba.csv'), index=False)

    with open(os.path.join(output_dir, 'gradient_dict.json'), 'w') as f:
        json.dump(gradient_dict, f)
    with open(os.path.join(output_dir, 'reward_dict.json'), 'w') as f:
        json.dump(reward_by_episode, f)

    print(f"CSVs have been saved at {output_dir}")

def start_train(config_file, param_file, save_file = True, 
                data_filename = 'data', image_filename = 'images', plot_curves = True):
    """
    Starts the training process for a reinforcement learning environment and agent.

    Parameters:
    - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".
    - param_file (str, optional): The file path to the hyperparameters file. Defaults to "hyperparameter_file.yaml".
    - save_file (bool, optional): Flag indicating whether to save the training results to files. Defaults to True.

    This function orchestrates the loading of configurations, creation of environments and agents, and the training process.
    """

    params, hidden = load_hyperparams(param_file)
    sim_environment = create_simulation_env(params, config_file)
    agent = create_ddpg_agent(sim_environment, params, hidden)
    
    next_state_model_list_all, critic_loss_list,\
    actor_loss_list, reward_by_episode, action_dict, \
    gradient_dict, transition_probas = train(params, agent, sim_environment)
    
    current_dir = os.getcwd()
    foundations_dir = 'foundations'
    csv_filepath = os.path.join(current_dir, foundations_dir, data_filename)
    image_filepath = os.path.join(current_dir, foundations_dir, image_filename)

    if save_file:
        save_all(next_state_model_list_all, critic_loss_list,\
          actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas)
    
    if plot_curves:
        plot(csv_filepath, image_filepath, transition_probas)


def plot_best(data_filepath, images_filepath):
    plot(data_filepath, images_filepath)


def start_tuning(project_name, num_runs, tune_param_filepath, config_param_filepath, eval_param_filepath, api_key, 
                 plot_best_param = True, 
                 data_filename = 'data',
                 image_filename = 'images',
                 tuner = 'wandb'):
    """
    Initiates the hyperparameter tuning process for a reinforcement learning project, optionally plots the best parameters,
    and starts a training session with those parameters.

    Parameters:
    - project_name (str): The name of the project in Wandb where the tuning results will be tracked.
    - num_runs (int): The number of tuning runs to perform.
    - tune_param_filepath (str): The file path to the YAML file containing the parameters to be tuned.
    - config_param_filepath (str): The file path to the YAML file containing the configuration parameters for the queueing environment.
    - eval_param_filepath (str): The file path to the YAML file containing the evaluation parameters for the model.
    - plot_best_param (bool, optional): A flag to indicate whether to plot the best parameters after tuning. Defaults to True.
    - data_filename (str, optional): The base name for data files where training results will be saved. Defaults to 'data'.
    - image_filename (str, optional): The base name for image files where plots will be saved. Defaults to 'images'.

    This function utilizes Wandb for hyperparameter tuning, tracking, and selecting the best parameters based on a specified metric.
    It then loads the best parameters, if found, and proceeds to create a simulation environment and an agent. Training is then
    conducted using these parameters, and results are optionally saved and plotted.

    Note: The function assumes access to Wandb and requires an API key for Wandb to be set up in advance.
    """
    if tuner == 'wandb':
        init_wandb(project_name, tune_param_filepath, config_param_filepath, eval_param_filepath,num_runs = num_runs, opt_target = 'reward')

        if plot_best_param:
            api = wandb.Api(api_key = api_key) # replace your api key
            runs = api.runs("datasparq")
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
            current_dir = os.getcwd()
            csv_filepath = os.path.join(current_dir, data_filename)
            image_filepath = os.path.join(current_dir, image_filename)

            plot_best(csv_filepath, image_filepath)

            rewards_list_all, next_state_list_all, \
                critic_loss_list_all, actor_loss_list_all, \
                reward_list, action_dict, gradient_dict, \
                transition_probas = train(params, agent, sim_environment, best_params = best_params)

            save_all(rewards_list_all, next_state_list_all, \
                    critic_loss_list_all, actor_loss_list_all, \
                    reward_list, action_dict, gradient_dict, \
                    transition_probas, base_path=csv_filepath)

            image_filepath = os.path.join(current_dir, image_filename)

            plot_best(image_filepath)
    else:
        ray_tune()
        
        
def start_evaluation(environment, agent, time_steps):
        """ This function is used to allow a trained agent to actively make decisions in the environment and returns the total reward obtained after a specified number of time steps.
        """
        #environment.simulate() 
        total_reward = 0 
        state = environment.reset()
        for _ in tqdm(range(time_steps),desc="Control"): 
            state = environment.get_state()
            action = agent.actor(state).detach()
            state = environment.get_next_state(action)[0]
            reward = environment.get_reward()
            total_reward += reward
        return total_reward
    
        
    
    
# Converting this folder into a class
class CoreFunctions():
    def __init__(self):
        pass
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
        with open(abs_file_path, 'r') as param_file:
            config_params = yaml.load(param_file, Loader=yaml.FullLoader)
        
        # Convert lists to tuples
        try:
            config_params['entry_nodes'] = [tuple(node) for node in config_params['entry_nodes']]

            for node, value in config_params['miu_list'].items():
                if value == 'inf':
                    config_params['miu_list'][node] = float('inf')
        except:
            pass

        return config_params

    def load_hyperparams(eval_param_filepath):
        """
        Load hyperparameters from a YAML file.

        Parameters:
        - param_filepath (str): The file path to the hyperparameters YAML file.

        Returns:
        - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
        """

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        abs_file_path = os.path.join(project_dir, eval_param_filepath)
        
        with open(abs_file_path, 'r') as env_param_file:
            parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)

        params = parameter_dictionary['rl_params']
        hidden = parameter_dictionary['network_params']

        return params, hidden

    def get_num_connections(adjacent_list):
        """
        Calculates the total number of connections and identifies exit nodes within the adjacency list of a network.

        Parameters:
        - adjacent_list (dict): A dictionary where keys are start nodes and values are lists of end nodes they connect to.

        Returns:
        - tuple: A tuple containing the total number of connections (int) and a list of exit nodes ([]).
        """
        exit_nodes = []

        for start_node, end_node_list in adjacent_list.items():

            for end_node in end_node_list:
                if end_node not in list(adjacent_list.keys()):
                    if end_node not in exit_nodes:
                        exit_nodes.append(end_node)

        return exit_nodes

    def make_edge_list(adjacent_list, exit_nodes):
        """
        Creates an edge list with types for each connection based on the adjacency list and identified exit nodes.

        Parameters:
        - adjacent_list (dict): A dictionary representing the network's adjacency list.
        - exit_nodes (list): A list of nodes identified as exit points in the network.

        Returns:
        - dict: A dict
        ionary representing the edge list, where keys are start nodes, and values are dictionaries of end nodes with their edge types.
        """
        edge_list = {}
        edge_type = 1

        for start_node, end_node_list in adjacent_list.items():        
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
        """
        Generates a dictionary mapping each node to a list of nodes that connect to it.

        Parameters:
        - adjacent_list (dict): A dictionary representing the network's adjacency list.

        Returns:
        - dict: A dictionary where keys are end nodes, and values are lists of start nodes that connect to these end nodes.
        """
        connection_info = {}

        for start_node, end_node_list in adjacent_list.items():

            for end_node in end_node_list:
                connect_start_node_list = connection_info.setdefault(end_node, [])
                connect_start_node_list.append(start_node)
                connection_info[end_node] = connect_start_node_list

        return connection_info

    def make_unique_edge_type(adjacent_list, edge_list):
        """
        Assigns a unique edge type to connections between nodes based on the adjacency and edge lists.

        Parameters:
        - adjacent_list (dict): A dictionary representing the network's adjacency list.
        - edge_list (dict): A dictionary representing the network's edge list, indicating connections between nodes.

        Returns:
        - dict: A dictionary where keys are node identifiers, and values are lists of unique edge types for edges ending at that node.
        """

        connection_info = get_connection_info(adjacent_list)
        edge_type_info = {}

        for end_node in connection_info.keys():
            start_node_list = connection_info[end_node]
            edge_type_list = []

            for start_node in start_node_list:
                edge_type = edge_list[start_node][end_node]
                edge_type_list.append(edge_type)

            edge_type_info[end_node] = edge_type_list
        return edge_type_info # keys are target node_id, values are the edge_types


    def create_params(config_file, disrupt_case = False, disrupt = False, queue_index = 2):
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
        max_agents = config_params['max_agents']
        sim_jobs = config_params['sim_jobs']
        entry_nodes = config_params['entry_nodes']
        exit_nodes = get_num_connections(adjacent_list)
        edge_list = make_edge_list(adjacent_list, exit_nodes)

        for source_node in edge_list.keys():
            for target_node in edge_list[source_node]:
                queue_type = edge_list[source_node][target_node]

                if queue_type == queue_index:
                    deactivate_node = target_node + 1
        
        if disrupt_case:
            std = 0.1
            miu_dict = {key: abs(np.random.normal(scale=std)) for key in miu_dict.keys()}

            if disrupt:
                miu_dict[deactivate_node] = float('inf')
        else:
            miu_dict = config_params['miu_list']

        q_classes = create_q_classes(edge_list)
        edge_type_info = make_unique_edge_type(adjacent_list, edge_list)
        buffer_size_for_each_queue = config_params['buffer_size_for_each_queue']
        q_args = create_q_args(edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes, edge_list, q_classes)
        arrival_rate = config_params['arrival_rate']  
        transition_proba_all = config_params['transition_proba_all']
        return arrival_rate, miu_dict, q_classes, q_args, adjacent_list, edge_list, transition_proba_all, max_agents, sim_jobs, entry_nodes

    def get_entry_nodes(config_file): 
        config_params = load_config(config_file)
        entry_nodes = [tuple(entry_node) for entry_node in config_params['entry_nodes']]
        return entry_nodes
        
    def create_q_classes(edge_list):
        """
        Creates a dictionary mapping queue identifiers to their corresponding queue class.

        Parameters:
        - num_queues (int): The number of queues to create classes for, excluding the null queue.

        Returns:
        - dict: A dictionary where keys are queue identifiers (starting from 1) and values are queue class types.
        """
        q_classes = {}

        for start_node,end_nodes_dict in edge_list.items():

            for end_node in end_nodes_dict.keys():
                edge_index = end_nodes_dict[end_node]
                if end_node in edge_list.keys():
                    q_classes[edge_index] = LossQueue
                else:
                    q_classes[edge_index] = NullQueue

        return q_classes

    def get_node_tuple_from_edgetype(edge_list):
        node_tuple_dict = {}

        for source_node, endnode_type_dict in edge_list.items():

            for end_node, edge_type in endnode_type_dict.items():
                if edge_type in node_tuple_dict:
                    node_tuple_dict[edge_type].append((source_node, end_node))
                else:
                    node_tuple_dict[edge_type] = [(source_node, end_node)]

        return node_tuple_dict


    def get_node_id(edge_type, edge_type_info):
        for node in edge_type_info.keys():
            if edge_type in edge_type_info[node]:
                return node

    def create_q_args(edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes, edge_list, q_classes):
        """
        Constructs arguments for queue initialization based on the network configuration.

        Parameters:
        - edge_type_info (dict): Information about edge types for each node.
        - config_params (dict): Configuration parameters including service rates and buffer sizes.
        - miu_dict (dict): A dictionary mapping nodes to their service rates.
        - buffer_size_for_each_queue (dict): A dictionary mapping queue identifiers to their buffer sizes.
        - exit_nodes (list): A list of nodes identified as exit points in the network.

        Returns:
        - dict: A dictionary of queue arguments where keys are queue identifiers, and values are dictionaries of arguments needed for initializing each queue.
        """

        q_args = {}
        edge_type_lists = []

        for key in edge_type_info.keys():
            if key not in exit_nodes:
                values = edge_type_info[key]
                edge_type_lists += values

        node_tuple_by_edgetype=get_node_tuple_from_edgetype(edge_list)
        entry_node_encountered = 0
        env_entry_nodes = [tuple(item) for item in config_params['entry_nodes']]

        for edge_type in edge_type_lists:
            queue_type = q_classes[edge_type]
            node_id = get_node_id(edge_type, edge_type_info) 
            service_rate = miu_dict[node_id]
            if queue_type == LossQueue:
                if node_tuple_by_edgetype[edge_type][0] in config_params['entry_nodes']:
                    max_arrival_rate = config_params['arrival_rate'][entry_node_encountered]
                    rate = lambda t: 0.1*(max_arrival_rate) + (1-0.1)*(max_arrival_rate) * np.sin(np.pi * t / 2)**2
                    q_args[edge_type] = {
                    'arrival_f': lambda t, rate=rate: poisson_random_measure(t, rate, max_arrival_rate),
                    'service_f': lambda t, en=node_id:t+np.exp(miu_dict[en]),
                    'qbuffer': buffer_size_for_each_queue[edge_type],
                    'service_rate': service_rate,
                    'active_cap': float('inf'), 
                    'active_status' : True
                    }
                    entry_node_encountered+=1
                else:
                    q_args[edge_type] = {
                    'service_f': lambda t, en=node_id:t+np.exp(miu_dict[en]),
                    'qbuffer':buffer_size_for_each_queue[edge_type],
                    'service_rate': service_rate,
                    'active_cap':float('inf'),
                    'active_status' : False
                    }
        return q_args

    def create_queueing_env(config_file, disrupt_case = False, disrupt = False, queue_index = 2):
        """
        Create and configure a queueing environment based on a given configuration file.

        Parameters:
        - config_file (str): The file path to the environment configuration file.

        Returns:
        - Queue_network: An instance of the queueing environment.
        """
        arrival_rate, miu_dict, q_classes, q_args, adjacent_list, edge_list, \
        transition_proba_all, max_agents, sim_jobs, entry_nodes = create_params(config_file, 
                                                                                                disrupt_case = disrupt_case, 
                                                                                                disrupt = disrupt, 
                                                                                                queue_index = queue_index)
        

        q_net = Queue_network()
        q_net.process_input(arrival_rate, miu_dict, q_classes, q_args, adjacent_list, 
                            edge_list, transition_proba_all, max_agents, sim_jobs)
        q_net.create_env()
        return q_net

    def create_RL_env(q_net, params, entry_nodes):
        """
        Create a reinforcement learning environment.

        Parameters:
        - q_net (Queue_network): The queueing network environment.
        - params (dict): Parameters for the RL environment.
        - hidden (dict): Hidden layer configurations.

        Returns:
        - RLEnv: An instance of the RL environment.
        """
        env = RLEnv(q_net, num_sim = params['num_sim'], entry_nodes=entry_nodes)
        return env

    def create_simulation_env(params, config_file, disrupt_case = False, disrupt = False, queue_index = 2):
        """
        Create a simulation environment for reinforcement learning based on given parameters and a configuration file.

        Parameters:
        - params (dict): Hyperparameters for the simulation.
        - hidden (dict): Hidden layer configurations.
        - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".

        Returns:
        - RLEnv: The RL environment ready for simulation.
        """
        q_net = create_queueing_env(config_file, disrupt_case, disrupt = disrupt, queue_index = queue_index)
        entry_nodes = get_entry_nodes(config_file)
        RL_env = create_RL_env(q_net, params, entry_nodes)

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
        reset = params['reset']
        if reset == False:
            reset_frequency = None
        else:
            reset_frequency = params['reset_frequency']

        return num_sample, device_here, w1, w2, epsilon, reset, reset_frequency

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
        num_train_AC = params['num_train_AC']
        return num_episodes, batch_size, num_epochs, time_steps, target_update_frequency, num_train_AC

    def init_transition_proba(env):
        transition_proba = {}
        adjacent_lists = env.qn_net.adja_list 

        for start_node in adjacent_lists.keys():
            if len(adjacent_lists[start_node]) > 1:
                transition_proba[start_node] = {}

        return transition_proba

    def update_transition_probas(transition_probas, env):
        for start_node in transition_probas.keys():
            next_nodes = env.transition_proba[start_node].keys()
            next_proba_dict = transition_probas[start_node]

            for next_node in next_nodes:
                proba_list = next_proba_dict.setdefault(next_node, []) 
                if len(proba_list) == 0:
                    proba_list.append(env.qn_net.transition_proba[start_node][next_node])
                proba_list.append(env.transition_proba[start_node][next_node])
                next_proba_dict[next_node] = proba_list

            transition_probas[start_node] = next_proba_dict
        return transition_probas

    def convert_format(state):
        initial_states = {}

        for index, num in enumerate(state):
            initial_states[index] = num
        return initial_states

    def save_agent(agent): 
        """
        Saves the trained RL agent to a file.

        This function creates a directory named 'Agent' in the current working directory if it doesn't exist,
        and saves the given agent model to a file named 'tensor.pt' within this directory.
        """
        base_path = os.getcwd() + "MScDataSparqProject\\"
        agent_dir = os.path.join(base_path, 'agents')

        # Create the directory if it does not exist
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)
            print(f"Directory created at {agent_dir}")
        file_path = os.path.join(agent_dir, 'trained_agent.pt')
        torch.save(agent, file_path)
        print(f"Agent saved successfully at {file_path}")

    def train(params, agent, env, best_params = None, blockage_qn_net = None):
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

        next_state_model_list_all = []
        reward_model_list_all =[]
        gradient_dict_all = {}
        action_dict = {}
        gradient_dict_all = {}
        transition_probas = init_transition_proba(env)
        actor_loss_list= []
        critic_loss_list = []
        reward_list = []
        reward_by_episode = {}
        num_episodes, _, num_epochs, time_steps, _, num_train_AC = get_params_for_train(params)
        latest_transition_proba = None

        for episode in tqdm(range(num_episodes), desc="Episode Progress"): 
            agent.train()
            if blockage_qn_net is None:
                env.reset()
            else:
                env.reset(blockage_qn_net)

            if latest_transition_proba is not None:
                env.net.set_transitions(latest_transition_proba)

            env.simulate()
            update = 0
            reward_list = []

            for _ in tqdm(range(time_steps), desc="Time Steps Progress"): 

                state = env.get_state()
                
                state_tensor = torch.tensor(state)
                action = agent.select_action(state_tensor).to(device) 
                action_list = action.cpu().numpy().tolist()

                for index, value in enumerate(action_list):
                    node_list = action_dict.setdefault(index, [])
                    node_list.append(value)
                    action_dict[index] = node_list

                next_state_tensor = torch.tensor(env.get_next_state(action)).float().to(device)
                reward = env.get_reward()
                reward_list.append(reward)                               
                experience = (state_tensor, action, reward, next_state_tensor) 
                agent.store_experience(experience)                             

            reward_model_loss_list, next_state_loss_list = agent.fit_model(batch_size=time_steps, epochs=num_epochs)
            next_state_model_list_all += next_state_loss_list
            reward_model_list_all += reward_model_loss_list
            transition_probas = update_transition_probas(transition_probas, env)

            for _ in tqdm(range(num_train_AC), desc="Train Agent"): 

                batch = agent.buffer.sample(batch_size=time_steps)
                critic_loss = agent.update_critic_network(batch)                   
                actor_loss, gradient_dict = agent.update_actor_network(batch)    
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

            agent.plan(batch)
            agent.soft_update(network="critic")
            agent.soft_update(network="actor")
            gradient_dict_all[update] = gradient_dict
            agent.buffer.clear()
            reward_by_episode[episode] = reward_list 
            latest_transition_proba = env.transition_proba
        
        save_agent(agent)
        return next_state_model_list_all, critic_loss_list, actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas

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
        n_states = (environment.net.num_edges - environment.num_nullnodes)
        n_actions = len(environment.get_state()) - environment.num_nullnodes
        agent = DDPGAgent(n_states, n_actions, hidden, params, device)
        return agent

    def get_transition_proba_df(transition_probas):
        flatten_dict = {}

        for start_node in transition_probas.keys():
            end_nodes = list(transition_probas[start_node].keys())

            for end_node in end_nodes:
                flatten_dict[end_node] = transition_probas[start_node][end_node]

        df_transition_proba = pd.DataFrame(flatten_dict)
        return df_transition_proba


    def save_all(next_state_model_list_all, critic_loss_list,\
            actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas, base_path = None):
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
        import json
        # Create the directory if it doesn't exist
        if base_path is None:
            base_path = os.getcwd()
            
        output_dir = os.path.join(base_path, "foundations", "output_csv")
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(actor_loss_list).to_csv(os.path.join(output_dir, 'actor_loss.csv'), index=False)
        pd.DataFrame(critic_loss_list).to_csv(os.path.join(output_dir, 'critic_loss.csv'), index=False)
        pd.DataFrame(next_state_model_list_all).to_csv(os.path.join(output_dir, 'next_state_model_loss.csv'), index=False)
        pd.DataFrame(action_dict).to_csv(os.path.join(output_dir, 'action_dict.csv'), index=False)

        df_transition = get_transition_proba_df(transition_probas)
        df_transition.to_csv(os.path.join(output_dir, 'transition_proba.csv'), index=False)

        with open(os.path.join(output_dir, 'gradient_dict.json'), 'w') as f:
            json.dump(gradient_dict, f)
        with open(os.path.join(output_dir, 'reward_dict.json'), 'w') as f:
            json.dump(reward_by_episode, f)

        print(f"CSVs have been saved at {output_dir}")

    def start_train(config_file, param_file, save_file = True, 
                    data_filename = 'data', image_filename = 'images', plot_curves = True):
        """
        Starts the training process for a reinforcement learning environment and agent.

        Parameters:
        - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".
        - param_file (str, optional): The file path to the hyperparameters file. Defaults to "hyperparameter_file.yaml".
        - save_file (bool, optional): Flag indicating whether to save the training results to files. Defaults to True.

        This function orchestrates the loading of configurations, creation of environments and agents, and the training process.
        """

        params, hidden = load_hyperparams(param_file)
        sim_environment = create_simulation_env(params, config_file)
        agent = create_ddpg_agent(sim_environment, params, hidden)
        
        next_state_model_list_all, critic_loss_list,\
        actor_loss_list, reward_by_episode, action_dict, \
        gradient_dict, transition_probas = train(params, agent, sim_environment)
        
        current_dir = os.getcwd()
        foundations_dir = 'foundations'
        csv_filepath = os.path.join(current_dir, foundations_dir, data_filename)
        image_filepath = os.path.join(current_dir, foundations_dir, image_filename)

        if save_file:
            save_all(next_state_model_list_all, critic_loss_list,\
            actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas)
        
        if plot_curves:
            plot(csv_filepath, image_filepath, transition_probas)


    def plot_best(data_filepath, images_filepath):
        plot(data_filepath, images_filepath)


    def start_tuning(project_name, num_runs, tune_param_filepath, config_param_filepath, eval_param_filepath, api_key, 
                    plot_best_param = True, 
                    data_filename = 'data',
                    image_filename = 'images',
                    tuner = 'wandb'):
        """
        Initiates the hyperparameter tuning process for a reinforcement learning project, optionally plots the best parameters,
        and starts a training session with those parameters.

        Parameters:
        - project_name (str): The name of the project in Wandb where the tuning results will be tracked.
        - num_runs (int): The number of tuning runs to perform.
        - tune_param_filepath (str): The file path to the YAML file containing the parameters to be tuned.
        - config_param_filepath (str): The file path to the YAML file containing the configuration parameters for the queueing environment.
        - eval_param_filepath (str): The file path to the YAML file containing the evaluation parameters for the model.
        - plot_best_param (bool, optional): A flag to indicate whether to plot the best parameters after tuning. Defaults to True.
        - data_filename (str, optional): The base name for data files where training results will be saved. Defaults to 'data'.
        - image_filename (str, optional): The base name for image files where plots will be saved. Defaults to 'images'.

        This function utilizes Wandb for hyperparameter tuning, tracking, and selecting the best parameters based on a specified metric.
        It then loads the best parameters, if found, and proceeds to create a simulation environment and an agent. Training is then
        conducted using these parameters, and results are optionally saved and plotted.

        Note: The function assumes access to Wandb and requires an API key for Wandb to be set up in advance.
        """
        if tuner == 'wandb':
            init_wandb(project_name, tune_param_filepath, config_param_filepath, eval_param_filepath,num_runs = num_runs, opt_target = 'reward')

            if plot_best_param:
                api = wandb.Api(api_key = api_key) # replace your api key
                runs = api.runs("datasparq")
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
                current_dir = os.getcwd()
                csv_filepath = os.path.join(current_dir, data_filename)
                image_filepath = os.path.join(current_dir, image_filename)

                plot_best(csv_filepath, image_filepath)

                rewards_list_all, next_state_list_all, \
                    critic_loss_list_all, actor_loss_list_all, \
                    reward_list, action_dict, gradient_dict, \
                    transition_probas = train(params, agent, sim_environment, best_params = best_params)

                save_all(rewards_list_all, next_state_list_all, \
                        critic_loss_list_all, actor_loss_list_all, \
                        reward_list, action_dict, gradient_dict, \
                        transition_probas, base_path=csv_filepath)

                image_filepath = os.path.join(current_dir, image_filename)

                plot_best(image_filepath)
        else:
            ray_tune()
            
            
    def start_evaluation(environment, agent, time_steps):
        """ This function is used to allow a trained agent to actively make decisions in the environment and returns the total reward obtained after a specified number of time steps.
        """
        
        environment.simulate()
        total_reward = 0 
        state = environment.reset()
        for _ in tqdm(range(time_steps),desc="Control"): 
            state = environment.get_state()
            action = agent.actor(state).detach()
            state = environment.get_next_state(action)[0]
            reward = environment.get_reward()
            total_reward += reward
        return total_reward