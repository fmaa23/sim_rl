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
from plot_datasparq import *

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

def create_params(config_file):
    """
    Generate parameters for the queueing environment based on a configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Multiple return values including lists and dictionaries essential for creating the queueing environment.
    """

    def get_service_time(miu_list):
        # compute the time of an agent’s service time from service rate
        services_f = []
        for miu in miu_list:
            def ser_f(t):
                return t + np.exp(miu)
            services_f.append(ser_f)
        return services_f
    
    config_params = load_config(config_file)

    num_queues = config_params['num_queues']
    arrival_rate = config_params['arrival_rate']
    miu_list = config_params['miu_list']
    active_cap = config_params['active_cap']
    deactive_t = config_params['deactive_cap']
    adjacent_list = config_params['adjacent_list']
    buffer_size_for_each_queue = config_params['buffer_size_for_each_queue']
    transition_proba_all = config_params['transition_proba_all']
    services_f = get_service_time(miu_list)

    q_classes, q_args, edge_list = init_env(config_params, buffer_size_for_each_queue, services_f, num_queues)
    return arrival_rate, miu_list, q_classes, q_args, adjacent_list, edge_list, transition_proba_all

def create_q_classes(num_queues):

    q_classes = {}
    q_classes[0] = qt.NullQueue
    for i in range(num_queues):
        q_classes[i+1] = qt.LossQueue
    # q_classes = {0: qt.NullQueue, 1: qt.LossQueue, 2: qt.LossQueue, 3:qt.LossQueue, 4:qt.LossQueue, 5:qt.LossQueue}
    return q_classes

def create_q_args(config_params, buffer_size_for_each_queue, services_f, num_queues):
    # feel free to add other properties
    def ser_f(t):
        return t + np.exp(config_params['sef_rate_first_node'])

    def rate(t):

        return 25 + 350 * np.sin(np.pi * t / 2)**2

    def arr(t):
        return qt.poisson_random_measure(t, rate, config_params['arrival_rate'])
    
    q_args = {}
    print("-----------add qbuffer to first node-----------")
    q_args[1] = {
        'arrival_f': arr,
        'service_f': ser_f,
        'qbuffer': buffer_size_for_each_queue[0]
        }
    

    for i in range(num_queues - 1):
        q_args[i+2] = {
        'service_f': services_f[i],
        'qbuffer':buffer_size_for_each_queue[i],
        }

    return q_args

def init_env(config_params, buffer_size_for_each_queue, services_f, num_queues):

    q_classes = create_q_classes(num_queues)
    q_args = create_q_args(config_params, buffer_size_for_each_queue, services_f, num_queues)

    edge_list = {0:{1:1}, 1: {k: 2 for k in range(2, 5)}, 2:{5:2}, 3:{6:3, 7:4},4:{8:5}, 5:{9:2}, 6:{9:4}, 7:{9:3}, 8:{9:5}, 9:{10:0}}
    return q_classes, q_args, edge_list 