# This class will be used to evaluate the effect of environmental noise on the performance of the agent
import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))
import torch 
import matplotlib.pyplot as plt
from rl_env.RL_Environment import *
from queue_env.queueing_network import * 
from foundations.supporting_functions import *
from foundations.supporting_functions import CoreFunctions
import numpy as np
import copy
import os 
from queue_env.queueing_network import Queue_network
from foundations.wandb_base_functions import * 

# Definition of the NoiseEvaluator class
class NoiseEvaluator(CoreFunctions):
    def __init__(self,frequency,mean,variance):
        """
        Args:
            frequency(float ): the frequency at which noise is added to the environment - enforce that its between 0 and 1
            mean (float): Mean of the distribution from which the noise is sampled
            variance (float): Variance of the distribution from which the noise is sampled
        """
        self.frequency = frequency
        self.mean = mean 
        self.variance = variance  

    def compute_increment(self): 
        """This function is main entry point for adding noise to the environment. This function samples from a normal distribution with mean and variance specified in the constructor and
        returns the noise increment to be added to the environment with a probability specified by the frequency parameter.
        Args:
        
        """
        if self.frequency > np.random.random():
            # Determines whether we are currently at a noise injection interval 
            noise = np.random.normal(self.mean,self.variance)
            return noise
        else:
            return 0
        
    def create_q_args(self,edge_type_info, config_params, miu_dict, buffer_size_for_each_queue, exit_nodes, edge_list, q_classes):
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
                    'arrival_f': lambda t, rate=rate: poisson_random_measure(t, rate , max_arrival_rate) + self.noise_evaluator.compute_increment(),
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

# Monkey patching for the Queue_network class
class Queue_network:
    def __init__(self, noise_evaluator):
        self.noise_evaluator = noise_evaluator

    def get_arrival_f(self, max_rate_list):
        self.arrivals_f = []
        for rate in max_rate_list:
            arrival_f = lambda t, rate=rate: poisson_random_measure(t, lambda t: 2 + np.sin(2 * np.pi * t) + self.noise_evaluator.compute_increment(), rate)
            self.arrivals_f.append(arrival_f)
            
# Running the code for the noise evaluation        
if __name__ == "__main__":

    frequency = 0.5
    mean = 0
    variance = 1
    timesteps = 100

    # Define the object of the NoiseEvaluator class
    noise_evaluator = NoiseEvaluator(frequency, mean, variance)
    
    # Define the agent and the environment configuration files
    agent = 'user_config/eval_hyperparams.yml'
    eval_env = 'user_config/configuration.yml' 
    
    # When introducing noise in the training we call the start_train method of the NoiseEvaluator object 
    noise_evaluator.start_train(eval_env, agent,save_file = True, data_filename = 'output_csv', image_filename = 'output_plots')
    
    # When introducing noise in the the control of the control of the environment we first define the agent 
    path_to_saved_agent = 'Agent/trained_agent.pt'
    saved_agent = torch.load(path_to_saved_agent)
    noise_evaluator.start_evaluation(eval_env , saved_agent,timesteps)