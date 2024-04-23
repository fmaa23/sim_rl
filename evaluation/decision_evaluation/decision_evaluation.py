import sys
import os
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
parent_dir = os.path.dirname(os.path.dirname(root_dir))
os.chdir(parent_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch 
import matplotlib.pyplot as plt
from foundations.supporting_functions import *
from foundations.supporting_functions import Engine 
from rl_env.RL_Environment import *
from queue_env.queueing_network import * 
from queueing_tool.queues.queue_servers import *
import numpy as np
import copy
import os 
import yaml

class save_plots:
    """
    Load configuration parameters from a YAML file.

    Parameters:
    - config_filepath (str): The file path to the configuration YAML file.

    Returns:
    - dict: A dictionary containing the configuration parameters.
    """
    def __init__(self):
        self.param_file = "user_config\\features_params\\blockage_demonstration_params.yml"
        self.save_file = "features\\blockage_demonstrations\output_plots"
    
    def save(self, queue_transition_proba_before_disrupt, queue_transition_proba_after_disrupt):
        trans_proba_changes = queue_transition_proba_before_disrupt + queue_transition_proba_after_disrupt
        plt.figure()
        plt.plot(trans_proba_changes)
        plt.title(f"Routing Proba Changes Before/After Disruption for Queue_{queue_index}")

        current_dir = os.getcwd() + "MScDataSparqProject"
        features_dir = 'features/blockage_demonstrations/output_plots'
        data_filename = "Transition_Proba_BeforeAfter_Blockage.png"
        save_filepath = os.path.join(current_dir, features_dir, data_filename)
        plt.savefig(save_filepath)

class ControlEvaluation(Engine):
    """
    This class contains all the methods needed for the agent to control the network 
    Inputs: 
    - environment - testing RL Environment 
    - agent - trained agent with the learnt policy 
    """
    def __init__(self, queue_index, metric='throughput'):
        """
            Initiates the class with the environment and the agent 
        """

        self.config_param_filepath = 'user_config/configuration.yml'
        self.environment = create_simulation_env({'num_sim':sim_jobs}, self.config_param_filepath, disrupt_case = True)

        self.agent = agent
        self.metric = metric 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = queue_index
        self.queue_metrics = [] 
        self.call_plot_num = 0
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Time Steps', ylabel = self.metric, title= self.metric + f' vs Time Steps for Queue {str(self.queue_index)}')
    
        
    
        
    def plot_transition_proba(self, transition_proba_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.ax.clear()  # Clear previous plots
        self.ax.plot(transition_proba_lists)
        self.ax.set(xlabel='Time Steps', ylabel='Transition Probability',
                title=f'Transition Probability vs Time Steps for Queue {self.queue_index}')
        self.ax.legend()  # Add a legend to differentiate the lines
        figure_name = f'{self.call_plot_num}_plot_transition_proba.png'

        current_dir = os.getcwd() + "MScDataSparqProject"
        features_dir = 'features/blockage_demonstrations/output_plots'
        save_filepath = os.path.join(current_dir, features_dir, figure_name)

        plt.savefig(save_filepath)
        self.call_plot_num+=1

    def plot_queue(self, labels, *queue_metrics_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(xlabel='Time Steps', ylabel=self.metric,
                title=self.metric + f' vs Time Steps for Queue {self.queue_index}')
        self.ax.legend()  # Add a legend to differentiate the lines
        figure_name = f'{self.call_plot_num}_plot_queue.png'

        current_dir = os.getcwd() + "MScDataSparqProject"
        features_dir = 'features/blockage_demonstrations/output_plots'
        save_filepath = os.path.join(current_dir, features_dir, figure_name)

        plt.savefig(save_filepath)
        self.call_plot_num+=1

    def start_evaluation(self, environment, agent, time_steps, num_simulations):
        source_edge = self.environment.net.edge2queue[queue_index].edge[0]
        target_edge = self.environment.net.edge2queue[queue_index].edge[1]
        queue_metrics = []
        queue_transition_proba = [] 
        self.environment.simulate()
        for time_step in range(time_steps): 
            state = self.environment.get_state()
            action = agent.actor(state).detach()
            state = self.environment.get_next_state(action)[0]
            queue_metrics.append(self.environment.return_queue(queue_index, metric=metric))
            queue_transition_proba.append(self.environment.transition_proba[source_edge][target_edge])
        self.plot_queue(metric,queue_metrics)  # Plot once after completing the loop
        self.plot_transition_proba(queue_transition_proba)
        
        return queue_metrics, queue_transition_proba


class DisruptionEvaluation(ControlEvaluation):
    """
    This class extends Control Evaluation for demonstrating scenarios where the agent has to handle disruptions
    """
    def __init__(self, agent, sim_jobs):
        super().__init__(agent , queue_index, metric)
        # Initialize any additional variables or settings specific to disruptions
        self.queue_index = 2

        self.standard_environment = create_simulation_env({'num_sim':sim_jobs}, self.config_param_filepath, disrupt_case = True)
        self.disrupted_environment = create_simulation_env(params = {'num_sim':sim_jobs}, 
                                                           config_file = self.config_param_filepath, 
                                                           disrupt_case = True, disrupt = True, queue_index = self.queue_index)
        self.environment = self.disrupted_environment

    def plot_queue(self, labels, *queue_metrics_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.fig, self.ax = plt.subplots()
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(xlabel='Time Steps', ylabel=self.metric,
                title=self.metric + f' vs Time Steps for Queue {self.queue_index}')
        self.ax.legend()  # Add a legend to differentiate the lines
        figure_name = f'{self.call_plot_num}_plot_queue_sd.png'
        #plt.show()
        plt.savefig(figure_name)
        self.call_plot_num+=1

    def plot_transition_proba(self, transition_proba_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.ax.clear()  # Clear previous plots
        self.ax.plot(transition_proba_lists)
        self.ax.set(xlabel='Time Steps', ylabel='Transition Probability',
                title=f'Transition Probability vs Time Steps for Queue {self.queue_index}')
        self.ax.legend()  # Add a legend to differentiate the lines
        figure_name = f'{self.call_plot_num}_plot_transition_proba_sd.png'
        plt.savefig(figure_name)
        self.call_plot_num+=1
        
    def multi_control(self):
        """
            This function shows the agent interacting with the orignal environment and the disrupted environment in parallel 
        """
        normal_metrics = self.start_evaluation(environment=self.standard_environment, 
                                      agent=self.agent, time_steps=self.time_steps) 
        
        disrupted_metrics = self.start_evaluation(environment=self.disrupted_environment, 
                                         agent=self.agent, time_steps=self.time_steps, 
                                         queue_index=self.queue_index, metric=self.metric)
        
        self.plot_queue(normal_metrics, disrupted_metrics, labels=['Normal', 'Disrupted'])
        
        

if __name__=="__main__": 

    
    sim_jobs = 100
    time_steps = 100
    queue_index = 2
    metric = 'throughput'

    config_param_filepath = 'user_config/configuration.yml'
    env = create_simulation_env({'num_sim':sim_jobs}, config_param_filepath)
    agent = torch.load('Agent/trained_agent.pt')

    # No Disruption
    nc = ControlEvaluation(queue_index = queue_index, metric=metric)
    nc.start_evaluation(environment=env , agent=agent, time_steps=time_steps, num_simulations=sim_jobs)
    

    ## Static Disruption 
    sd = DisruptionEvaluation(agent, sim_jobs)
    queue_metrics_dis, queue_transition_proba_after_disrupt = sd.multi_control()

    # Save Plot
    queue_metrics, queue_transition_proba_before_disrupt = None , None 
    save_plots().save(queue_transition_proba_before_disrupt, queue_transition_proba_after_disrupt)
    
    
    
    ### CHANGES ### 
    # 1. Add functionality for saving the plots to a specified file path 
    # 2. Ensure that the function calls in the static disruption are correct 
    # 3. Clean up names 
    # Change to OOP structure