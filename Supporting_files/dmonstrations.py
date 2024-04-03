import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import torch 
import matplotlib.pyplot as plt
from environments.RL_Environment import *
from queueing_network import * 
import numpy as np

# This file contains all the demonstration classes for agent learning 

class config():
    # Creates the configuration object for the demonstrations 
    def __init__(self, environment, agent): 
        self.environment = environment 
        self.agent = agent 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = 0
        self.queue_metrics = [] 
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Time Steps', ylabel = self.metric, title= self.metric + f' vs Time Steps for Queue {str(self.queue_index)}')
        
    def retrieve_components(self):
        # Getter function allowing other classes to access the environment and agent objects 
        return self.environment , self.agent
                    
class Network_Control():
    """
    This class contains all the methods needed for the agent to control the network 
    Inputs: 
    - environment - testing RL Environment 
    - agent - trained agent with the learnt policy 
    """
    def __init__(self , environment , agent):
        """
            Initiates the class with the environment and the agent 
        """
        self.environment = environment
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = 0
        self.queue_metrics = [] 
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Time Steps', ylabel = self.metric, title= self.metric + f' vs Time Steps for Queue {str(self.queue_index)}')
        
        
    def configure(self,time_steps , metric, queue_index = None):
        """
            This function configures the network control parameters       
        """
        self.queue_index = queue_index
        self.time_steps = time_steps
        self.metric = metric
        
    
    def plot_queue_realtime(self):
        """
            This function continually plots the queue length at each time step in real time 
        """
        self.ax.clear()
        self.ax.plot(range(len(self.queue_metrics)), self.queue_metrics)
        self.ax.set(xlabel='Time Steps', ylabel=self.metric, title=self.metric + f' vs Time Steps for Queue {str(self.queue_index)}')
        plt.draw()
        plt.pause(0.01)
        plt.show()
        
    def plot_queue(self, *queue_metrics_lists, labels):
        """Plotting function that supports a variable number of queue metrics lists and labels."""
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(xlabel='Time Steps', ylabel=self.metric,
                    title=self.metric + f' vs Time Steps for Queue {str(self.queue_index)}')
        self.ax.legend()  # Add a legend to differentiate the lines
        plt.show()

        
    def control(self, environment=None, agent=None, time_steps=None, queue_index=None, metric=None):
        """
        This function is the main control loop for the agent and network interaction.
        """
        # Use instance attributes if no arguments are provided
        environment = environment or self.environment
        agent = agent or self.agent
        time_steps = time_steps if time_steps is not None else self.time_steps
        queue_index = queue_index if queue_index is not None else self.queue_index
        metric = metric or self.metric
        queue_metrics = []
        
        for time_step in range(time_steps): 
            state = environment.get_state()
            action = agent.select_action(state).to(self.device)
            environment.get_next_state(action)
            queue_metrics.append(environment.return_queue(queue_index, metric=metric))
            if time_step % 10 == 0:  # Corrected: to ensure it executes when time_step is a multiple of 10
                # self.plot_queue() - this would be the real time plotting logic 
                pass 
        
        # self.plot_queue()  # Plot once after completing the loop
        return queue_metrics

            
class Static_Disruption(Network_Control):
    """
    This class extends Network_Control for demonstrations where the agent has to handle disruptions
    """
    def __init__(self, environment, agent , source_node, target_node):
        super().__init__(environment, agent)
        # Initialize any additional variables or settings specific to disruptions
        self.standard_environment = environment
        self.disrupted_environment = self.deactivate_node(source_node,target_node)
        
    def deactivate_node(self,source_node, target_node):
        q_classes = self.environment.qn_net.q_classes
        q_args = self.environment.qn_net.q_args 
        edge_list = self.environment.qn_net.edge_list 
        new_class = len(q_classes)
        q_classes[new_class] = qt.LossQueue
        q_args[new_class] = {'service_f': lambda t: t+np.inf}
        edge_list[source_node][target_node] = new_class

        environment2 = self.environment.deep_copy() 
        self.environment.edge_list = edge_list
        return environment2
        
    def multi_control(self):
        """
            This function shows the agent interating with the orignal environment and the disrupted environment in parallel 
        """
        normal_metrics = self.control(environment=self.standard_environment, agent=self.agent, time_steps=self.time_steps, queue_index=self.queue_index, metric=self.metric)
        disrupted_metrics = self.control(environment=self.disrupted_environment, agent=self.agent, time_steps=self.time_steps, queue_index=self.queue_index, metric=self.metric)
        self.plot_queue(normal_metrics, disrupted_metrics, labels=['Normal', 'Disrupted'])