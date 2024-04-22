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
from rl_env.RL_Environment import *
from queue_env.queueing_network import * 
<<<<<<< HEAD:features/blockage_demonstrations/demonstrations_old.py
from queueing_tool.queues.queue_servers import *
=======
>>>>>>> 85c17b78781cc3f77c2b32b6dd335b457d39191d:blockage_demonstrations/demonstrations.py
import numpy as np
import copy
import os 
from queueing_tool.queues.queue_servers import *

class config():
    # Creates the configuration object for the demonstrations 
    def __init__(self, environment, agent, queue_index, metric='throughput'): 
        self.environment = environment 
        self.agent = agent 
        self.metric = metric 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = queue_index
        self.queue_metrics = [] 
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Time Steps', ylabel = self.metric, title= self.metric + f' vs Time Steps for Queue {self.queue_index}')
        
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
    def __init__(self , environment , agent, queue_index, metric='throughput'):
        """
            Initiates the class with the environment and the agent 
        """
        self.environment = environment
        self.agent = agent
        self.metric = metric 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = queue_index
        self.queue_metrics = [] 
        self.call_plot_num =0
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
        
    def plot_transition_proba(self, transition_proba_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.ax.clear()  # Clear previous plots
        self.ax.plot(transition_proba_lists)
        self.ax.set(xlabel='Time Steps', ylabel='Transition Probability',
                title=f'Transition Probability vs Time Steps for Queue {self.queue_index}')
        parent_path = os.getcwd()
        figure_path = os.path.join(parent_path, 'blockage_demonstrations', 'results', 'Normal_Transition_Proba.png')
        directory = os.path.dirname(figure_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(figure_path)
        self.call_plot_num+=1

    def plot_queue(self, labels, *queue_metrics_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(xlabel='Time Steps', ylabel=self.metric,
                title=self.metric + f' vs Time Steps for Queue {self.queue_index}')
        parent_path = os.getcwd()
        figure_path = os.path.join(parent_path, 'blockage_demonstrations', 'results', 'Normal_Plot_Queue.png')
        directory = os.path.dirname(figure_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(figure_path)
        self.call_plot_num+=1


        
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
        source_edge = environment.net.edge2queue[queue_index].edge[0]
        target_edge = environment.net.edge2queue[queue_index].edge[1]
        queue_metrics = []
        queue_transition_proba = [] 
        
        for time_step in range(time_steps): 
            state = environment.get_state()
            action = agent.actor(state).detach()
            state =environment.get_next_state(action)
            queue_metrics.append(environment.return_queue(queue_index, metric=metric))
            queue_transition_proba.append(environment.transition_proba[source_edge][target_edge])
        
        self.plot_queue(metric,queue_metrics)  # Plot once after completing the loop
        self.plot_transition_proba(queue_transition_proba)
        return queue_metrics

class Static_Disruption(Network_Control):
    """
    This class extends Network_Control for demonstrations where the agent has to handle disruptions
    """
    def __init__(self, environment, agent , queue_index, source_node, target_node):
        super().__init__(environment, agent, queue_index)
        # Initialize any additional variables or settings specific to disruptions
        self.standard_environment = environment
        self.disrupted_environment = self.deactivate_node(source_node,target_node)
        
    def deactivate_node(self,source_node, target_node):
        
        q_classes = self.environment.qn_net.q_classes
        q_args = self.environment.qn_net.q_args 
        edge_list = self.environment.qn_net.edge_list 
        new_class = len(q_classes)
        q_classes[new_class] = LossQueue
        q_args[new_class] = {'service_f': lambda t: t+np.inf}
        edge_list[source_node][target_node] = new_class

        org_net = self.environment.qn_net
        new_net = copy.copy(org_net)
        new_net.process_input(org_net.lamda, org_net.miu, q_classes, q_args, org_net.adja_list, 
<<<<<<< HEAD:features/blockage_demonstrations/demonstrations_old.py
                        edge_list, org_net.transition_proba, max_agents = float('inf'), sim_jobs = 100)
=======
                        edge_list, org_net.transition_proba, org_net.max_agents, org_net.sim_time)
>>>>>>> 85c17b78781cc3f77c2b32b6dd335b457d39191d:blockage_demonstrations/demonstrations.py
        new_net.create_env()
        dis_environment = RLEnv(qn_net=new_net, num_sim=100)
        
        return dis_environment
    
    def plot_queue(self, labels, *queue_metrics_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.fig, self.ax = plt.subplots()
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(xlabel='Time Steps', ylabel=self.metric,
                title=self.metric + f' vs Time Steps for Queue {self.queue_index}')
        parent_path = os.getcwd()
        figure_path = os.path.join(parent_path, 'blockage_demonstrations', 'results', 'Blocked_Plot_Queue.png')
        directory = os.path.dirname(figure_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(figure_path)
        self.call_plot_num+=1

    def plot_transition_proba(self, transition_proba_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels.""" 
        self.ax.clear()  # Clear previous plots
        self.ax.plot(transition_proba_lists)
        self.ax.set(xlabel='Time Steps', ylabel='Transition Probability',
                title=f'Blocked Transition Probability vs Time Steps for Queue {self.queue_index}')
        parent_path = os.getcwd()
        figure_path = os.path.join(parent_path, 'blockage_demonstrations', 'results', 'Blocked_Transition_Proba.png')
        directory = os.path.dirname(figure_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(figure_path)
        self.call_plot_num+=1
        
    def multi_control(self):
        """
            This function shows the agent interating with the orignal environment and the disrupted environment in parallel 
        """
        normal_metrics = self.control(environment=self.standard_environment, agent=self.agent, time_steps=self.time_steps, queue_index=self.queue_index, metric=self.metric)
        disrupted_metrics = self.control(environment=self.disrupted_environment, agent=self.agent, time_steps=self.time_steps, queue_index=self.queue_index, metric=self.metric)
        self.plot_queue(normal_metrics, disrupted_metrics, labels=['Normal', 'Disrupted'])

if __name__=="__main__": 
    agent = torch.load('Agent/trained_agent.pt')
    config_param_filepath = 'user_config/configuration.yml'
    eval_param_filepath = 'user_config/eval_hyperparams.yml'
<<<<<<< HEAD:features/blockage_demonstrations/demonstrations_old.py
    env = create_simulation_env({'num_sim':100}, config_param_filepath)
    #env = get_env(n=5000)
    if False:
        nc = Network_Control(agent, env)
        nc.plot_queue_realtime()
        queue_metrics = nc.control(environment=env, agent=agent, time_steps=100, queue_index=2, metric='throughput')
=======
    env = create_simulation_env({'num_sim':5000}, config_param_filepath)
    nc = Network_Control(agent, env, queue_index=1)
    queue_metrics = nc.control(environment=env, agent=agent, time_steps=100, queue_index=1, metric='throughput')
>>>>>>> 85c17b78781cc3f77c2b32b6dd335b457d39191d:blockage_demonstrations/demonstrations.py

    ## Static Disruption 
    sd = Static_Disruption(env, agent, queue_index=1, source_node=1, target_node=2)
    disrupted_env = sd.disrupted_environment
<<<<<<< HEAD:features/blockage_demonstrations/demonstrations_old.py
    queue_metrics_dis = sd.control(environment=disrupted_env, agent=agent, time_steps=100, queue_index=2, metric='throughput')
    #breakpoint()
=======
    queue_metrics_dis = sd.control(environment=disrupted_env, agent=agent, time_steps=100, queue_index=1, metric='throughput')
>>>>>>> 85c17b78781cc3f77c2b32b6dd335b457d39191d:blockage_demonstrations/demonstrations.py
