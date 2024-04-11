import numpy as np
import tkinter as tk
import yaml 
import sys
import os

from queueing_tool.network.queue_network import QueueNetwork
from queueing_tool.graph.graph_wrapper import adjacency2graph
from queueing_tool.queues.agents import Agent
from queueing_tool.queues.queue_servers import *

class Queue_network:
    def __init__(self):
        pass
    
    def process_config(self, filename):
        """
            This function accepts the name of the yaml file as the input and returns the variables for the process_input function
            
            Input : filename (str) : Name of the yaml file
            Output : lamda_list (list) : List of arrival rates for each queue
                     miu_list (list) : List of service rates for each queue
                     active_cap (int) : Active capacity of the server
                     deactive_t (float) : Deactivation time
                     adjacent_list (dict) : Adjacency list of the network
                     buffer_size_for_each_queue (list) : List of buffer sizes for each queue
                     transition_proba (dict) : Transition probability matrix
        """
        parameters = (open(filename, 'r'))
        parameter_dictionary = yaml.load(parameters, Loader=yaml.FullLoader)
        lambda_list = parameter_dictionary['lambda_list']
        lambda_list = [float(i) for i in lambda_list]
        miu_list = parameter_dictionary['miu_list']
        miu_list = [float(i) for i in miu_list]
        active_cap = parameter_dictionary['active_cap']
        active_cap = float(active_cap)
        deactive_cap = parameter_dictionary['deactive_cap']
        deactive_cap = float(deactive_cap)
        adjacent_list = parameter_dictionary['adjacent_list']
        adjacent_list = {int(k): [int(i) for i in v] for k, v in adjacent_list.items()}
        buffer_size_for_each_queue = parameter_dictionary['buffer_size']
        buffer_size_for_each_queue = [int(i) for i in buffer_size_for_each_queue]
        if 'transition_proba' in parameter_dictionary.keys():
            transition_proba = parameter_dictionary['transition_proba']
        else:
            transition_proba = None
        return lambda_list, miu_list, active_cap, deactive_cap, adjacent_list, buffer_size_for_each_queue, transition_proba
    
    def process_input(self, arrival_rate, miu_list, q_classes, q_args, adjacent_list, 
                        edge_list, transition_proba):

        # param for first server
        self.lamda = arrival_rate
        self.miu = miu_list 
        
        # Configure the network
        self.adja_list = adjacent_list
        
        self.edge_list = edge_list
        
        self.q_classes = q_classes
        
        self.q_args = q_args

        if transition_proba is None: 
            self.transition_proba = generate_transition_matrix(self.g)
        else:
            self.transition_proba = transition_proba

    def get_arrival_f(self):
        # compute the time of next arriva given arrival rate 
        self.arrivals_f = []
        max_rate = 375
        rate = lambda t: 2 + np.sin(2 * np.pi * t)
        arrival_f = lambda t: poisson_random_measure(t, rate, max_rate)
        self.arrivals_f = arrival_f
        
    def get_service_time(self):
        # compute the time of an agent’s service time from service rate
        self.services_f = []
        for miu in self.miu:
            def ser_f(t):
                return t + np.random.exponential(miu)
            self.services_f.append(ser_f)
        return self.services_f
        
    def get_edge_list(self):
        # get self.edge list from self.adj_list
        """
        example: edge_list = {0: {1: 1}, 1: {k: 2 for k in range(2, 22)}}
        """
        self.edge_list = {}
        edge = 1
        for q in self.adja_list.keys():
            q_edge_list = {}
            for q_adj in self.adja_list[q]:
                q_edge_list[q_adj] = edge
                edge += 1
            self.edge_list[q] = q_edge_list
            
    def get_q_classes(self):
        """
        example: q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
        # When we have specific buffer size we have to change it as follows to LossQueue(qbuffer=0) class 
        """
        LossQueueList = [] 
        
        for i in range(len(self.buffer_size_for_each_queue)): 
            if i == 0:
                LossQueueList.append(QueueServer)
            else:
                LossQueueList.append(LossQueue)
        
        self.q_classes= {}
        for i,queue_types in enumerate(LossQueueList):
            if i == 1:
                print("Loss Queue Object:", queue_types)
            self.q_classes[i+1]=queue_types
    
    def get_q_arg(self):
        """
        example: q_args = {
                        1: {
                            'arrival_f': arr_f,
                            'service_f': lambda t: t,
                            'AgentFactory': qt.GreedyAgent
                        },
                        2: {
                            'num_servers': 1,
                            'service_f': ser_f
                        }
                        }
        """
        q_args = {}
        for index, q in enumerate(list(self.q_classes.keys())):
            if q == 1:
                q_info = {"arrival_f": self.arrivals_f,
                        "service_f": self.services_f[index],
                        "AgentFactory": Agent, 
                        "active_cap": self.active_cap,
                        "deactive_t": self.deactive_cap,
                        # 'qbuffer':self.buffer_size_for_each_queue[index]
                        }
            else:
                q_info = {"service_f": self.services_f[index],
                        "qbuffer":self.buffer_size_for_each_queue[index],
                        "AgentFactory": Agent
                        }
            q_args[q] = q_info

        self.q_args = q_args
    
    def create_env(self):

        self.g = adjacency2graph(adjacency=self.adja_list, edge_type=self.edge_list, adjust = 2)
        self.queueing_network = QueueNetwork(g=self.g, q_classes = self.q_classes, q_args = self.q_args)
        self.queueing_network.set_transitions(self.transition_proba)
        self.queueing_network.draw(figsize=(6, 3))
    
    def run_simulation(self, num_events = 50, collect_data = True):
        # specify which edges and queue to activate at the beginning
        self.queueing_network.initial()
        if collect_data:
            self.queueing_network.start_collecting_data()
            self.queueing_network.simulate(n = num_events)
            self.agent_data = self.queueing_network.get_agent_data() # check the output