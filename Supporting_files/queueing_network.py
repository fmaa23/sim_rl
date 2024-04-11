import queueing_tool as qt
import numpy as np
import tkinter as tk
import yaml 

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
            self.transition_proba = qt.graph.generate_transition_matrix(self.g)
        else:
            self.transition_proba = transition_proba

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
            
    
    def create_env(self):

        self.g = qt.adjacency2graph(adjacency=self.adja_list, edge_type=self.edge_list, adjust = 2)
        self.queueing_network = qt.QueueNetwork(g=self.g, q_classes = self.q_classes, q_args = self.q_args)
        self.queueing_network.set_transitions(self.transition_proba)
        self.queueing_network.draw(figsize=(6, 3))
    
    def run_simulation(self, num_events = 50, collect_data = True):
        # specify which edges and queue to activate at the beginning
        self.queueing_network.initial()
        if collect_data:
            self.queueing_network.start_collecting_data()
            self.queueing_network.simulate(n = num_events)
            self.agent_data = self.queueing_network.get_agent_data() # check the output