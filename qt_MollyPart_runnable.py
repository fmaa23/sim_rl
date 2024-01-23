import queueing_tool as qt
import numpy as np
import tkinter as tk

class Queue_network:
    def __init__(self):
        pass

    def process_input(self, lamda_list, miu_list, active_cap, deactive_t, adjacent_list, buffer_size_for_each_queue, transition_proba= None):

        # param for first server
        self.node_ids = list(range(1, len(lamda_list) + 1))
        self.lamda = lamda_list
        self.miu = miu_list # Should we change that to a miu for each server cause they have different service rates
        self.active_cap = active_cap
        self.deactive_cap = deactive_t
        self.buffer_size_for_each_queue = buffer_size_for_each_queue
        
        self.get_arrival_f()
        self.get_service_time()
        
        # Configure the network
        self.adja_list = adjacent_list
        print("adja_list:", self.adja_list)
        
        self.get_edge_list()
        print("edge list:", self.edge_list)
        
        self.get_q_classes()
        print("q_classes:", self.q_classes)
        
        self.get_q_arg()
        print("q_arg:", self.q_args)

        if transition_proba is None: 
            self.transition_proba = qt.graph.generate_transition_matrix(self.g)
        else:
            self.transition_proba = transition_proba

    def get_arrival_f(self):
        # compute the time of next arriva given arrival rate 
        self.arrivals_f = []
        max_rate = 3
        rate = lambda t: 2 + np.sin(2 * np.pi * t)
        for node_id in self.node_ids:
            arrival_f = lambda t: qt.poisson_random_measure(t, rate, max_rate)
        self.arrivals_f.append(arrival_f)
        
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
        # create q_class from self.edge_list and specify each queue type
        """
        example: q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
        # When we have specific buffer size we have to change it as follows to LossQueue(qbuffer=0) class 
        """
        # Getting the unique types of queues 
        LossQueueList = [] 
        
        for i in range(len(self.buffer_size_for_each_queue)): 
            LossQueueList.append(qt.LossQueue)
        
        self.q_classes= {}
        for i,queue_types in enumerate(LossQueueList):
            if i == 1:
                print("Loss Queue Object:", queue_types)
            self.q_classes[i+1]=queue_types
        # We should add the departing system queue (as the absorbing state)
        self.q_classes[len(LossQueueList)]=qt.NullQueue
    
    def get_q_arg(self):
        # create self.q_args from self.q_classes
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
                q_info = {"arrival_f": self.arrivals_f[index],
                        "service_f": self.services_f[index],
                        "AgentFactory": qt.Agent, 
                        "active_cap": self.active_cap,
                        "deactive_t": self.deactive_cap
                        }
            else:
                q_info = {"service_f": self.services_f[index],
                        "AgentFactory": qt.Agent
                        }
            q_args[q] = q_info

        self.q_args = q_args
    
    def create_env(self):
        g = qt.adjacency2graph(adjacency=self.adja_list, edge_type=self.edge_list)
        self.queueing_network = qt.QueueNetwork(g=g, q_classes = self.q_classes, q_args = self.q_args)
        self.queueing_network.set_transitions(self.transition_proba)
        self.queueing_network.draw(figsize=(6, 3))
    
    def run_simulation(self, num_events = 50, collect_data = True):
        # specify which edges and queue to activate at the beginning
        self.queueing_network.initial()
        if collect_data:
            self.queueing_network.start_collecting_data()
            self.queueing_network.simulate(n = num_events)
            self.agent_data = self.queueing_network.get_agent_data() # check the output
    
def main():

    # example input
    lamda_list = [0.2, 0.2, 0.2, 0.2]
    miu_list = [0.1, 0.1, 0.1, 0.2]
    active_cap = 5
    deactive_t = 0.12
    adjacent_list = {0:[1,2], 1:[3], 2:[3]}
    buffer_size_for_each_queue = [10, 10, 10, 10]
    transition_proba= {0:{1:0.5, 2:0.5}, 1:{3:1}, 2:{3:1}}

    q_net = Queue_network()
    q_net.process_input(lamda_list, miu_list, active_cap, deactive_t, adjacent_list, buffer_size_for_each_queue, transition_proba)
    q_net.create_env()

if __name__ == "__main__":
    main()
