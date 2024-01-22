import queueing_tool as qt
import numpy as np
import tkinter as tk

class Queue_network:
    def __init__(self):
        pass

    def process_input(self, lamda_list, miu_list, active_cap, deactive_t, adjacent_list, buffer_size_for_each_queue, transition_proba= None):

        # param for first server
        self.lamda = lamda_list
        self.miu = miu_list # Should we change that to a miu for each server cause they have different service rates
        self.active_cap = active_cap
        self.deactive_cap = deactive_t

        self.get_arrival_f()
        self.get_servce_time()
        
        # Configure the network
        self.adja_list = adjacent_list
        self.edge_list = self.get_edge_list()
        print(f"This is your edge list {self.edge_list}")
        self.q_classes = self.get_q_classes()
        self.q_arg = self.get_q_arg()
        self.queues_size = buffer_size_for_each_queue # LossQueue(qbuffer=0)

        self.g = qt.adjacency2graph(adjacency=self.adja_list, edge_type=self.edge_list)

        if transition_proba is None: 
            self.transition_proba = qt.graph.generate_transition_matrix(self.g)
        else:
            self.transition_proba = transition_proba

    def get_arrival_f(self, t):
        # compute the time of next arriva given arrival rate
        ' Sample implmentation if the arrivals are following a piosson process' 
        self.arrival_f = [qt.poisson_random_measure(t, lamda, lamda) for lamda in self.lamda]
        
    def get_service_time(self, t):
        # compute the time of an agent’s service time from service rate
        self.service_f = [t + np.random.exponential(miu) for miu in self.miu]
        
    def get_edge_list(self):
        # get self.edge list from self.adj_list
        """
        example: edge_list = {0: {1: 1}, 1: {k: 2 for k in range(2, 22)}}
        """
        edge_list = {}
        for q in self.adja_list.keys():
            q_edge_list = {}
            for q_adj in range(1, self.adja_list.values()+1):
                q_edge_list[q_adj] = q
            edge_list[q] = q_edge_list

        return edge_list
    
    def get_q_classes(self):
        # create q_class from self.edge_list and specify each queue type
        """
        example: q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
        # When we have specific buffer size we have to change it as follows to LossQueue(qbuffer=0) class 
        """
        # Getting the unique types of queues 
        LossQueueList = [] 
        
        for i in range(len(self.buffer_size_for_each_queue)): 
            LossQueueList.append(qt.LossQueue(self.buffer_size_for_each_queue[i]))
        
        q_classes= {}
        for i,queue_types in enumerate(LossQueueList):
            q_classes[i]=queue_types
        # We should add the departing system queue (as the absorbing state)
        q_classes[len(LossQueueList)]=qt.NullQueue

        return q_classes
    
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
        for q_pos, q in enumerate(list(self.q_class.keys())):
            if q == 1:
                q_info = {"arrival_f": self.arrival_f[q_pos],
                        "service_f": self.service_f[q_pos],
                        "AgentFactory": qt.Agent, # use default here but according to agent's preferences
                        "active_cap": self.active_cap[q_pos],
                        "deactive_t": self.deactive_cap[q_pos]
                        }
            else:
                q_info = {"service_f": self.service_f[q_pos],
                        "AgentFactory": qt.Agent,
                        "active_cap": self.active_cap[q_pos],
                        }
            q_args[q] = q_info

        self.q_args = q_args
        # color and collect data functionality

    
    def create_env(self):
        self.queueing_network = qt.QueueNetwork(g=self.g, q_classes = self.q_classes, q_args = self.q)
        self.queueing_network.set_transition(self.transition_proba)
    
    def run_simulation(self, num_events = 50, collect_data = True):
        # specify which edges and queue to activate at the beginning
        self.queueing_network.initial()
        if collect_data:
            self.queueing_network.start_collecting_data()
            self.queueing_network.simulate(n = num_events)
            self.agent_data = self.queueing_network.get_agent_data() # check the output

    def get_reward(self):
        # Still need my Beautiful Fatima's help: get reward from self.agent_data
        pass

    def get_next_state(self):
        # Still need my Beautiful Fatima's help: return next state for the RL agent
        pass

# Very rough framework, still need my Charming man Jevon
class Interactive_Interface:
    def __init__(self, root, q_network):
        self.root = root
        self.q_network = q_network

        self.root.title("User Prompt")

        self.entry = tk.Entry(root)
        self.entry.pack()

        self.submit_button = tk.Button(root, text="Submit", command=self.retrieve_input)
        self.submit_button.pack()

    def retrieve_input(self):
        input_data = self.entry.get()
        self.q_network.process_input(input_data)

def main():
    q_network_instance = Queue_network()
    root = tk.Tk()

    interface = Interactive_Interface(root, q_network_instance)
    root.mainloop()

if __name__ == "__main__":
    main()
