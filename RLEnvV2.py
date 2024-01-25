import queueing_tool as qt 
import numpy as np

from numpy.random import uniform
from queueing_tool.queues.choice import _choice, _argmin

class RLEnv: 
    def __init__(self, net, start_state=None, n=5000): 
        # The queue network to optimize actions on 
        self.net = net # Assuming it is a queuing network applied on the sample 
        self.transition_proba = self.net.transitions(False)  # Equal proportion for next nodes 
        self.sim_n = n # Take next step (num_events)
        self.iter = 1
        # self._t = 0 
        self.departure_nodes = 1 # For now set to 0, should be changed to be got from the net structure 

        # Getting the queue type and ID
        self.current_queue_id = 0 # Edge Index, assuming we always starting as the 
        self.current_source_vertex = net.edge2queue[self.current_queue_id].edge[0] # Source Queue Vertex, the source node 
        self.current_edge_tuple = net.edge2queue[self.current_queue_id].edge # (source_vertex, target_vertex, edge_index, type)
        self.current_queue = net.edge2queue[self.current_queue_id]

        # Starting simulation to allow for external arrivals
        self.net.start_collecting_data()

        # Simulation is specified by time in seconds, the number of events will depend on the arrival rate
        self.net.initialize(queues=0)
        self.net.simulate(n=n)

        # The starting queue length (backlog) at the all nodes in a numpy array
        if start_state is None: 
            self._state = np.zeros(net.num_edges-self.departure_nodes)
        
        # Time since simulation start 
        # self._t += simulation_time 
    def get_net_connections(self):
        return self.transition_proba


    def get_state(self):
        # Returns the queue length (backlog) in the current state queue ()
        for edge in range(self.net.num_edges-self.departure_nodes):
            edge_data = self.net.get_queue_data(queues=edge)
            self._state[edge]=edge_data[-1][4]
        return self._state     

    def get_next_state(self, action, state=None):

        # Add dim compatibility test 

        # Resetting conditions when a departure node is reached or when the simulation max times is exceeded 
        if self.current_edge_tuple[-1]==0 or self.iter>=25: # Trial and Error  
            self.reset()

        if state is None:
            state=self.current_source_vertex

        for i, next_node in enumerate(self.transition_proba[state].keys()):
            self.transition_proba[state][next_node]= action[i]

        self.iter +=1
        self.net.start_collecting_data()
        self.net.initialize()
        self.net.simulate(n=int(self.sim_n*self.iter))
        agent = qt.queues.Agent

        new_queue_id = agent.desired_destination(agent, network=self.net, edge=self.current_edge_tuple)
        self.current_queue_id = new_queue_id
        self.current_source_vertex = self.net.edge2queue[self.current_queue_id].edge[0] # Source Queue Vertex, the source node 
        self.current_edge_tuple = self.net.edge2queue[self.current_queue_id].edge
        self.current_queue = self.net.edge2queue[self.current_queue_id]

        return self.get_state(), new_queue_id, self.current_edge_tuple # Convert as a dictionary

    def get_reward(self):
        if isinstance(self.current_queue, qt.NullQueue):
            pass
        else: 
            # Loop over nodes: 
               # Average service time for each node * queue length for each node  
            queue_data = self.get_state()
            reward_array=[]
            # Returns the queue length (backlog) in the current state queue ()
            for edge in range(self.net.num_edges-self.departure_nodes): 
                service_time=0
                edge_data = self.net.get_queue_data(queues=edge)
                for points in edge_data:
                    service_time += abs(points[2]-points[1])
                service_time /= len(edge_data)
                reward_queue = queue_data[edge] * service_time
                reward_array.append(reward_queue)
        return reward_array
                

    def reset(self): 
        self.net.clear_data()
        self.__init__(self.net)
        
import queueing_tool as qt
import numpy as np
adja_list = {0:[1],1:[2,3], 2:[4], 3:[4]}
edge_list = {0: {1:1}, 1:{2:2, 3:2}, 2: {4: 3}, 3: {4: 2}}
# edge_list contains the source queue as key the value is a dict, 
# where each key is the respective target to that source and the value is the type of the connection 
g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)

def rate(t):
    return 250 + 350 * np.sin(np.pi * t / 2)**2
def arr_f(t):
    return qt.poisson_random_measure(t, rate, 375)
def ser_f(t):
    return t + np.random.exponential(0.0001)

Q_type =qt.QueueServer

q_classes = {1:Q_type, 2: Q_type}
q_args    = {
    1: {
        'arrival_f': arr_f,
        'service_f': lambda t: t+100,
        'AgentFactory': qt.Agent
    },
    2: {
        'num_servers': 1,
        'service_f': ser_f,
    },
    
    3: {
        'num_servers': 1,
        'service_f': lambda t:t+300,
    }
}
qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)

# Questions to ask: 
# How often do we reset
# What is a good simulation time
# The reward is calculated as a cost of how many states or should I make the reward function time dependent?

breakpoint()
