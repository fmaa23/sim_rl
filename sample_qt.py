import queueing_tool as qt
import numpy as np
adja_list = {0: [1], 1: [k for k in range(2, 22)]}
edge_list = {0: {1: 1}, 1: {k: 2 for k in range(2, 22)}}
# edge_list contains the source queue as key the value is a dict, 
# where each key is the respective target to that source and the value is the type of the connection 
g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)

def rate(t):
    return 25 + 350 * np.sin(np.pi * t / 2)**2
def arr_f(t):
    return qt.poisson_random_measure(t, rate, 375)
def ser_f(t):
    return t + np.random.exponential(0.2 / 2.1)

q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
q_args    = {
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
qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)


qn.initialize(edge_type=1)
qn.start_collecting_data()
qn.simulate(t=1.9)
data = qn.get_queue_data()
print(qn.edge2queue[40].edge)







