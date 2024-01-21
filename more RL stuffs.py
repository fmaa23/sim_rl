"""
initialise Q(s,a) --> table of Q values

initialise Model(s,a) --> given s and a, returns S and R that is PREDICTED by the agent
(this is not from the simulated environment)
"""

"""
A: list of states, each state is represented by a 2 tuple (service time, queue size)
- number of rows = number of nodes
- number of columns = 2 (service time, queue size)
we will reference a state by its index in this matrix A

but maybe service time should be a constant that is part of the environment? 
"""



"""
Q: 2D matrix
- number of rows = number of states, each state will be referred to by its index from A
- number of columns = number of actions possible
but this is will be very sparse and computationally inefficient

although
"""



"""
Model: 2D matrix
same dimensions as Q

problem: initialisation

if we are at layer 1, and we took action a that changes routing proportion between
layer 1 and layer 2, then only the service time and queue size of nodes in layer 2 should
change --> can use this to help in initalisation, making Model(s,a) converge faster
during learning, i.e. the agent is able to make a good prediction of the next state
and next reward with fewer iterations
"""



"""
We think reward R = throughput rate - baseline

Loop until abs(throughput rate - baseline) < epsilon
"""


"""
What we need:

Given the
(a) routing ratios for every node,
(b) service time for each node (probably constant in the simulation), and
(c) a defined arrival rate,
calculate the throughput rate. We need this because we will set the reward function
to be R = throughput rate - baseline.


A list of all possible states of the system. And, for each possible state of the
system, we also need the action space.

"""

import numpy as np
def main():
    dyna_q(num_episodes, alpha, gamma, epsilon, n)


def epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])
    
def initialise_Model(s,a):
    // ?

def next_state(s,a):






def dyna_q(num_episodes, alpha, gamma, epsilon, n,A,actions,threshold):
    # Assuming A is a set of states and actions(s) is a dictionary mapping each state to its set of actions
    Q = np.zeros((len(A), max(len(actions[s]g) for s in A)))
    Model = {(s, a): initialise_Model(s,a) for s in A for a in actions[s]}
    #or can initialize it to 0,0
    Model = {(s, a): (0,0) for s in A for a in actions[s]}


    for episode in range(num_episodes):
        state = np.random.choice(list(A))  # (a) Start in a random state
        action = epsilon_greedy(state, Q, epsilon)  # (b) Choose action using epsilon-greedy

        while abs(reward)>threshold:
            # (c) Take action, observe reward and next state
            next_state = next_state(state,action)  
            reward = REWARD(state,action)

            # (d) Direct RL update
            Q[state, action] += alpha * (reward + gamma * np.min(Q[next_state,action]) - Q[state, action])

            # (e) Model learning update
            Model[(state, action)] = (reward, next_state)

            # (f) Q-planning updates
            for _ in range(n):
                sampled_state, sampled_action = np.random.choice(list(Model.keys()))
                sampled_reward, sampled_next_state = Model[(sampled_state, sampled_action)]

                Q[sampled_state, sampled_action] += alpha * (
                        sampled_reward + gamma * np.min(Q[sampled_next_state]) - Q[sampled_state, sampled_action])

            
            state = next_state
            action = epsilon_greedy(state, Q, epsilon)

    return Q



"""
AARON:
"""

class Node:
    def __init__(self, buffer_size, service_time):
        self._buffer_size = buffer_size
        self._service_time = service_time
        self.queue_length = 0

class System:
    def __init__(self, nodes):
        self.nodes = nodes
    
    def step(self, state, action):    
        reward = None
    
        return reward

    def reset(self):
        for node in self.nodes:
            node.queue_length = 0


def dyna_q(n_episodes, alpha, gamma, epsilon_i, n, nodes, A):
    """
    Function which trains the Dyna-Q learning agent.
    -   state = node that agent is at
    -   action = routing proportion at this node (action space is different for
        every node)
    -   reward = service_time*queue_length, summed over all nodes and *(-1)
    -   Q and A are DICTIONARIES rather than tables because action space is
        different for every node
    
    State and actions are identified by their indices. States: keys in the
    dictionaries Q and A. Actions: indices in the numpy array Q[key] or A[key]

    Args:
        n_episodes (int)    : number of epsiodes to train agent on
        alpha (float)       : learning rate (alpha)
        gamma (float)       : discount factor
        epsilon_i (float)   : initial value of epsilon
        n (int)             : number of times planning loop is executed
        nodes (list)        : list of node instances
        A (dict)            : key = node index, value = np.ndarray of actions
    """
    keys = np.arange(len(nodes))     # indices from 0, ... n_nodes-1
    Q = dict.fromkeys(keys, 0)
    for k in A.keys():
        Q[k] = np.zeros(A[k].shape)

    # initialise Model(s,a)
    epsilon = epsilon_i
    for episode in range(n_episodes):
        state = np.random.choice(len(nodes))            # (a)
        action = epsilon_greedy(state, Q, epsilon)      # (b)
        next_state, reward = step(state, action)        # (c)

        # (d)
        Q[state][action] += alpha*(reward + \
                                   gamma*np.max(Q[next_state][action]) - \
                                    Q[state][action])

        Model[(state, action)] = (reward, next_state)   # (e)

        # (f)
        for _ in range(n):
            pass





if __name__ == "__main__":
    main()
