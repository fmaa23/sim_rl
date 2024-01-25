import numpy as np
from itertools import product
from RLEnvV2 import get_env


class Node:
    def __init__(self, buffer_size, service_time):
        self._buffer_size = buffer_size
        self._service_time = service_time
        self.queue_length = 0

def generate_action_space(adja_list, interval):
    """
    Args:
        adja_list (dict):   Here, key = node index, value = np.array of nodes
                            connected downstream to this node (identified by indices) e.g. adja_list = {0:[1], 1:[2,3,4]}
                            means node 1 is connected to nodes 2,3,4
        interval (float):   How action space will be discretised e.g. interval
                            = 0.2 means routing proportions can only change
                            in increments/decrements of 0.2

    Returns:
        A (dict)        :   Action space of problem as a dictionary. Key = node
                            index. value = nested np.ndarray. len(value) = no.
                            of nodes, and each element is a np.ndarray of the
                            appropriate size representing the routing
                            proportions at this node.

    Example. If node with index 0 is connected to two downstream nodes, with
    interval = 0.2, the output is of the form:
                            
    A = {   0: [ [0,1], [0.2,0.8], [0.4,0.6], [0.6,0.4], [0.8,0.2], [1,0] ], 
            1: ...
        }
    """
    A = {}
    values = np.arange(0, 1+interval, interval)
    for node in adja_list.keys():
        n = len(adja_list[node])        # no. of nodes connected downstream
        actions = [np.array(x) for x in product(*[values]*n) if sum(x) == 1]
        A[node] = np.array(actions)
    return A


def epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])
    

def dyna_q(n_episodes, alpha, gamma, epsilon_i, n, A, env):
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
    # Initialise Q(s,a) and Model(s,a) for all states and actions
    keys = env.adja_list.keys()                                         # indices from 0, ... n_nodes-1
    Q = dict.fromkeys(keys, 0)
    for k in A.keys():
        Q[k] = np.zeros(A[k].shape)
    Model = Q.copy()
    
    epsilon = epsilon_i
    for episode in range(n_episodes):                                   # loop forever
        # visited = {key = state: value = set of taken actions}
        visited = {}

        if episode == 1:
            state = np.random.choice(len(keys))                         # (a) 
        action = epsilon_greedy(state, Q, epsilon)                      # (b)

        if state in visited.keys():
            visited[state].add(action)
        else:
            visited[state] = set([action])

        """
        Get_next_state(action, state) gives 3 items
         - first item = numpy array of queue lengths
         - second item = new queue id (new edge index)
         - third item = 4 tuple = (source vertex, target vertex, edge index, edge type)
        and we are interested in the target vertex.
        """
        _, _, third = env.get_next_state(action, state)
        # next_state = third[1]                                           # (c)
        reward = env.get_reward()

        if state < keys[-1]:
            next_state = state + 1
        else:
            next_state = 0

        # (d)
        Q[state][action] += alpha*(reward + \
                                gamma*np.max(Q[next_state]) - \
                                    Q[state][action])
        
        Model[state][action] = (reward, next_state)                     # (e)

        for _ in range(n):
            s = np.random.choice(list(visited.keys()))
            a = np.random.choice(list(visited[s]))
            reward, next_state = Model[s][a]
            Q[s][a] += alpha*(reward + \
                                gamma*np.max(Q[next_state]) - \
                                Q[s][a])
            state = s
        
        epsilon *= 0.999
    
    return Q


def test_training():
    env = get_env()                 # instance of RLEnv
    n_episodes, n = 10,10
    alpha,gamma = 0.05, 0.9
    epsilon_i = 0.9
    adja_list = env.adja_list
    A = generate_action_space(adja_list=adja_list, interval=0.5)
    Q = dyna_q(n_episodes, alpha, gamma, epsilon_i, n, A, env)
    print(Q)


if __name__ == "__main__":
    test_training()
    
