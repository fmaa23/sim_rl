# from tensorflow import random
import numpy as np
from .ddpg import DDPGAgent
import torch
from .queueing_network import Queue_network

def explore_state(DDPG_agent, queue_model, num_sample, device, visit_counts, w1 = 0.5, w2 = 0.5, epsilon = 1):
    """
    This function is used for state exploration for real word environment
    It helps the agent decide which state to explore next based on factors, including Q-values and visit counts. 
    
    Parameters:
    - DDPG_agent: the class of the DDPGAGent
    - queue_model: the class of the queue model
    - num_sample: number of states we sample from to get the lowest reward and times visited
    - visit_counts: a dictionary that stores the number of times being visited for each state;
    The key is the tuple of the state and the value is the number of time visited
    - w1: Weight for the influence of Q-values (default: 0.5)
    - w2: Weight for the influence of visit counts (default: 0.5)
    - epsilon: Probability of exploration (default: 1)

    Returns:
    - chosen_state: The selected state for exploration.
    """
    max_buffer_size = [20] * 12
    
    if False:
        for q_buffer in queue_model.q_args.keys():
            if q_buffer != 1 and q_buffer != 0:
                max_buffer = queue_model.q_args[q_buffer]['qbuffer']
                max_buffer_size.append(max_buffer)
    
    sample_states = []
    for _ in range(num_sample):
        array = np.array([np.random.randint(0, max_val) for max_val in max_buffer_size])
        sample_states.append(array)

    states_array = np.array(sample_states)

    # explore state with lower Q values and being visited less frequently
    states_ordered_by_reward = rank_states_by_Q_values(DDPG_agent, states_array, device)
    states_ordered_by_visits = rank_states_by_visits(visit_counts, states_array, device)

    reward_rankings = {state: rank for rank, state in enumerate(states_ordered_by_reward)}
    visit_rankings = {state: rank for rank, state in enumerate(states_ordered_by_visits)}

    # calculate weighted average of rankings for each state
    weighted_averages = {}
    for state in reward_rankings.keys():
        weighted_avg = w1 * reward_rankings[state]+ \
                        w2 * visit_rankings[state]
        weighted_averages[state] = weighted_avg

    # sort states by weighted average
    sorted_states = sorted(weighted_averages.items(), key=lambda x: x[1])

    # epsilon-greedy selection
    if np.random.rand() < epsilon:
        # exploit: choose the state with the lowest weighted average
        chosen_state = sorted_states[0][0]
    else:
        # explore: randomly choose any state
        chosen_state = np.random.choice(list(weighted_averages.keys()))

    return torch.tensor(chosen_state).to(device)

def rank_states_by_Q_values(DDPG_agent, states_array, device):
    """
    This function ranks states based on their Q-values. 
    The function first generates all possible states based on the maximum buffer sizes.
    Then calculates normalized Q-values for each state using the critic network.

    Parameters:
    - queue_model: The queue network class
    - critic_network: The critic neural network used to estimate Q-values.

    Returns:
    - sorted_state_q_value_pairs: A list of state-Q-value pairs sorted in ascending order of Q-values.
    """
    action_vector =  DDPG_agent.actor(torch.tensor(states_array).to(device))
    q_values = DDPG_agent.critic.forward((torch.tensor(states_array).float(), action_vector))

    state_q_value_pairs = list(zip(torch.tensor(states_array).float(), q_values))
    sorted_state_q_value_pairs = sorted(state_q_value_pairs, key=lambda x: x[1])

    sorted_dict = {}
    for info in sorted_state_q_value_pairs:
        state = info[0].numpy().tolist()
        q_value = info[1].item()

        sorted_dict[tuple(state)] = q_value

    return sorted_dict

def rank_states_by_visits(visit_counts, states_array, device):
    """
    This function ranks states based on the number of visits they have received. 
    The function takes a matrix of visit counts and sorts the states in descending order of visit counts.

    Parameters:
    - visit_counts: A matrix containing visit counts for each state.

    Returns:
    - ordered_states_visit_dict: A dict of visited states and their corresponding visit counts, sorted in descending order of visit counts.
    """
    states_visit_dict = {}
    for state in states_array.tolist():
        if state in list(visit_counts.keys()):
            num_visited = visit_counts[state]
        else:
            num_visited = 0
        states_visit_dict[tuple(state)] = num_visited
    ordered_states_visit_dict = dict(sorted(states_visit_dict.items(), key=lambda item: item[1], reverse=True))
    return ordered_states_visit_dict

if __name__ == "__main__":
    # example input
    lamda_list = [0.2, 0.2, 0.2, 0.2]
    miu_list = [0.1, 0.1, 0.1, 0.2]
    active_cap = 5
    deactive_t = 0.12
    adjacent_list = {0:[1,2], 1:[3], 2:[3]}
    buffer_size_for_each_queue = [10, 10, 10, 10]
    transition_proba= {0:{1:0.5, 2:0.5}, 1:{3:1}, 2:{3:1}}

    params = {'tau':0.4,'lr':0.1,'discount':0.3,'epsilon':0.2,'planning_steps':100}

    q_net = Queue_network()
    q_net.process_input(lamda_list, miu_list, active_cap, deactive_t, adjacent_list, buffer_size_for_each_queue, transition_proba)
    q_net.create_env()
    DDPG_agent = DDPGAgent(4, 4, params)

    current_state = explore_state(DDPG_agent, queue_model = q_net,
                                num_sample = 50, visit_counts = DDPG_agent.visited_count,
                                w1 = 0.5, w2 = 0.5, epsilon = 1)
    
    print(f"explore state: {current_state}")
