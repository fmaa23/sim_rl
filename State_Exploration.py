from tensorflow import random
import numpy as np
import itertools

def explore_state(DDPG_agent, queue_model, num_sample, visit_counts, w1 = 0.5, w2 = 0.5, epsilon = 1):
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

    max_buffer_size = []
    for q_buffer in queue_model.q_args.keys():
        max_buffer = queue_model.q_args[q_buffer]
        max_buffer_size.append(max_buffer)
    
    sample_states = []
    for _ in range(num_sample):
        array = np.array([np.random.randint(0, max_val) for max_val in max_buffer_size])
        sample_states.append(array)

    states_array = np.array(sample_states)

    # explore state with lower Q values and being visited less frequently
    states_ordered_by_reward = rank_states_by_Q_values(DDPG_agent)
    states_ordered_by_visits = rank_states_by_visits(visit_counts, states_array)

    reward_rankings = {state[0]: rank for rank, state in enumerate(states_ordered_by_reward)}
    visit_rankings = {state[0]: rank for rank, state in enumerate(states_ordered_by_visits)}

    # calculate weighted average of rankings for each state
    weighted_averages = {}
    for state in reward_rankings.keys():
        weighted_avg = w1 * (reward_rankings[state] / len(states_ordered_by_reward)) + \
                        w2 * (visit_rankings[state] / len(states_ordered_by_visits))
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

    return chosen_state

def rank_states_by_Q_values(DDPG_agent, states_array):
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
    
    q_values = DDPG_agent.critic_network.forward(states_array)

    state_q_value_pairs = list(zip(states_array, q_values))
    sorted_state_q_value_pairs = sorted(state_q_value_pairs, key=lambda x: x[1])

    return sorted_state_q_value_pairs


def rank_states_by_visits(visit_counts, states_array):
    """
    This function ranks states based on the number of visits they have received. 
    The function takes a matrix of visit counts and sorts the states in descending order of visit counts.

    Parameters:
    - visit_counts: A matrix containing visit counts for each state.

    Returns:
    - ordered_states_visit_dict: A dict of visited states and their corresponding visit counts, sorted in descending order of visit counts.
    """
    states_visit_dict = {}
    for state in states_array:
        if state in visit_counts:
            num_visited = visit_counts[state]
        else:
            num_visited = 0
    states_visit_dict[state] = num_visited
    ordered_states_visit_dict = dict(sorted(states_visit_dict.items(), key=lambda item: item[1], reverse=True))
    return ordered_states_visit_dict

    
if __name__ == "__main__":

    explore_state(DDPG_agent, queue_model, num_sample, visit_counts, w1 = 0.5, w2 = 0.5, epsilon = 1)