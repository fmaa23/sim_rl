from tensorflow import random
from Aaron_Part import DDPGAgent
import numpy as np
import itertools

def explore_state(w1 = 0.5, w2 = 0.5, epsilon = 1):
    """
    This function is used for state exploration for real word environment
    It helps the agent decide which state to explore next based on factors, including Q-values and visit counts. 
    
    Parameters:
    - w1: Weight for the influence of Q-values (default: 0.5)
    - w2: Weight for the influence of visit counts (default: 0.5)
    - epsilon: Probability of exploration (default: 1)

    Returns:
    - chosen_state: The selected state for exploration.
    """

    # explore state with lower Q values and being visited less frequently
    states_ordered_by_reward = rank_states_by_Q_values()
    states_ordered_by_visits = rank_states_by_visits()

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

def rank_states_by_Q_values(self):
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

    max_buffer_size = []
    for q_buffer in self.queue_model.q_args.keys():
        max_buffer = self.queue_model.q_args[q_buffer]
        max_buffer_size.append(max_buffer)
    
    queue_ranges = [range(max_buffer_size[i]) for i in range(len(max_buffer))]
    all_states = np.array(list(itertools.product(*queue_ranges)))
    
    all_q_values = []
    batch_size = 64
    for i in range(0, len(all_states), batch_size):
        batch_states = all_states[i:i+batch_size]
        batch_q_values = self.critic_network.forward(batch_states)
        all_q_values.append(batch_q_values)
    
    max_q_value = np.max(all_q_values)  
    normalized_q_values = all_q_values / max_q_value  

    state_q_value_pairs = list(zip(all_states, normalized_q_values))
    sorted_state_q_value_pairs = sorted(state_q_value_pairs, key=lambda x: x[1])

    return sorted_state_q_value_pairs

def track_state(visit_counts, state_indices):
    """
    This function tracks the number of times each state is being visited.

    Parameters:
    - visit_counts: a numpy array that tracks the number of times visited for each state 
                    and has shape np.zeros(self.max_size_per_dimension, dtype=int)
    - state_indicies: a tuple that represents the queue length for all nodes 

    Returns: increment the number of times being visited for the provided state
    """
    
    # Increment the visit count for a given state
    visit_counts[state_indices] += 1

def rank_states_by_visits(self):
    """
    This function ranks states based on the number of visits they have received. 
    The function takes a matrix of visit counts and sorts the states in descending order of visit counts.

    Parameters:
    - visit_counts: A matrix containing visit counts for each state.

    Returns:
    - visited_states_and_counts: A list of visited states and their corresponding visit counts, sorted in descending order of visit counts.
    """

    flattened_visit_counts = self.visit_counts.flatten()
    sorted_indices = np.argsort(flattened_visit_counts)[::-1]
    sorted_visit_counts = flattened_visit_counts[sorted_indices]
    multi_dim_indices = np.unravel_index(sorted_indices, self.visit_counts.shape)
    ranked_states_and_counts = list(zip(zip(*multi_dim_indices), sorted_visit_counts))
    visited_states_and_counts = [item for item in ranked_states_and_counts if item[1] > 0]
    
    return visited_states_and_counts

    
if __name__ == "__main__":
    DDPG_agent = DDPGAgent()
    num_stored_experiences = 0
    batch_size = 100
    
    while True:
        current_state = DDPG_agent.explore_state()
        action = DDPG_agent.get_action(current_state)
        #TODO: need to modify Queue_model.get_next_state() such that it returns reward, next_state, done
        reward, next_state, done = DDPG_agent.model.get_next_state(action)
        DDPG_agent.store_to_buffer(current_state, action, reward, next_state)
        num_stored_experiences += 1

        if num_stored_experiences > batch_size:
            DDPG_agent.train()