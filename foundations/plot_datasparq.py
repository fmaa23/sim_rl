import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np


def plot_gradient(data_filepath, images_filepath, layer_name = 'layers.0.weight'):

    plt.figure()
    filename = data_filepath + '/gradient_dict.json'

    with open(filename, 'r') as f:
        gradient_dict_loaded = json.load(f)

    layer_0_weight = gradient_dict_loaded[layer_name]
    num_params = len(gradient_dict_loaded[layer_name])

    param_evolution_all = []
    for i in range(num_params):
        param_evolution = [num_list[i] for num_list in layer_0_weight]
        param_evolution_all.append(param_evolution)

    for i, line in enumerate(param_evolution_all):
        plt.plot(line, label=f'Element {i + 1}')

    plt.title(f'Gradient of {layer_name}')
    plt.xlabel('List Index')
    plt.ylabel('Value')

    save_path = os.path.join(images_filepath, 'Gradient.png')
    plt.savefig(save_path)
    plt.close()

def plot_actor(data_filepath, images_filepath):
    actor_loss = pd.read_csv(data_filepath + '/actor_loss.csv', index_col=0)

    plt.figure()
    plt.plot(actor_loss)
    plt.title("Actor Loss")

    save_path = os.path.join(images_filepath, 'Acotor_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot_critic(data_filepath, images_filepath):
    critic_loss = pd.read_csv(data_filepath + '/critic_loss.csv', index_col=0)

    plt.figure()
    plt.plot(critic_loss)
    plt.title('Critic Loss')

    save_path = os.path.join(images_filepath, 'Critic_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot_reward(data_filepath, images_filepath):
    plt.figure()

    filename = os.path.join(data_filepath, 'reward_dict.json')

    with open(filename, 'r') as f:
        reward_data = json.load(f)
    
    #alm = np.array([value[0] for value in reward_data.values()])
    plt.plot(reward_data[list(reward_data.keys())[-1]])
    plt.title('Last Episode Reward per Time Step')
    save_path = os.path.join(images_filepath, 'Reward.png')
    plt.savefig(save_path)
    plt.close()

def plot_reward(data_filepath, images_filepath):
    plt.figure()

    filename = os.path.join(data_filepath, 'reward_dict.json')

    with open(filename, 'r') as f:
        reward_data = json.load(f)
    
    #alm = np.array([value[0] for value in reward_data.values()])
    plt.plot(reward_data[list(reward_data.keys())[-1]])
    plt.title('Last Episode Reward per Time Step')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    save_path = os.path.join(images_filepath, 'Reward Per Time Step.png')
    plt.savefig(save_path)
    plt.close()

def plot_first_reward(data_filepath, images_filepath):
    plt.figure()

    filename = os.path.join(data_filepath, 'reward_dict.json')

    with open(filename, 'r') as f:
        reward_data = json.load(f)
    
    first_reward_per_episode = np.array([value[0] for value in reward_data.values()])
    plt.plot(first_reward_per_episode)
    plt.title('First Reward Value Per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    save_path = os.path.join(images_filepath, 'First_Reward_Per_Episode.png')
    plt.savefig(save_path)
    plt.close()

def plot_last_reward(data_filepath, images_filepath):
    plt.figure()

    filename = os.path.join(data_filepath, 'reward_dict.json')

    with open(filename, 'r') as f:
        reward_data = json.load(f)
    
    last_reward_per_episode = np.array([value[-1] for value in reward_data.values()])
    plt.plot(last_reward_per_episode)
    plt.title('Last Reward Value Per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    save_path = os.path.join(images_filepath, 'Last_Reward_Per_Episode.png')
    plt.savefig(save_path)
    plt.close()

def plot_average_reward(data_filepath, images_filepath):
    plt.figure()

    filename = os.path.join(data_filepath, 'reward_dict.json')

    with open(filename, 'r') as f:
        reward_data = json.load(f)
    
    average_reward_per_episode = np.array([value for value in reward_data.values()])
    plt.plot(average_reward_per_episode.mean(axis=1))
    plt.title('Average Reward Per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    save_path = os.path.join(images_filepath, 'Average_Reward_Per_Episode.png')
    plt.savefig(save_path)
    plt.close()



def plot_actor_vector(data_filepath, images_filepath):
    plt.figure()

    action_vector = pd.read_csv(data_filepath + "/action_dict.csv", index_col=0)
    for column in action_vector.columns:
        plt.plot(action_vector.index, action_vector[column], label=column)

    plt.title("Action Vector")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Coordinates for the legend box
    plt.show()

    save_path = os.path.join(images_filepath, 'Actor_space.png')
    plt.savefig(save_path)
    plt.close()

def plot_transition_proba(data_filepath, images_filepath, transition_proba_dict, node = 1):

    df = pd.read_csv(data_filepath + '/transition_proba.csv', index_col = 0)

    num_col = [int(index) for index in list(df.columns)]
    df.columns = num_col

    nodes = list(transition_proba_dict[node].keys())

    data_to_plot = df[nodes]
    plt.figure(figsize=(10, 6))
    for key, values in data_to_plot.items():
        plt.plot(values, label=f'Transitions from 1 to {key}')

    plt.title('Transition Probabilities from 1')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.legend()

    save_path = os.path.join(images_filepath, 'transition probas.png')
    plt.savefig(save_path)
    plt.close()


def plot_reward_model_loss(data_filepath, images_filepath):
    reward_model_loss_new = pd.read_csv(data_filepath + '/reward_model_loss.csv', index_col=0)
    
    plt.figure()
    plt.plot(reward_model_loss_new.rolling(window=12).mean())
    plt.title("reward_model_loss_new")

    save_path = os.path.join(images_filepath, 'Reward_model_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot_next_state_model_loss(data_filepath, images_filepath):
    next_state_model = pd.read_csv(data_filepath + '/next_state_model_loss.csv', index_col=0)
    
    plt.figure()
    plt.plot(next_state_model)
    plt.title("next_state_model_loss")

    save_path = os.path.join(images_filepath, 'Next_model_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot(data_filepath, images_filepath, transition_probas = None):
    filepath = os.path.join(os.getcwd(), "foundations", "output_plots")

    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    plot_reward(data_filepath, images_filepath)
    plot_actor(data_filepath, images_filepath)
    plot_critic(data_filepath, images_filepath)
    plot_actor_vector(data_filepath, images_filepath)
    plot_gradient(data_filepath, images_filepath)
    plot_transition_proba(data_filepath, images_filepath, transition_probas, node = 1)
    plot_reward_model_loss(data_filepath, images_filepath)
    plot_next_state_model_loss(data_filepath, images_filepath)

    print(f"plots have been saved at {filepath}")

if __name__=="__main__": 
    breakpoint()