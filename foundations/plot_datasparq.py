import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


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

    import os
    save_path = os.path.join(images_filepath, 'Gradient.png')
    plt.savefig(save_path)
    plt.close()

def plot_actor(data_filepath, images_filepath):
    actor_loss = pd.read_csv(data_filepath + '/actor_loss.csv', index_col=0)

    plt.figure()
    plt.plot(actor_loss)
    plt.title("Actor Loss")

    import os
    save_path = os.path.join(images_filepath, 'Acotor_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot_critic(data_filepath, images_filepath):
    critic_loss = pd.read_csv(data_filepath + '/critic_loss.csv', index_col=0)

    plt.figure()
    plt.plot(critic_loss)
    plt.title('Critic Loss')

    import os
    save_path = os.path.join(images_filepath, 'Critic_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot_reward(data_filepath, images_filepath):
    reward = pd.read_csv(data_filepath + '/reward.csv', index_col=0)
    
    plt.figure()
    plt.plot(reward)
    plt.title("Reward")
    
    import os
    save_path = os.path.join(images_filepath, 'Reward.png')
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

    import os
    save_path = os.path.join(images_filepath, 'Actor_space.png')
    plt.savefig(save_path)
    plt.close()

def plot_transition_proba(data_filepath, images_filepath):

    transition_proba = pd.read_csv(data_filepath + "/transition_proba.csv", index_col=0)
    for column in transition_proba.columns:
        plt.plot(transition_proba.index, transition_proba[column], label=column)

    plt.title("Transition Proba")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Coordinates for the legend box
    plt.show()

    import os
    save_path = os.path.join(images_filepath, 'Transition_proba.png')
    plt.savefig(save_path)
    plt.close()

def plot_reward_model_loss(data_filepath, images_filepath):
    reward_model_loss_new = pd.read_csv(data_filepath + '/reward_model_loss.csv', index_col=0)
    
    plt.figure()
    plt.plot(reward_model_loss_new.rolling(window=12).mean())
    plt.title("reward_model_loss_new")

    import os
    save_path = os.path.join(images_filepath, 'Reward_model_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot_next_state_model_loss(data_filepath, images_filepath):
    next_state_model = pd.read_csv(data_filepath + '/next_state_model_loss.csv', index_col=0)
    
    plt.figure()
    plt.plot(next_state_model)
    plt.title("next_state_model_loss")

    import os
    save_path = os.path.join(images_filepath, 'Next_model_loss.png')
    plt.savefig(save_path)
    plt.close()

def plot(data_filepath, images_filepath):
    import os
    filepath = os.getcwd() + '/Supporting_files' + '/images'

    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    plot_reward(data_filepath, images_filepath)
    plot_actor(data_filepath, images_filepath)
    plot_critic(data_filepath, images_filepath)
    plot_actor_vector(data_filepath, images_filepath)
    plot_gradient(data_filepath, images_filepath)
    plot_transition_proba(data_filepath, images_filepath)
    plot_reward_model_loss(data_filepath, images_filepath)
    plot_next_state_model_loss(data_filepath, images_filepath)