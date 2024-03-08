import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_gradient(layer_name = 'layers.0.weight'):

    filename = 'gradient_dict.json'

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

    plt.savefig(f'Gradient_{layer_name}.png')

def plot_actor():
    actor_loss = pd.read_csv('actor_loss_new.csv', index_col=0)
    plt.plot(actor_loss)
    plt.title("Actor Loss")

    plt.savefig('Actor_Loss.png')

def plot_critic():
    critic_loss = pd.read_csv('critic_loss_new.csv', index_col=0)
    plt.plot(critic_loss)
    plt.title('Critic Loss')

    plt.savefig('Critic_Loss.png')

def plot_reward():
    reward = pd.read_csv('reward.csv', index_col=0)
    plt.plot(reward.rolling(window=5).mean())
    plt.title("Reward")

    plt.savefig('Reward.png')

def plot_actor_vector():
    action_vector = pd.read_csv("action_dict.csv", index_col=0)
    for column in action_vector.columns:
        plt.plot(action_vector.index, action_vector[column], label=column)

    plt.title("Action Vector")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Coordinates for the legend box
    plt.show()

    plt.savefig('Actor_Vector.png')

def plot_transition_proba():
    transition_proba = pd.read_csv("transition_proba.csv", index_col=0)[["2", "3", "4"]]
    for column in transition_proba.columns:
        plt.plot(transition_proba.index, transition_proba[column], label=column)

    plt.title("Transition Proba")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Coordinates for the legend box
    plt.show()

    plt.savefig('Transition_Proba.png')

def plot_reward_model_loss():
    reward_model_loss_new = pd.read_csv('reward_model_loss_new.csv', index_col=0)
    plt.plot(reward_model_loss_new.rolling(window=12).mean())
    plt.title("reward_model_loss_new")

    plt.savefig('Reward_Model_Loss.png')

def plot_next_state_model_loss():
    next_state_model = pd.read_csv('next_state_model_loss.csv', index_col=0)
    plt.plot(next_state_model)
    plt.title("next_state_model_loss")

    plt.savefig('Next_State_Model_Loss.png')

def plot():
    plot_reward()
    plot_actor()
    plot_critic()
    plot_actor_vector()
    plot_gradient()
    plot_transition_proba()
    plot_reward_model_loss()
    plot_next_state_model_loss()