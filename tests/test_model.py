import torch
from model import Actor, Critic, RewardModel, NextStateModel 

def test_actor_architecture():
    n_states, n_actions, hidden = 4, 2, [64, 64]
    model = Actor(n_states, n_actions, hidden)
    expected_num_layers = 2 * len(hidden) + 2  # times 2 for Linear+ReLU, +2 for input and output layers

    assert len(model.layers) == expected_num_layers, f"Actor model should have {expected_num_layers} layers but has {len(model.layers)}."

def test_critic_architecture():
    n_states, n_actions, hidden = 4, 2, [64, 64]
    model = Critic(n_states, n_actions, hidden)
    # Critic has 1 additional layer for action input concatenation
    expected_num_layers = 2 * (len(hidden) - 1) + 3  # +3 for the first two layers and the last layer, layers are reduced by 1 due to custom structure

    # Checking only the last sequential block for simplicity
    assert len(model.layer3) == expected_num_layers - 3, f"Critic model should have {expected_num_layers} layers in total but has {len(model.layer3) + 3} when considering the structured layers."

### 2. Testing Forward Pass

""" Verifying that the networks can process input data and produce outputs without throwing any errors.
We will use random input data of the correct shape and check if the output is of the correct shape."""
    
def test_actor_forward_pass():
    n_states, n_actions, hidden = 4, 2, [64, 64]
    model = Actor(n_states, n_actions, hidden)
    input_state = torch.randn(1, n_states)

    output = model(input_state)
    assert output.shape == (1, n_actions), f"Actor output shape should be {(1, n_actions)} but is {output.shape}."

def test_critic_forward_pass():
    n_states, n_actions, hidden = 4, 2, [64, 64]
    model = Critic(n_states, n_actions, hidden)
    input_state = torch.randn(1, n_states)
    input_action = torch.randn(1, n_actions)

    output = model([input_state, input_action])
    assert output.shape == (1, 1), f"Critic output shape should be {(1, 1)} but is {output.shape}."

if __name__=="__main__":
    test_actor_architecture()
    test_critic_architecture()
    test_actor_forward_pass()
    test_critic_forward_pass()
    print("All tests passed")