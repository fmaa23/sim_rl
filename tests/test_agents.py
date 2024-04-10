import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import torch
from agents.ddpg import DDPGAgent
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def ddpg_agent():
    """
    Pytest fixture to setup DDPGAgent for testing.
    """
    n_states = 10
    n_actions = 2
    hidden = {'actor': [64, 64], 'critic': [64, 64], 'reward_model': [10, 10], 'next_state_model': [10, 10]}
    params = {'tau': 0.1, 'learning_rate': 0.001, 'discount': 0.99, 'epsilon': 0.1, 'planning_steps': 5}
    agent = DDPGAgent(n_states, n_actions, hidden, params)
    return agent

def test_fit_model(ddpg_agent):
    """
    Test the `fit_model` method of the DDPGAgent to ensure it processes the training data correctly,
    applying model updates as expected. This test uses a mock of the Adam optimizer's step method
    to prevent actual parameter updates during the test.
    """
    with patch('torch.optim.Adam.step') as mock_step:
        mock_step.return_value = None

        for _ in range(10):
            state = torch.randn(10)
            action = torch.randn(2)
            reward = torch.randn(1)
            next_state = torch.randn(10)
            ddpg_agent.buffer.push((state, action, reward, next_state))

        initial_loss_reward, initial_loss_next_state = ddpg_agent.fit_model(batch_size=5, epochs=1)
        assert len(initial_loss_reward) > 0, "No reward model training loss was recorded."
        assert len(initial_loss_next_state) > 0, "No next state model training loss was recorded."

def test_select_action(ddpg_agent):
    """
    Test the `select_action` method of the DDPGAgent to ensure it returns an action of the correct shape
    given a state input. This test verifies the action selection process is functioning as expected.
    """
    state = torch.randn(10)
    action = ddpg_agent.select_action(state)
    assert action.shape == torch.Size([2]), "Action shape mismatch."

def test_update_q_values(ddpg_agent):
    """
    Test the Q-value updating process within the critic network of the DDPGAgent. This method tests
    the critic's ability to update its Q-values by observing changes in loss before and after an update cycle,
    ensuring the network learns from the provided batch of experiences.
    """
    batch = [(torch.randn(10), torch.randn(2), torch.randn(1), torch.randn(10)) for _ in range(5)]
    initial_loss = ddpg_agent.update_critic_network(batch)
    
    updated_loss = ddpg_agent.update_critic_network(batch)
    assert updated_loss < initial_loss, "Critic network Q-value update did not reduce loss."