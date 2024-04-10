import sys
from pathlib import Path
import numpy as np
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import torch
from agents.ddpg import DDPGAgent
from Supporting_files.State_Exploration import *
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

@pytest.fixture
def explore_state_engine():
    return ExploreStateEngine()

def test_hyperparameter_loading(explore_state_engine):
    """Test if hyperparameters are loaded correctly."""
    params, hidden = explore_state_engine.load_hyperparams()
    assert isinstance(params, dict), "Expected params to be a dictionary"
    assert isinstance(hidden, dict), "Expected hidden to be a dictionary"
    # Add more assertions based on expected values

@patch.object(ExploreStateEngine, 'load_hyperparams')
def test_state_ranking_by_q_values(mock_load_hyperparams, explore_state_engine, ddpg_agent):
    """Test ranking of states based on Q-values."""
    # Mock load_hyperparams to return predefined values
    mock_load_hyperparams.return_value = ({'num_sample': 5, 'w1': 0.5, 'w2': 0.5}, {})
    # Define a simple scenario for testing
    test_states = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    # Call the method under test
    ranked_states = explore_state_engine.rank_states_by_Q_values(ddpg_agent, test_states, explore_state_engine.device)
    # Assert that states are ranked (implement according to your logic)
    assert isinstance(ranked_states, dict), "Expected a dictionary of ranked states"