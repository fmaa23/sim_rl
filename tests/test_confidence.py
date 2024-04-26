import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
from unittest.mock import patch, MagicMock

from evaluation.convergence_evaluation.convergence_evaluation import Confidence


@pytest.fixture
def confidence_instance():
    # Set up a Confidence instance with fixed episodes and timesteps for testing
    num_episodes = [10, 50, 100]
    timesteps = 100
    return Confidence(num_episodes, timesteps)


@patch("foundations.supporting_functions.create_simulation_env")
@patch("foundations.supporting_functions.create_ddpg_agent")
@patch("foundations.supporting_functions.load_hyperparams")
def test_start_train(
    mock_load_hyperparams,
    mock_create_ddpg_agent,
    mock_create_simulation_env,
    confidence_instance,
):
    # Mock the environment and agent to be returned by the mocked functions
    mock_env = MagicMock()
    mock_agent = MagicMock()
    mock_create_simulation_env.return_value = mock_env
    mock_create_ddpg_agent.return_value = mock_agent
    mock_load_hyperparams.return_value = (
        {},
        {},
    )  # Return empty dicts for params and hidden

    # Mock internal methods that we don't want to execute during this test
    with patch.object(
        confidence_instance, "train", return_value=(None, [], [], {}, {}, {}, None)
    ) as mock_train, patch.object(
        confidence_instance, "evaluate_agent", return_value=100
    ) as mock_evaluate, patch(
        "your_script.save_all"
    ), patch(
        "os.makedirs"
    ), patch(
        "os.getcwd", return_value="/fakepath"
    ):
        confidence_instance.start_train(
            "fake_config.yml", "fake_eval_config.yml", "fake_param_file.yml"
        )

        # Verify that train and evaluate_agent are called the expected number of times
        assert mock_train.call_count == len(confidence_instance.num_episodes)
        assert mock_evaluate.call_count == len(confidence_instance.num_episodes)


def test_evaluate_agent(confidence_instance):
    # Mock the environment and agent needed for evaluation
    mock_env = MagicMock()
    mock_env.get_state.return_value = [0]  # Simplistic state
    mock_env.get_next_state.return_value = ([0],)  # Simplistic next state
    mock_env.get_reward.return_value = 1  # Constant reward
    mock_agent = MagicMock()
    mock_agent.actor.return_value = MagicMock(
        detach=MagicMock(return_value=[0])
    )  # Simplistic action

    total_reward = confidence_instance.evaluate_agent(mock_agent, mock_env, 100)
    assert total_reward == 100  # 100 timesteps * 1 reward per timestep


@patch("matplotlib.pyplot.savefig")
def test_save_reward_plot(mock_savefig, confidence_instance):
    # Assume the total_rewards have been populated after some hypothetical runs
    confidence_instance.total_rewards = [100, 200, 300]
    confidence_instance.num_episodes = [10, 50, 100]
    confidence_instance.save_reward_plot("/fakepath", "test_plot.png")
    mock_savefig.assert_called_once()  # Check if the plot is actually saved


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
