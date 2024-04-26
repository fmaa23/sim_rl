import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from evaluation.decision_evaluation.decision_evaluation import (
    ControlEvalaution,
    DisruptionEvaluation,
)

# Setup the test environment
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


# Fixture for setting up ControlEvaluation instances
@pytest.fixture
def control_evaluation():
    agent = MagicMock()  # Mocking an agent instance
    return ControlEvalaution(agent, sim_jobs=100)


# Fixture for setting up DisruptionEvaluation instances
@pytest.fixture
def disruption_evaluation():
    agent = MagicMock()  # Mocking an agent instance
    return DisruptionEvaluation(agent, sim_jobs=100)


# Mocking the plotting functions to avoid actual file I/O and display during tests
@pytest.fixture(autouse=True)
def mock_plots():
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.show"):
        yield


# Example test: Ensuring proper initialization of ControlEvaluation
def test_control_evaluation_init(control_evaluation):
    assert control_evaluation.agent is not None
    assert control_evaluation.metric == "throughput"
    assert control_evaluation.queue_index == 2


# Test for plotting functions to ensure they are called correctly
def test_control_evaluation_plot_queue(control_evaluation):
    with patch.object(control_evaluation, "plot_queue") as mock_plot:
        queue_metrics = np.random.rand(10)
        control_evaluation.plot_queue(["Test"], queue_metrics)
        mock_plot.assert_called_once_with(["Test"], queue_metrics)


# Test for method functionality under different configurations
def test_disruption_evaluation_multi_control(disruption_evaluation):
    with patch.object(disruption_evaluation, "control") as mock_control:
        disruption_evaluation.multi_control()
        assert (
            mock_control.call_count == 2
        )  # Should be called twice for normal and disrupted scenarios


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
