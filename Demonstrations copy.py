import torch
import matplotlib.pyplot as plt
from RLEnvV2 import *
from qt_MollyPart_runnable import *
import numpy as np
import copy


# This file contains all the demonstration classes for agent learning
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class config:
    # Creates the configuration object for the demonstrations
    def __init__(self, environment, agent, metric="throughput"):
        self.environment = environment
        self.agent = agent
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = 0
        self.queue_metrics = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(
            xlabel="Time Steps",
            ylabel=self.metric,
            title=self.metric + f" vs Time Steps for Queue {str(self.queue_index)}",
        )

    def retrieve_components(self):
        # Getter function allowing other classes to access the environment and agent objects
        return self.environment, self.agent


class Network_Control:
    """
    This class contains all the methods needed for the agent to control the network
    Inputs:
    - environment - testing RL Environment
    - agent - trained agent with the learnt policy
    """

    def __init__(self, environment, agent, metric="throughput"):
        """
        Initiates the class with the environment and the agent
        """
        self.environment = environment
        self.agent = agent
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = 0
        self.queue_metrics = []
        self.call_plot_num = 0
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set(
            xlabel="Time Steps",
            ylabel=self.metric,
            title=self.metric + f" vs Time Steps for Queue {str(self.queue_index)}",
        )

    def configure(self, time_steps, metric, queue_index=None):
        """_summary_

        Args:
            time_steps (_type_): _description_
            metric (_type_): _description_
            queue_index (_type_, optional): _description_. Defaults to None.
        Returns:
        """
        self.queue_index = queue_index
        self.time_steps = time_steps
        self.metric = metric

    def plot_queue_realtime(self):
        """
        This function continually plots the queue length at each time step in real time
        """
        self.ax.clear()
        self.ax.plot(range(len(self.queue_metrics)), self.queue_metrics)
        self.ax.set(
            xlabel="Time Steps",
            ylabel=self.metric,
            title=self.metric + f" vs Time Steps for Queue {str(self.queue_index)}",
        )
        plt.draw()
        plt.pause(0.01)
        plt.show()

    def plot_queue(self, labels, *queue_metrics_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels."""
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(
            xlabel="Time Steps",
            ylabel=self.metric,
            title=self.metric + f" vs Time Steps for Queue {str(self.queue_index)}",
        )
        self.ax.legend()  # Add a legend to differentiate the lines
        figure_name = f"{self.call_plot_num}_plot_queue.png"
        # plt.show()
        plt.savefig(figure_name)
        self.call_plot_num += 1

    def control(
        self,
        environment=None,
        agent=None,
        time_steps=None,
        queue_index=None,
        metric=None,
    ):
        """
        This function is the main control loop for the agent and network interaction.
        """
        # Use instance attributes if no arguments are provided
        environment = environment or self.environment
        agent = agent or self.agent
        time_steps = time_steps if time_steps is not None else self.time_steps
        queue_index = queue_index if queue_index is not None else self.queue_index
        metric = metric or self.metric
        queue_metrics = []

        for time_step in range(time_steps):
            state = environment.get_state()
            action = agent.actor(state).detach()
            state = environment.get_next_state(action)[0]
            queue_metrics.append(environment.return_queue(queue_index, metric=metric))
            if (
                time_step % 10 == 0
            ):  # Corrected: to ensure it executes when time_step is a multiple of 10
                # self.plot_queue() - this would be the real time plotting logic
                pass

        self.plot_queue(metric, queue_metrics)  # Plot once after completing the loop
        plt.figure()
        plt.plot(queue_metrics, label=metric)
        plt.show()
        return queue_metrics


class Static_Disruption(Network_Control):
    """
    This class extends Network_Control for demonstrations where the agent has to handle disruptions
    """

    def __init__(self, environment, agent, source_node, target_node):
        super().__init__(environment, agent)
        # Initialize any additional variables or settings specific to disruptions
        self.standard_environment = environment
        self.disrupted_environment = self.deactivate_node(source_node, target_node)

    def deactivate_node(self, source_node, target_node):
        environment2 = get_env(n=5000)
        q_classes = environment2.qn_net.q_classes
        q_args = environment2.qn_net.q_args
        edge_list = environment2.qn_net.edge_list
        new_class = len(q_classes)
        q_classes[new_class] = qt.LossQueue
        q_args[new_class] = {"service_f": lambda t: t + np.inf}
        edge_list[source_node][target_node] = new_class
        # environment2 = copy.copy(self.environment)
        environment2.qn_net.edge_list = edge_list
        return environment2

    def multi_control(self):
        """
        This function shows the agent interating with the orignal environment and the disrupted environment in parallel
        """
        normal_metrics = self.control(
            environment=self.standard_environment,
            agent=self.agent,
            time_steps=self.time_steps,
            queue_index=self.queue_index,
            metric=self.metric,
        )
        disrupted_metrics = self.control(
            environment=self.disrupted_environment,
            agent=self.agent,
            time_steps=self.time_steps,
            queue_index=self.queue_index,
            metric=self.metric,
        )
        self.plot_queue(
            normal_metrics, disrupted_metrics, labels=["Normal", "Disrupted"]
        )


if __name__ == "__main__":
    agent = torch.load("trained_agent.pt")
    env = get_env(n=5000)
    nc = Network_Control(agent, env)
    nc.plot_queue_realtime()
    queue_metrics = nc.control(
        environment=env, agent=agent, time_steps=10, queue_index=2, metric="throughput"
    )
    print(queue_metrics)
    breakpoint()

    ## Static Disruption
    sd = Static_Disruption(env, agent, 1, 3)
    disrupted_env = sd.disrupted_environment
    sd.multi_control()
    breakpoint()


# Assuming that we have written a python file called t.py that contains the function square
# And we are writing tests in the file t_test.py
import pytest
import t


# FUNCTION BASED TESTS 
def test_square_positive_int(self): # all test functions should start with test_
    assert t.square(4) == 16
    
def test_square_negative_int(self):
    assert t.square(-4) == 16

def test_square_zero(self):
    assert t.square(0) == 0

# CLASS BASED TESTS
class TestSquare: # All test classes should start with Test
    def test_square_positive_float(self): # all methods within the class should start with test_
        assert t.square(3.5) == 12.25
        
    # Test that attempting to square a string raises a TypeError
    def test_square_string(self):
        with pytest.raises(TypeError):
            t.square("a string")

    # Test that attempting to square None raises a TypeError
    def test_square_none(self):
        with pytest.raises(TypeError):
            t.square(None)

    # Parametrized test that verifies squaring for multiple inputs and outputs
    # This is a more compact way of testing multiple scenarios
    @pytest.mark.parametrize("input,expected", [
        (2, 4),
        (-2, 4),
        (0, 0),
        (1e5, 1e10),
    ])
    def test_square_parametrized(self, input, expected):
        assert t.square(input) == expected

# Main entry point to run tests when the script is executed directly
# Incoporate the argparser and the logic for choosing the tests to run based on the arguments passed 
if __name__ == "__main__":
    pytest.main()


# Define the command to run pytest
PYTEST = pytest

# Define the directory containing the tests
TEST_DIR = tests

# Define the default target that make will aim to build
test:
	pytest tests

.PHONY: test

cd path/to/my_project
make test


poetry build
poetry config pypi-token.pypi YOUR_PYPI_TOKEN
poetry publish