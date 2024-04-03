import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

# Now you can do an absolute import
from agents.ddpg import DDPGAgent
import unittest
from unittest.mock import MagicMock, patch
import torch

class TestDDPG(unittest.TestCase):
    """
    Unit tests for the DDPGAgent class to ensure correctness of initialization,
    model fitting, action selection, and Q-value updates.

    Attributes:
        n_states (int): The number of states in the environment.
        n_actions (int): The number of possible actions the agent can take.
        hidden (dict): Configuration of hidden layers for different neural networks within the agent.
        params (dict): Hyperparameters for the DDPG agent, including learning rate, discount factor, etc.
        agent (DDPGAgent): An instance of the DDPGAgent class to be tested.
    """
    def setUp(self):
        """
        Set up the test environment by initializing a DDPGAgent with predefined states, actions, and parameters.
        This setup runs before each test method execution.
        """
        self.n_states = 10
        self.n_actions = 2
        self.hidden = {'actor': [64, 64], 'critic': [64, 64], 'reward_model': [10, 10], 'next_state_model': [10, 10]}
        self.params = {'tau': 0.1, 'learning_rate': 0.001, 'discount': 0.99, 'epsilon': 0.1, 'planning_steps': 5}
        self.agent = DDPGAgent(self.n_states, self.n_actions, self.hidden, self.params)
    
    @patch('torch.optim.Adam.step')
    def test_fit_model(self, mock_step):
        """
        Test the `fit_model` method of the DDPGAgent to ensure it processes the training data correctly,
        applying model updates as expected. This test uses a mock of the Adam optimizer's step method
        to prevent actual parameter updates during the test.
        """
        mock_step.return_value = None

        for _ in range(10):
            state = torch.randn(self.n_states)
            action = torch.randn(self.n_actions)
            reward = torch.randn(1)
            next_state = torch.randn(self.n_states)
            self.agent.buffer.push((state, action, reward, next_state))

        initial_loss_reward, initial_loss_next_state = self.agent.fit_model(batch_size=5, threshold=10, epochs=1)
        self.assertTrue(len(initial_loss_reward) > 0, "No reward model training loss was recorded.")
        self.assertTrue(len(initial_loss_next_state) > 0, "No next state model training loss was recorded.")

    def test_select_action(self):
        """
        Test the `select_action` method of the DDPGAgent to ensure it returns an action of the correct shape
        given a state input. This test verifies the action selection process is functioning as expected.
        """
        state = torch.randn(self.n_states)
        action = self.agent.select_action(state)
        self.assertEqual(action.shape, torch.Size([self.n_actions]), "Action shape mismatch.")

    def test_update_q_values(self):
        """
        Test the Q-value updating process within the critic network of the DDPGAgent. This method tests
        the critic's ability to update its Q-values by observing changes in loss before and after an update cycle,
        ensuring the network learns from the provided batch of experiences.
        """
        batch = [(torch.randn(self.n_states), torch.randn(self.n_actions), torch.randn(1), torch.randn(self.n_states)) for _ in range(5)]
        initial_loss = self.agent.update_critic_network(batch)
        
        updated_loss = self.agent.update_critic_network(batch)
        self.assertTrue(updated_loss < initial_loss, "Critic network Q-value update did not reduce loss.")

if __name__ == '__main__':
    unittest.main()