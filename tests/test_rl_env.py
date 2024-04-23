import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

# Now you can do an absolute import
from rl_env.RL_Environment import RLEnv
from foundations.supporting_functions import * 
import pytest
import numpy as np 

# Dynamically construct the path to the configuration directory
config_path = Path(__file__).resolve().parent.parent

@pytest.fixture(scope='module')
def setup_rlenv(request):
    """
    Set up the test environment by initializing an environment with a predefined initialization.
    This setup runs before each test within the module that requests it.
    """
    configuration_file = config_path / request.param
    params = {'num_sim': 5000}
    env = create_simulation_env(params, config_file=configuration_file)
    yield env

class TestRLEnv:
    """
    Test for the RLEnv class to ensure correctness of environment initialization, 
    simulation of movement of the environment, calculation of reward. 
    """
    def test_environment_initialization(self, setup_rlenv):
        """
        Test if the environment is initialized correctly
        """
        assert setup_rlenv is not None, "Failed to initialize the RL environment."
    
    @pytest.mark.parametrize('setup_rlenv, expected', 
                             [('user_config/configuration.yml', 1), 
                              ('user_config/configuration2.yml', 2), 
                              ('user_config/configuration3.yml', 0)], 
                              indirect=['setup_rlenv'])
    def test_get_entrynodes(self, setup_rlenv, expected):
        """
        Test if the environment is initialized with the correct number of entry nodes
        for different configurations.
        """
        actual_entrynodes = setup_rlenv.get_entrynodes()
        assert actual_entrynodes == expected, f"Expected {expected} entry nodes, but got {actual_entrynodes}."
    
    @pytest.mark.parametrize('setup_rlenv, expected', 
                             [('user_config/configuration.yml', 1), 
                              ('user_config/configuration2.yml', 0), 
                              ('user_config/configuration3.yml', 2)], 
                              indirect=['setup_rlenv'])
    def test_get_nullnodes(self, setup_rlenv, expected):
        """
        Test if the environment is initialized with the correct number of entry nodes
        for different configurations.
        """
        actual_nullnodes = setup_rlenv.get_nullnodes()
        assert actual_nullnodes == expected, f"Expected {expected} exit nodes, but got {actual_nullnodes}."
    
    @pytest.mark.parametrize('setup_rlenv, expected', 
                             [('user_config/configuration.yml', 12), 
                              ('user_config/configuration2.yml', 4), 
                              ('user_config/configuration3.yml', 6)], 
                              indirect=['setup_rlenv'])
    def test_get_state(self, setup_rlenv, expected):
        """
        Tests that the states length is equal to all edges except the null nodes
        """
        actual_states_length = len(setup_rlenv.get_state())
        assert (actual_states_length == expected) and (not np.isnan(actual_states_length))
    
    
    @pytest.mark.parametrize('setup_rlenv, actions', 
                             [('user_config/configuration.yml',[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]), 
                              ('user_config/configuration2.yml',[0.5,0.5,0.5,0.5]), 
                              ('user_config/configuration3.yml',[[0.5,0.5,0.5,0.5,0.5,0.5]])], 
                              indirect=['setup_rlenv'])
    def test_get_next_state(self, setup_rlenv, actions):
        """
        Tests that masking of the action is corretly performed
        """
        
        setup_rlenv.get_next_state(actions)
        for node in setup_rlenv.transition_proba.keys(): 
            actions_node_proba = []
            for next_node in setup_rlenv.transition_proba[node]: 
                actions_node_proba.append(setup_rlenv.transition_proba[node][next_node])
            assert  np.sum(np.array(actions_node_proba))==1
    @pytest.mark.parametrize('setup_rlenv, expected_reward', 
                             [('user_config/configuration.yml',-150), 
                              ('user_config/configuration2.yml',-70), 
                              ('user_config/configuration3.yml',-30)], 
                              indirect=['setup_rlenv'])
    def test_get_reward(self, setup_rlenv, expected_reward): 
        actual_reward = setup_rlenv.get_reward()
        assert actual_reward == expected_reward, f"Expected {expected_reward} exit nodes, but got {actual_reward}."

if __name__ == '__main__':
    pytest.main()