import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import torch
from Supporting_files.supporting_functions import *
import pytest
from unittest.mock import MagicMock, patch
import yaml
import tempfile

def test_load_config(tmpdir):
    """
    Test the load_config function by creating a temporary YAML configuration file
    with known content and verifying that the function correctly loads this file into a dictionary.
    """
    sample_config = {
        'param1': 'value1',
        'param2': 'value2'
    }
    config_file = tmpdir.join("config.yml")
    with config_file.open("w") as f:
        yaml.dump(sample_config, f)
    
    config_path = str(config_file)
    loaded_config = load_config(config_path)
    assert loaded_config == sample_config, "The configuration was not loaded correctly."

def test_get_num_connections():
    """
    Test the get_num_connections function by providing a sample adjacency list and verifying
    that the function correctly calculates the total number of connections and identifies the exit nodes.
    """
    adjacent_list = {
        0: [1, 2],
        2: [3],
        1: [3],
        3: [4]
    }
    num_connections, exit_nodes = get_num_connections(adjacent_list)
    assert num_connections == 4, "Total number of connections is incorrect."
    assert exit_nodes == [4], "Exit nodes identified incorrectly."

def test_make_edge_list():
    """
    Test the make_edge_list function by providing a sample adjacency list and exit nodes,
    then verifying that the function correctly creates an edge list with appropriate weights.
    """
    adjacent_list = {
        1: [2],
        2: [3]
    }
    exit_nodes = [3]
    expected_edge_list = {
        1: {2: 1},
        2: {3: 0}
    }
    edge_list = make_edge_list(adjacent_list, exit_nodes)
    assert edge_list == expected_edge_list, "Edge list created incorrectly."

# Mock configuration for testing
TEST_CONFIG = {
    'arrival_rate': 375,
    'miu_list': {1:0.5, 2:0.5},
    'adjacent_list': {0: [1], 1: [2], 2:[3]},
    'buffer_size_for_each_queue': {1:10, 2:10},
    'transition_proba_all':{0:{1:1}, 1:{2:1}, 2:{3:1}}
    # Add other necessary parameters for your environment setup
}

@pytest.fixture
def mock_config_file():
    """
    Pytest fixture that creates a temporary configuration file with a predefined set of parameters
    for use in tests that require loading a configuration file.
    """
    with tempfile.NamedTemporaryFile('w', delete=False) as tmpfile:
        yaml.dump(TEST_CONFIG, tmpfile)
        return tmpfile.name

def test_graph_construction(mock_config_file): 
    """
    Test the graph construction within the queueing environment by verifying that the
    vertices and edges in the created graph match those expected from a mock configuration.
    """
    q_net = create_queueing_env(mock_config_file)
    
    # Derive expected vertices and edges from TEST_CONFIG
    expected_vertices = set(TEST_CONFIG['adjacent_list'].keys()) | set([node for sublist in TEST_CONFIG['adjacent_list'].values() for node in sublist])
    expected_edges = [(start, end) for start, ends in TEST_CONFIG['adjacent_list'].items() for end in ends]

    # Verify vertices exist in the graph
    actual_vertices = set(q_net.g.nodes())  # Adjust based on your graph's attribute
    assert expected_vertices == actual_vertices, f"Graph vertices do not match expected. Expected: {expected_vertices}, Actual: {actual_vertices}"
    
    # Verify edges exist in the graph
    actual_edges = set(q_net.g.edges())  # Adjust based on your graph's attribute
    assert set(expected_edges) == actual_edges, f"Graph edges do not match expected. Expected: {expected_edges}, Actual: {actual_edges}"

def test_environment_attributes(mock_config_file):
    """
    Test the initialization of environment attributes by verifying that attributes of the
    queueing environment match the expected values from the mock configuration file.
    """
    # Create the environment using the mock configuration file
    q_net = create_queueing_env(mock_config_file)
    
    # Assert that environment attributes match expected values from the configuration
    assert q_net.lamda == TEST_CONFIG['arrival_rate'], "arrival_rate does not match configuration"
    assert q_net.miu == TEST_CONFIG['miu_list'], "miu_list does not match configuration"
    # Add other assertions as needed for your environment
    
    # Clean up the temporary file
    os.remove(mock_config_file)