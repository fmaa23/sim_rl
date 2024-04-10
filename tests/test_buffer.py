import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

from Supporting_files.buffer import ReplayBuffer
import pytest

def test_replay_buffer_capacity_handling():
    """
    Test that the ReplayBuffer correctly handles its capacity.
    When more transitions are added than the buffer's max size, it should discard the oldest transitions.
    """
    max_size = 5  # Example max size
    buffer = ReplayBuffer(max_size=max_size)

    # Add more transitions than max_size
    for i in range(max_size + 3):
        transition = (f"state{i}", f"action{i}", f"reward{i}", f"next_state{i}")
        buffer.push(transition)

    assert buffer.buffer[0] == (f"state3", f"action3", f"reward3", f"next_state3"), "Oldest transitions should be discarded"

def test_replay_buffer_sampling():
    """
    Test the sampling functionality of the ReplayBuffer.
    It should return a sample of the requested size, and each item in the sample should be a tuple.
    """
    buffer = ReplayBuffer(max_size=10)

    # Populate buffer
    for i in range(10):
        transition = (f"state{i}", f"action{i}", f"reward{i}", f"next_state{i}")
        buffer.push(transition)

    sample_size = 4
    sample = buffer.sample(sample_size)
    assert len(sample) == sample_size, "Sample size should match requested"
    assert all(isinstance(item, tuple) for item in sample), "Sampled items should be tuples"

def test_replay_buffer_edge_cases():
    """
    Test edge cases for the ReplayBuffer, including sampling from an empty buffer and
    attempting to sample more items than are available in the buffer.
    """
    buffer = ReplayBuffer(max_size=10)

    # Test sampling from an empty buffer
    with pytest.raises(ValueError):
        buffer.sample(1), "Sampling from an empty buffer should raise an error"

    # Add fewer transitions than a sample size
    for i in range(3):
        transition = (f"state{i}", f"action{i}", f"reward{i}", f"next_state{i}")
        buffer.push(transition)

    with pytest.raises(ValueError):
        buffer.sample(4), "Sampling more than available should raise an error"