import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))


from foundations.buffer import ReplayBuffer
import pytest
import torch


def test_replay_buffer_capacity_handling():
    max_size = 5
    buffer = ReplayBuffer(max_size=max_size)

    # Add more transitions than max_size
    for i in range(max_size + 3):
        transition = tuple([torch.tensor(i) for _ in range(4)])
        buffer.push(transition)

    assert len(buffer) == max_size
    assert buffer.buffer[0] == tuple([torch.tensor(3) for _ in range(4)]), "Oldest transitions should be discarded."


def test_replay_buffer_sampling():
    buffer = ReplayBuffer(max_size=10)

    # Populate buffer
    for i in range(10):
        transition = tuple([torch.tensor(i) for _ in range(4)])
        buffer.push(transition)

    sample_size = 4
    sample = buffer.sample(sample_size)
    assert len(sample) == sample_size, "Sample size should match requested"
    assert isinstance(sample, list), "Sample should be a list"
    assert all(isinstance(item, tuple) for item in sample), "Sampled items should be tuples"
    assert all(len(item) == 4 for item in sample), "Sampled items should be 4-tuples"


def test_replay_buffer_edge_cases():
    buffer = ReplayBuffer(max_size=10)

    # Test sampling from an empty buffer
    with pytest.raises(ValueError):
        buffer.sample(1), "Sampling from an empty buffer should raise a ValueError."

    # Add fewer transitions than a sample size and try to sample
    for i in range(3):
        transition = tuple([torch.tensor(i) for _ in range(4)])
        buffer.push(transition)
    with pytest.raises(ValueError):
        buffer.sample(4), "Sampling more items than there are in the buffer should raise a ValueError."


if __name__=="__main__":
    test_replay_buffer_capacity_handling()
    test_replay_buffer_sampling()
    test_replay_buffer_edge_cases()
    print("All tests passed")