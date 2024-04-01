from buffer import ReplayBuffer
import pytest


def test_replay_buffer_capacity_handling():
    max_size = 5  # Example max size
    buffer = ReplayBuffer(max_size=max_size)

    # Add more transitions than max_size
    for i in range(max_size + 3):
        transition = (f"state{i}", f"action{i}", f"reward{i}", f"next_state{i}")
        buffer.push(transition)

    # assert len(buffer) == max_size
    assert buffer.buffer[0] == (f"state3", f"action3", f"reward3", f"next_state3"), "Oldest transitions should be discarded"

def test_replay_buffer_sampling():
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

if __name__=="__main__":
    test_replay_buffer_capacity_handling()
    test_replay_buffer_sampling()
    test_replay_buffer_edge_cases()
    print("All tests passed")