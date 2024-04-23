import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from evaluation.noise_evaluation.noise_evaluation import *
from queueing_tool.queues.queue_servers import *

from unittest.mock import patch, MagicMock
import numpy as np

def test_initialization():
    evaluator = NoiseEvaluator(0.3, 0, 1)
    assert evaluator.frequency == 0.3
    assert evaluator.mean == 0
    assert evaluator.variance == 1

def test_compute_increment():
    evaluator = NoiseEvaluator(1, 0, 1)  # Frequency of 1 ensures noise is always added
    np.random.seed(0)  # Seed for reproducibility
    noise = evaluator.compute_increment()
    assert noise != 0  # Check noise is added

    evaluator = NoiseEvaluator(0, 0, 1)  # Frequency of 0 ensures noise is never added
    noise = evaluator.compute_increment()
    assert noise == 0  # Check no noise is added

# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()