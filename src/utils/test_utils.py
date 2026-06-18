# Test for utilities
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.context_compressor import ContextCompressor
from src.utils.validator import PipelineValidator

def test_utils():
    compressor = ContextCompressor()
    validator = PipelineValidator()
    print("Successfully initialized ContextCompressor and PipelineValidator")

if __name__ == "__main__":
    test_utils()
