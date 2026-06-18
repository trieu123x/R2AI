# Basic test for ingestion
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

def test_loader_exists():
    loader_path = os.path.join(PROJECT_ROOT, "src/ingestion/loader.py")
    assert os.path.exists(loader_path)
    print("Ingestion loader script exists and is located at:", loader_path)

if __name__ == "__main__":
    test_loader_exists()
