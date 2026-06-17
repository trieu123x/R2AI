# Test for embeddings module
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.embeddings.embedder import VietnameseBiEncoder

def test_embedder():
    print("Initializing VietnameseBiEncoder...")
    encoder = VietnameseBiEncoder()
    print("Testing embed_query...")
    embedding = encoder.embed_query("Ươm tạo doanh nghiệp nhỏ và vừa")
    print(f"Successfully generated embedding vector of shape: {embedding.shape}")
    assert len(embedding) > 0

if __name__ == "__main__":
    test_embedder()
