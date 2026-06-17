# Test for vectordb module
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.vectordb.vector_store import VectorStore

def test_vectordb():
    print("Initializing VectorStore...")
    vstore = VectorStore()
    
    print("Testing connection...")
    conn = vstore.get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Found tables in DB: {tables}")
    
    print("Testing FAISS index retrieval...")
    index = vstore.get_faiss_index()
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    
    vstore.close()
    print("VectorStore test complete!")

if __name__ == "__main__":
    test_vectordb()
