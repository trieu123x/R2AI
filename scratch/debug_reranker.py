import sqlite3
import faiss
import sys
import time
import os

print("Importing SentenceTransformer...")
from sentence_transformers import SentenceTransformer

print("Testing SQLite...")
db_path = os.path.join(os.path.abspath("database"), "legal_corpus.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
cur = conn.cursor()
cur.execute("PRAGMA journal_mode=WAL;")
cur.execute("PRAGMA synchronous=OFF;")
cur.execute("PRAGMA cache_size=-100000;") # 100MB cache
cur.execute("PRAGMA temp_store=MEMORY;")
cur.execute("PRAGMA mmap_size=500000000;") # 500MB memory map
print("SQLite connected.")

print("Loading bi-encoder...")
m1 = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device="cuda")

print("Loading faiss...")
index = faiss.read_index(os.path.join(os.path.abspath("database"), "local_chunks.index"))

print("Initializing QwenGenerator...")
from retrieval.qwen_generator import QwenGenerator
generator = QwenGenerator("Qwen/Qwen2.5-0.5B-Instruct")

def rerank():
    import torch
    from sentence_transformers import CrossEncoder
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(os.path.abspath("retrieval"), "bge-reranker-v2-m3")
    print(f"model_path = {model_path}")
    
    print("Loading CrossEncoder...")
    try:
        automodel_args = {}
        if device == "cuda":
            automodel_args = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
        model = CrossEncoder(model_path, device=device, automodel_args=automodel_args)
        print("Success!")
        print("Running quick test prediction...")
        scores = model.predict([["câu hỏi", "nội dung luật"]], show_progress_bar=False)
        print(f"Prediction success! Scores: {scores}")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

rerank()
