import os
import sys
import time
import sqlite3
import numpy as np
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
MODEL_NAME = "BAAI/bge-m3"

def worker_process(input_queue, output_queue, thread_id):
    try:
        # Use 2 threads per process for PyTorch to optimize core utilization
        torch.set_num_threads(2)
        
        # Load independent model copy per process
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(MODEL_NAME, device=device)
        
        batch_ids = []
        batch_texts = []
        batch_size = 64
        
        while True:
            item = input_queue.get()
            if item is None:
                # Process final batch
                if batch_texts:
                    embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
                    for i, emb in enumerate(embeddings):
                        emb_bytes = emb.astype(np.float32).tobytes()
                        output_queue.put((batch_ids[i], emb_bytes))
                break
                
            chunk_id, content = item
            batch_ids.append(chunk_id)
            batch_texts.append(content)
            
            if len(batch_texts) == batch_size:
                embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
                for i, emb in enumerate(embeddings):
                    emb_bytes = emb.astype(np.float32).tobytes()
                    output_queue.put((batch_ids[i], emb_bytes))
                batch_ids = []
                batch_texts = []
    except Exception as e:
        print(f"\n[-] Worker-{thread_id} error: {e}", flush=True)
            
    # Send sentinel to writer signaling this worker finished
    output_queue.put(None)

def writer_process(output_queue, num_workers, remaining_chunks):
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        # SQLite performance tunings
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=OFF;")
        
        processed = 0
        start_time = time.time()
        active_workers = num_workers
        
        batch = []
        batch_size = 1000
        
        while active_workers > 0:
            item = output_queue.get()
            if item is None:
                active_workers -= 1
                continue
                
            batch.append((item[1], item[0]))  # Swapped to (embedding, id)
            if len(batch) == batch_size:
                cursor.executemany("UPDATE document_chunks SET embedding = ? WHERE id = ?;", batch)
                conn.commit()
                processed += len(batch)
                batch = []
                
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = remaining_chunks - processed
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  Progress: {processed:>8,} / {remaining_chunks:,} ({processed/remaining_chunks*100:5.1f}%) | "
                    f"Speed: {rate:>5.1f} chunks/s | ETA: {eta:>5.0f}s",
                    flush=True
                )
                
        # Flush remaining
        if batch:
            cursor.executemany("UPDATE document_chunks SET embedding = ? WHERE id = ?;", batch)
            conn.commit()
            processed += len(batch)
            
        conn.close()
        elapsed = time.time() - start_time
        print(f"\n[+] Multiprocessing Completed: {processed:,} chunks in {elapsed:.1f}s ({processed/elapsed:.1f} chunks/s avg)")
    except Exception as e:
        print(f"\n[-] Writer process error: {e}", flush=True)

def main():
    if not os.path.exists(LOCAL_DB_PATH):
        print(f"[-] Local SQLite database not found at: {LOCAL_DB_PATH}")
        return

    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()

    # Ensure embedding column exists
    cursor.execute("PRAGMA table_info(document_chunks);")
    columns = [col[1] for col in cursor.fetchall()]
    if "embedding" not in columns:
        print("[+] Adding 'embedding' column to local SQLite...")
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN embedding BLOB;")
        conn.commit()

    print("[+] Loading missing chunks into memory from SQLite...")
    cursor.execute("SELECT id, content FROM document_chunks WHERE embedding IS NULL;")
    all_rows = cursor.fetchall()
    remaining_chunks = len(all_rows)
    conn.close()

    print(f"[+] Loaded {remaining_chunks:,} chunks into memory.")
    if remaining_chunks == 0:
        print("[+] All chunks already have embeddings.")
        return

    # Queues
    input_q = mp.Queue(maxsize=15000)
    output_q = mp.Queue()

    # Safe worker count to prevent memory pressure
    num_workers = 3
    print(f"[+] Spawning {num_workers} worker processes...")

    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(input_q, output_q, i))
        p.start()
        workers.append(p)

    # Start writer process
    writer = mp.Process(target=writer_process, args=(output_q, num_workers, remaining_chunks))
    writer.start()

    # Feed input queue from memory list
    for row in all_rows:
        input_q.put(row)

    # Sentinels
    for _ in range(num_workers):
        input_q.put(None)

    # Wait for workers and writer
    for p in workers:
        p.join()
        
    writer.join()

if __name__ == "__main__":
    mp.freeze_support()  # Windows safety check
    main()
