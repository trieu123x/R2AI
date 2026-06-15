import os
import sys
import time
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
BATCH_SIZE = 5000  # Load 5k at a time to process

def main():
    if not os.path.exists(LOCAL_DB_PATH):
        print(f"[-] Local SQLite database not found at: {LOCAL_DB_PATH}")
        return

    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()

    # Check if embedding column exists, if not add it
    cursor.execute("PRAGMA table_info(document_chunks);")
    columns = [col[1] for col in cursor.fetchall()]
    if "embedding" not in columns:
        print("[+] Adding 'embedding' column to local SQLite...")
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN embedding BLOB;")
        conn.commit()

    # Get count of total chunks and remaining chunks
    cursor.execute("SELECT COUNT(*) FROM document_chunks;")
    total_chunks = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL;")
    remaining_chunks = cursor.fetchone()[0]

    print(f"[+] Total chunks: {total_chunks:,}")
    print(f"[+] Chunks remaining to embed: {remaining_chunks:,}")

    if remaining_chunks == 0:
        print("[+] All chunks already have embeddings. Exiting.")
        conn.close()
        return

    print(f"[+] Loading model '{MODEL_NAME}' on CPU...")
    start_load = time.time()
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    print(f"[+] Model loaded in {time.time() - start_load:.2f}s.")

    print("[+] Starting embedding generation...")
    processed = 0
    start_time = time.time()

    while True:
        # Fetch batch
        cursor.execute("SELECT id, content FROM document_chunks WHERE embedding IS NULL LIMIT ?;", (BATCH_SIZE,))
        rows = cursor.fetchall()
        if not rows:
            break
            
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        
        # Generate embeddings (internally batching by 128 for CPU optimization)
        embeddings = model.encode(texts, batch_size=128, show_progress_bar=False)
        
        # Prepare updates
        update_data = []
        for i, emb in enumerate(embeddings):
            # Serialize numpy array to bytes
            emb_bytes = emb.astype(np.float32).tobytes()
            update_data.append((emb_bytes, ids[i]))
            
        # Bulk update
        cursor.executemany(
            "UPDATE document_chunks SET embedding = ? WHERE id = ?;",
            update_data
        )
        conn.commit()
        
        processed += len(rows)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = remaining_chunks - processed
        eta = remaining / rate if rate > 0 else 0
        
        print(
            f"  Progress: {processed:>8,} / {remaining_chunks:,} ({processed/remaining_chunks*100:5.1f}%) | "
            f"Speed: {rate:>5.1f} chunks/s | ETA: {eta:>5.0f}s",
            flush=True
        )

    conn.close()
    print("[+] Local embedding generation completed successfully!")

if __name__ == "__main__":
    main()
