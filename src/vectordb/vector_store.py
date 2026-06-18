import os
import time
import sqlite3
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

class VectorStore:
    def __init__(self, db_path: str = LOCAL_DB_PATH):
        self.db_path = db_path
        self._conn = None
        self._faiss_index = None

    def get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            try:
                cur = self._conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=OFF;")
                cur.execute("PRAGMA cache_size=-2000000;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA mmap_size=3000000000;")
            except Exception as e:
                print(f"[vector_store] Warning: Failed to apply SQLite PRAGMAs: {e}")
        return self._conn

    def get_faiss_index(self):
        import faiss
        index_path = os.path.join(os.path.dirname(self.db_path), "local_chunks.index")
        if self._faiss_index is None:
            if os.path.exists(index_path):
                t0 = time.time()
                self._faiss_index = faiss.read_index(index_path)
                print(f"[vector_store] FAISS index loaded in {time.time()-t0:.2f}s", flush=True)
            else:
                print(f"[vector_store] FAISS index not found. Building index...", flush=True)
                t0 = time.time()
                conn = self.get_conn()
                cur = conn.cursor()
                cur.execute("SELECT rowid, embedding FROM document_chunks WHERE embedding IS NOT NULL;")
                rowids = []
                embeddings = []
                for rowid, emb_bytes in cur:
                    if not emb_bytes:
                        continue
                    emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    rowids.append(rowid)
                    embeddings.append(emb)
                if not embeddings:
                    raise ValueError("No embeddings found in database!")
                embeddings = np.array(embeddings, dtype=np.float32)
                rowids = np.array(rowids, dtype=np.int64)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                dim = embeddings.shape[1]
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIDMap(quantizer)
                index.add_with_ids(embeddings, rowids)
                faiss.write_index(index, index_path)
                self._faiss_index = index
                print(f"[vector_store] FAISS index built in {time.time()-t0:.2f}s", flush=True)
        return self._faiss_index

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
