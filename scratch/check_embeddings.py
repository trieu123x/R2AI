import sqlite3
import numpy as np

conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

c.execute("SELECT embedding FROM document_chunks WHERE id = '479312_37'")
row = c.fetchone()
if row:
    emb_blob = row[0]
    if emb_blob is None:
        print("Embedding is None!")
    else:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        print("Embedding shape:", emb.shape)
        print("First 5 values:", emb[:5])
else:
    print("Chunk not found!")
