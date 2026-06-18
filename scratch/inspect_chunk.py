import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

ids = [f"479312_{i}" for i in range(37, 42)]
placeholders = ",".join("?" for _ in ids)
c.execute(f"SELECT id, content FROM document_chunks WHERE id IN ({placeholders})", ids)
for row in c.fetchall():
    print(f"--- ID: {row[0]} ---")
    print(row[1])
