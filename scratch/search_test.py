import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

print("Search for 'giữ' in 12/2022/NĐ-CP")
c.execute("SELECT id, content FROM document_chunks WHERE content LIKE '%12/2022/NĐ-CP%' AND content LIKE '%giữ%' LIMIT 10")
for row in c.fetchall():
    print(f"ID: {row[0]}")
    print(row[1][:400])
    print("-" * 60)
