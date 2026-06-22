import sqlite3
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT content FROM document_chunks WHERE content LIKE '%20/VBHN-VPQH%' LIMIT 5;")
rows = cursor.fetchall()
for i, r in enumerate(rows):
    print(f"Row {i+1}: {r[0][:250]}...")
conn.close()
