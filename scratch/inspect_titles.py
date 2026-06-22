import sqlite3
import os
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT content FROM document_chunks WHERE content LIKE '%20/VBHN-VPQH%';")
for r in cursor:
    content = r[0] or ""
    header_end = content.find(" | Nội dung:")
    if header_end != -1:
        prefix = content[:header_end]
        m = re.search(r'Văn bản:\s*(.+?)\s*\(([^)]+)\)', prefix)
        if m:
            print(f"Prefix: {prefix!r}")
            print(f"Title: {m.group(1)!r} | DocNum: {m.group(2)!r}")
            break
conn.close()
