import sqlite3
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

c.execute("SELECT content FROM document_chunks")
docs = {}

from retrieval.local_retriever import _parse_meta_from_content

count = 0
for row in c.fetchall():
    content = row[0]
    meta = _parse_meta_from_content(content)
    doc_key = (meta["legal_type"], meta["document_number"], meta["title"])
    docs[doc_key] = docs.get(doc_key, 0) + 1
    count += 1
    if count % 100000 == 0:
        print(f"Processed {count} chunks...")

print(f"Total chunks in database: {count}")
print(f"Total unique documents in database: {len(docs)}")
print("\nList of unique documents and their chunk counts:")
for doc, num_chunks in sorted(docs.items(), key=lambda x: -x[1]):
    print(f"- {doc[0]} {doc[1]} {doc[2]}: {num_chunks} chunks")
