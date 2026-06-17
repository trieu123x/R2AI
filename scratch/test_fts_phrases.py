import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

conn = sqlite3.connect("database/local_chunks.db")
cur = conn.cursor()

# Phrase query
phrase_expr = '"giữ bản chính" OR "văn bằng" OR "chứng chỉ" OR "hợp đồng lao động" OR "khắc phục"'
print("FTS5 query:", phrase_expr)

cur.execute("""
    SELECT rowid, -bm25(chunks_fts5) AS score
    FROM chunks_fts5
    WHERE chunks_fts5 MATCH ?
    ORDER BY bm25(chunks_fts5)
    LIMIT 10;
""", (phrase_expr,))
rows = cur.fetchall()

rowids = [r[0] for r in rows]
scores = {r[0]: float(r[1]) for r in rows}

if rowids:
    placeholders = ",".join("?" for _ in rowids)
    cur.execute(f"""
        SELECT rowid, id, content
        FROM document_chunks
        WHERE rowid IN ({placeholders})
    """, rowids)
    details = cur.fetchall()
    
    # Sort by score descending
    details.sort(key=lambda x: scores[x[0]], reverse=True)
    
    print("\nResults:")
    for idx, row in enumerate(details, 1):
        rowid, cid, content = row
        print(f"[{idx}] ID: {cid} | Score: {scores[rowid]:.4f}")
        print(content[:300])
        print("-" * 50)
else:
    print("No results found.")
