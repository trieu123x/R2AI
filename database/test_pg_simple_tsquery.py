import sys
import io
import time
import psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PG_CONN_STR = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"
conn = psycopg2.connect(PG_CONN_STR)
cur = conn.cursor()

# Query using & instead of <-> for phrases
fts_expr_simple = "((người & lao & động) | (lao & động) | (nhân & viên) | (hợp & đồng & lao & động)) & (giữ | (thu & giữ) | (tạm & giữ) | (văn & bằng) | (chứng & chỉ) | (giấy & tờ & tùy & thân) | (bản & chính) | (bằng & cấp))"

print("Explain query plan for simple conjunction:")
cur.execute("""
    EXPLAIN SELECT c.id, c.document_id, c.chunk_index, c.content,
           ts_rank(to_tsvector('simple', c.content), to_tsquery('simple', %s)) AS score,
           d.document_number, d.title, d.legal_type
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
    ORDER BY score DESC
    LIMIT 15;
""", (fts_expr_simple, fts_expr_simple))
for row in cur.fetchall():
    print(row[0])

print("\nExecuting query...")
t0 = time.time()
cur.execute("""
    SELECT c.id, c.document_id, c.chunk_index, c.content,
           ts_rank(to_tsvector('simple', c.content), to_tsquery('simple', %s)) AS score,
           d.document_number, d.title, d.legal_type
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
    ORDER BY score DESC
    LIMIT 15;
""", (fts_expr_simple, fts_expr_simple))
rows = cur.fetchall()
print(f"Executed in {time.time()-t0:.4f}s. Results returned: {len(rows)}")
for idx, row in enumerate(rows[:5], 1):
    print(f"[{idx}] doc={row[5]}, title={row[6][:50]}, score={row[4]:.4f}")

conn.close()
