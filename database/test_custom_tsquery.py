import sys
import io
import re
import psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def fts_to_postgres_tsquery(expr: str) -> str:
    def replace_phrase(match):
        phrase = match.group(1).strip()
        words = phrase.split()
        return "(" + " <-> ".join(f"'{w}'" for w in words) + ")"
        
    res = re.sub(r'"([^"]+)"', replace_phrase, expr)
    res = res.replace(" OR ", " | ")
    res = res.replace(" AND ", " & ")
    return res

PG_CONN_STR = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"
conn = psycopg2.connect(PG_CONN_STR)
cur = conn.cursor()

fts_expr = '("người lao động" OR "lao động" OR "nhân viên" OR "hợp đồng lao động") AND ("giữ" OR "thu giữ" OR "tạm giữ" OR "văn bằng" OR "chứng chỉ" OR "giấy tờ tùy thân" OR "bản chính" OR "bằng cấp")'
pg_tsquery = fts_to_postgres_tsquery(fts_expr)
print("Generated pg_tsquery:")
print(pg_tsquery, flush=True)

# Test to_tsquery
cur.execute("SELECT to_tsquery('simple', %s)", (pg_tsquery,))
tsquery = cur.fetchone()[0]
print("\nPostgres to_tsquery parsed:")
print(tsquery, flush=True)

# Find matching documents in Postgres
cur.execute("""
    SELECT c.content, d.document_number, d.title
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
    LIMIT 5
""", (pg_tsquery,))
rows = cur.fetchall()
print(f"\nFound {len(rows)} matching results:")
for idx, row in enumerate(rows, 1):
    print(f"[{idx}] doc={row[1]}, title={row[2][:50]}")
    # print(f"    content={row[0][:200]}...")

conn.close()
