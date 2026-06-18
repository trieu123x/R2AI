import sqlite3
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

conn = sqlite3.connect("database/local_chunks.db")
cur = conn.cursor()

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"

# Grammar stopwords (very generic)
GRAMMAR_STOPWORDS = {
    "và", "hoặc", "nhưng", "vì", "nên", "của", "các", "những", "là", "có", 
    "trong", "tại", "để", "theo", "được", "bị", "cho", "ra", "vào", "lên", 
    "xuống", "do", "từ", "đến", "bằng", "with", "với", "về", "như", "thì", 
    "mà", "khi", "gì", "này", "đó", "kia", "nọ", "thế", "vậy", "nào", "ai", 
    "đâu", "cái", "con", "chiếc", "nếu", "sẽ", "phải", "sao", "thế"
}

# Clean and tokenize
fts_query = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', query)
fts_query = re.sub(r'\s+', ' ', fts_query).strip().lower()
words = [w for w in fts_query.split() if len(w) >= 2]
keywords = [w for w in words if w not in GRAMMAR_STOPWORDS]

print("Keywords:", keywords)

# OR of all keywords
or_expr = " OR ".join(keywords)
print("OR expression:", or_expr)

cur.execute("""
    SELECT rowid, -bm25(chunks_fts5) AS score
    FROM chunks_fts5
    WHERE chunks_fts5 MATCH ?
    ORDER BY bm25(chunks_fts5)
    LIMIT 10;
""", (or_expr,))
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
    
    print("\nResults for FTS5 with all-keyword OR query:")
    for idx, row in enumerate(details, 1):
        r_id, cid, content = row
        print(f"[{idx}] ID: {cid} | Score: {scores[r_id]:.4f}")
        print(content[:250])
        print("-" * 50)
else:
    print("No results found.")
