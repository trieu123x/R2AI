import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval.local_retriever import LocalRetriever

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"

retriever = LocalRetriever(top_k=5)

# Let's perform a hybrid search
print("Running retrieve with standard hybrid search (no expansion) + rerank...")
res_std = retriever.retrieve(query, mode="hybrid", rerank=True)
for i, r in enumerate(res_std, 1):
    print(f"[{i}] {r.doc_number} | {r.legal_type} | {r.article_hint} | score: {r.score:.4f} | {r.title[:60]}")

print("\nRunning retrieve with manual phrase query + rerank...")
# Let's mock a manual query expansion by overriding retrieve or running FTS + Vector manually
fts_phrases = [
    "công ty", "giữ", "bản chính", "bằng cấp", "nhân viên", "ký", "hợp đồng", "xử lý", "khắc phục",
    "văn bằng", "chứng chỉ", "người lao động", "giao kết hợp đồng", "người sử dụng lao động"
]
phrase_expr = " OR ".join(f'"{p}"' for p in fts_phrases)

# Get FTS results
conn = sqlite3.connect("database/local_chunks.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("""
    SELECT rowid, -bm25(chunks_fts5) AS score
    FROM chunks_fts5
    WHERE chunks_fts5 MATCH ?
    ORDER BY bm25(chunks_fts5)
    LIMIT 15;
""", (phrase_expr,))
fts_rows = cur.fetchall()

from retrieval.local_retriever import _parse_meta_from_content, extract_article_hint, RetrievalResult

fts_results = []
scores_map = {r[0]: float(r[1]) for r in fts_rows}
if fts_rows:
    rowids = [r[0] for r in fts_rows]
    placeholders = ",".join("?" for _ in rowids)
    cur.execute(f"SELECT rowid, id, document_id, chunk_index, content FROM document_chunks WHERE rowid IN ({placeholders})", rowids)
    for row in cur.fetchall():
        r_id = row["rowid"]
        meta = _parse_meta_from_content(row["content"])
        fts_results.append(RetrievalResult(
            chunk_id=str(row["id"]),
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            doc_number=meta["document_number"],
            title=meta["title"],
            legal_type=meta["legal_type"],
            score=scores_map[r_id],
            source="fts_manual",
            article_hint=extract_article_hint(row["content"])
        ))
    fts_results.sort(key=lambda x: x.score, reverse=True)

# Vector search results
vector_results = retriever.vector_search(query, top_k=15)

# Combine using RRF
chunk_map = {}
rrf_scores = {}

def add_ranked(results, weight):
    for rank, r in enumerate(results, start=1):
        cid = r.chunk_id
        if cid not in chunk_map:
            chunk_map[cid] = r
            rrf_scores[cid] = 0.0
        rrf_scores[cid] += weight / (rank + 60)

add_ranked(fts_results, 0.4)
add_ranked(vector_results, 0.6)

sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
hybrid_results = [chunk_map[cid] for cid in sorted_ids[:15]]

# Rerank
reranked_results = retriever.rerank(query, hybrid_results, top_k=5)
print("\nReranked Results with phrase expansion:")
for i, r in enumerate(reranked_results, 1):
    print(f"[{i}] {r.doc_number} | {r.legal_type} | {r.article_hint} | score: {r.score:.4f} | {r.title[:60]}")
