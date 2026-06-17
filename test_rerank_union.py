import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval.local_retriever import LocalRetriever, _parse_meta_from_content, extract_article_hint, RetrievalResult

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
retriever = LocalRetriever(top_k=30)

fts_phrases = [
    "công ty", "giữ", "bản chính", "bằng cấp", "nhân viên", "ký", "hợp đồng", "xử lý", "khắc phục",
    "văn bằng", "chứng chỉ", "người lao động", "giao kết hợp đồng", "người sử dụng lao động"
]
phrase_expr = " OR ".join(f'"{p}"' for p in fts_phrases)

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

fts_results = []
if fts_rows:
    rowids = [r[0] for r in fts_rows]
    scores_map = {r[0]: float(r[1]) for r in fts_rows}
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
            source="fts",
            article_hint=extract_article_hint(row["content"])
        ))
    fts_results.sort(key=lambda x: x.score, reverse=True)

vector_results = retriever.vector_search(query, top_k=15)

union_map = {}
for r in fts_results:
    union_map[r.chunk_id] = r
for r in vector_results:
    union_map[r.chunk_id] = r

candidate_pool = list(union_map.values())
reranked = retriever.rerank(query, candidate_pool, top_k=30)
print("\nTOP 15 RERANKED RESULTS:")
for i, r in enumerate(reranked[:15], 1):
    print(f"[{i}] CID: {r.chunk_id} | {r.doc_number} | {r.legal_type} | {r.article_hint} | score: {r.score:.6f} | source: {r.source}")
