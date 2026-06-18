import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval.local_retriever import LocalRetriever

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
retriever = LocalRetriever(top_k=5)

fts_results = retriever.fts_search(query, top_k=30)
print("FTS Search top 30 results from LocalRetriever:")
for idx, r in enumerate(fts_results, 1):
    print(f"[{idx}] {r.chunk_id} | {r.doc_number} | {r.legal_type} | {r.article_hint} | score: {r.score:.4f}")
