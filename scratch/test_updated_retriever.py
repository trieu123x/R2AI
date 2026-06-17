import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval.local_retriever import LocalRetriever

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
retriever = LocalRetriever(top_k=15)

results = retriever.retrieve(query, mode="hybrid", rerank=True)
print("\nTop 15 retrieved results:")
for idx, r in enumerate(results, 1):
    print(f"[{idx}] {r.doc_number} | {r.legal_type} | {r.article_hint} | score: {r.score:.4f} | {r.title[:60]}")
