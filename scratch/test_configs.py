import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval.local_retriever import LocalRetriever

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"

retriever = LocalRetriever(top_k=5)

for mode in ["fts", "vector", "hybrid"]:
    for rerank in [False, True]:
        print(f"\n==========================================")
        print(f"Mode: {mode} | Rerank: {rerank}")
        print(f"==========================================")
        results = retriever.retrieve(query, mode=mode, rerank=rerank)
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r.doc_number} | {r.legal_type} | {r.article_hint} | score: {r.score:.4f}")
            print(f"    Title: {r.title[:100]}")
            print(f"    Snippet: {r.content[:150]}")
