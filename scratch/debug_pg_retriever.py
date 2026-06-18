import sys
import io
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever

def main():
    r = LegalRetriever(use_postgres=True)
    query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
    print(f"Query: {query}")
    
    # Test vector search
    vector_candidates = r.vector_search(query, top_k=50)
    print(f"\nVector candidates count: {len(vector_candidates)}")
    for i, c in enumerate(vector_candidates[:10], 1):
        print(f"  [{i}] Doc: {c.doc_number} | Index: {c.chunk_index} | Title: {c.title[:50]} | Score: {c.score:.4f}")

    # Test FTS search
    fts_candidates = r.fts_search(query, top_k=50)
    print(f"\nFTS candidates count: {len(fts_candidates)}")
    for i, c in enumerate(fts_candidates[:10], 1):
        print(f"  [{i}] Doc: {c.doc_number} | Index: {c.chunk_index} | Title: {c.title[:50]} | Score: {c.score:.4f}")

    # Reranking on vector candidates
    unique = []
    for c in vector_candidates:
        if not any(r.are_chunks_duplicate(c, ur) for ur in unique):
            unique.append(c)
    print(f"\nUnique candidates count: {len(unique)}")
    
    expanded = r.expand_to_parent_article(unique)
    print(f"Expanded candidates count: {len(expanded)}")
    
    reranked = r.rerank(query, expanded, top_k=None)
    print("\nReranked scores on vector candidates:")
    for i, res in enumerate(reranked[:10], 1):
        print(f"  [{i}] Doc: {res.doc_number} | Article: {res.article_hint} | Score: {res.score:.4f}")

if __name__ == "__main__":
    main()
