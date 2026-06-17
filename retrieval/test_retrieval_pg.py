import sys
import io
import os
import time

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever

def main():
    query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"
    print(f"Query: {query}\n")
    
    print("=== POSTGRESQL HYBRID RETRIEVER WITH RERANKING ===")
    r_pg = LegalRetriever(use_postgres=True, top_k=5)
    
    t_start = time.time()
    results = r_pg.retrieve(query, mode="hybrid", top_k=5, rerank=True)
    print(f"\nRetrieve completed in {time.time() - t_start:.4f}s. Results returned: {len(results)}")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Doc: {res.doc_number} | {res.legal_type}")
        print(f"    Title   : {res.title[:70]}")
        print(f"    Article : {res.article_hint or '-'}")
        print(f"    Score   : {res.score:.4f}")
        print(f"    Source  : {res.source}")
        print(f"    Content : {res.content[:200]}...")
        
    r_pg.close()

if __name__ == "__main__":
    main()
