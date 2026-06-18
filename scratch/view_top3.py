import os
import sys
import io

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever
from retrieval.qwen_generator import extract_evidence

def main():
    retriever = LegalRetriever(use_postgres=False, top_k=5)
    query = "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?"
    results = retriever.retrieve(query, mode="hybrid", top_k=5)
    
    for idx, r in enumerate(results[:3], start=1):
        print(f"=== [Căn cứ {idx}] score={r.score:.4f} ===")
        print(f"Văn bản: {r.legal_type} số {r.doc_number} {r.title}")
        print(f"Điều: {r.article_hint}")
        print("Trích xuất nội dung:")
        print(extract_evidence(r.content, r.article_hint))
        print("=" * 60)

if __name__ == "__main__":
    main()
