import os
import sys
import time
from typing import List, Optional

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import RetrievalResult

class PipelineReranker:
    """
    Step 5: Reranker (PhoRanker)
    Step 6: Top 5
    """
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", top_n=5): # Hoặc đường dẫn thư mục model BAAI bạn đã tải về (VD: "C:/path/to/bge-reranker-v2-m3")
        self.model_name = model_name
        self.top_n = top_n
        self._reranker = None
    def _lazy_load(self):
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[pipeline] Loading reranker model {self.model_name} on '{device}'...", flush=True)
            t0 = time.time()
            automodel_args = {}
            if device == "cuda":
                automodel_args = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True
                }
            self._reranker = CrossEncoder(self.model_name, device=device, automodel_args=automodel_args)
            print(f"[pipeline] Reranker loaded in {time.time()-t0:.1f}s", flush=True)

    def rerank_and_filter(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank danh sách ứng viên và trả về Top N."""
        if not results:
            return []

        self._lazy_load()

        # Tokenize word level for Vietnamese query
        segmented_query = query
        try:
            import underthesea
            segmented_query = underthesea.word_tokenize(query, format="text")
        except Exception:
            pass

        pairs = []
        for r in results:
            # We use content as default if segmented_content is not available directly
            pairs.append([segmented_query, r.content])

        t_rerank = time.time()
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        print(f"[pipeline] Reranked {len(pairs)} pairs in {time.time() - t_rerank:.2f}s.")

        LEGAL_BOOST = {
            "Luật": 0.3,
            "Nghị định": 0.2,
            "Thông tư": 0.1
        }

        for r, score in zip(results, scores):
            final_score = float(score)
            # Thêm điểm thưởng dựa trên loại văn bản pháp lý
            bonus = LEGAL_BOOST.get(r.legal_type, 0.0)
            if r.article_hint and "Điều" in r.article_hint:
                bonus += 0.1
            r.score = final_score + bonus
            r.source = f"rerank({r.source})"

        # Sort descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Lọc max 2 chunk mỗi document
        filtered_results = []
        doc_count = {}
        for r in results:
            doc_id = r.document_id
            if doc_count.get(doc_id, 0) < 2:
                filtered_results.append(r)
                doc_count[doc_id] = doc_count.get(doc_id, 0) + 1

        # Step 6: Top 5
        return filtered_results[:self.top_n]
