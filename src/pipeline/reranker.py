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
    def __init__(self, model_name="itdainb/PhoRanker", top_n=5):
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
            self._reranker = CrossEncoder(self.model_name, device=device)
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

        for r, score in zip(results, scores):
            r.score = float(score)
            r.source = f"rerank({r.source})"

        # Sort descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Step 6: Top 5
        return results[:self.top_n]
