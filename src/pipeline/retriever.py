from typing import List
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.local_retriever import LocalRetriever
from retrieval.retriever import RetrievalResult

class PipelineRetriever:
    """
    Step 3: BM25 (top 10) + Embedding (top 10)
    Step 4: Merge
    """
    def __init__(self, db_path=None, top_k_each=50):
        if db_path:
            self.retriever = LocalRetriever(db_path=db_path)
        else:
            self.retriever = LocalRetriever()
        self.top_k_each = top_k_each

    def retrieve_and_merge(self, query: str) -> List[RetrievalResult]:
        """Thực hiện song song FTS5 và Vector Search, sau đó merge bằng RRF."""
        # Step 3: Lấy top_k_each (10) cho mỗi phương pháp
        fts_results = self.retriever.fts_search(query, top_k=self.top_k_each)
        vector_results = self.retriever.vector_search(query, top_k=self.top_k_each)

        # Step 4: Merge (RRF)
        chunk_map = {}
        rrf_scores = {}
        
        # RRF k constant
        rrf_k = 60

        def add_ranked(results: List[RetrievalResult], weight: float):
            for rank, r in enumerate(results, start=1):
                cid = r.chunk_id
                if cid not in chunk_map:
                    chunk_map[cid] = r
                    rrf_scores[cid] = 0.0
                rrf_scores[cid] += weight / (rank + rrf_k)

        # Merge with equal weights
        add_ranked(fts_results, 0.5)
        add_ranked(vector_results, 0.5)

        # Lọc trùng lặp ngữ nghĩa (đã có hàm trong LocalRetriever)
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        merged_results = []
        for cid in sorted_ids:
            r = chunk_map[cid]
            # Tạo kết quả mới với score được cập nhật
            merged_results.append(RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                content=r.content,
                doc_number=r.doc_number,
                title=r.title,
                legal_type=r.legal_type,
                score=rrf_scores[cid],
                source="merged_rrf",
                article_hint=r.article_hint,
            ))

        # Lọc trùng
        unique_results = []
        for r in merged_results:
            is_dup = False
            for ur in unique_results:
                if self.retriever.are_chunks_duplicate(r, ur):
                    is_dup = True
                    break
            if not is_dup:
                unique_results.append(r)

        # Gộp chunk thành Article
        grouped = {}
        for r in unique_results:
            key = (r.document_id, r.article_hint)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
            
        article_results = []
        for key, chunks in grouped.items():
            chunks.sort(key=lambda x: x.chunk_index)
            base_chunk = chunks[0]
            if len(chunks) == 1 or not base_chunk.article_hint:
                article_results.append(base_chunk)
            else:
                combined_content = "\n...\n".join([c.content for c in chunks])
                max_score = max([c.score for c in chunks])
                
                new_res = RetrievalResult(
                    chunk_id=base_chunk.chunk_id,
                    document_id=base_chunk.document_id,
                    chunk_index=base_chunk.chunk_index,
                    content=combined_content,
                    doc_number=base_chunk.doc_number,
                    title=base_chunk.title,
                    legal_type=base_chunk.legal_type,
                    score=max_score,
                    source=base_chunk.source + "_aggregated",
                    article_hint=base_chunk.article_hint,
                )
                article_results.append(new_res)
                
        article_results.sort(key=lambda x: x.score, reverse=True)
        return article_results
