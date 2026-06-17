"""
retriever.py
============
Pipeline Retrieval cho hệ thống RAG pháp lý tiếng Việt.

Ba chế độ tìm kiếm:
  1. vector   — cosine similarity trên pgvector (embedding 768-dim bkai-bi-encoder)
  2. fts      — Full-Text Search bằng PostgreSQL tsvector (BM25-style ranking)
  3. hybrid   — Kết hợp vector + FTS bằng thuật toán Reciprocal Rank Fusion (RRF)

Kết quả trả về danh sách RetrievalResult với:
  - chunk_id, document_id, chunk_index, content (đoạn văn bản)
  - doc_number, title, legal_type (metadata văn bản)
  - score (điểm tổng hợp), source (nguồn: vector/fts/hybrid)
  - article_hint (gợi ý số Điều nếu chunk chứa "Điều X")
"""

import os
import sys
import re
import time
import numpy as np
import psycopg2
from dataclasses import dataclass, field
from typing import List, Optional, Literal

# Thêm project root vào path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import Config


# ─── Data class kết quả ────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: int
    chunk_index: int
    content: str
    doc_number: str
    title: str
    legal_type: str
    score: float
    source: str                          # "vector" | "fts" | "hybrid"
    article_hint: Optional[str] = None   # "Điều 4", "Điều 5", ...

    def format_relevant_doc(self) -> str:
        """Định dạng relevant_docs cho file nộp bài."""
        return f"{self.doc_number}|{self.legal_type} {self.doc_number} {self.title}"

    def format_relevant_article(self) -> str:
        """Định dạng relevant_articles cho file nộp bài."""
        doc_str = self.format_relevant_doc()
        article = self.article_hint or "Toàn bộ"
        return f"{doc_str}|{article}"


# ─── Regex trích xuất số Điều ──────────────────────────────────────────────────

_ARTICLE_PATTERN = re.compile(
    r"(?:^|\s)(Điều\s+\d+[a-zA-Z]?)",
    re.MULTILINE | re.UNICODE
)

def extract_article_hint(content: str) -> Optional[str]:
    """Trả về điều luật đầu tiên tìm thấy trong chunk (vd: 'Điều 4')."""
    m = _ARTICLE_PATTERN.search(content)
    if m:
        return m.group(1).strip()
    return None


# ─── Lớp Retriever chính ───────────────────────────────────────────────────────

class LegalRetriever:
    """
    Retriever kết nối trực tiếp tới Supabase PostgreSQL.

    Tham số khởi tạo:
        top_k        : số kết quả trả về (mặc định 10)
        vector_weight: trọng số cho vector score trong hybrid (0.0–1.0)
        fts_weight   : trọng số cho FTS score trong hybrid (0.0–1.0)
        rrf_k        : hằng số làm mịn RRF (mặc định 60)
    """

    def __init__(
        self,
        top_k: int = 10,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight
        self.rrf_k = rrf_k
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._model = None   # lazy-load embedding model

    # ── Kết nối DB ──────────────────────────────────────────────────────────────

    def _get_conn(self):
        """Trả về kết nối DB (tạo mới nếu chưa có / đã đóng)."""
        if self._conn is None or self._conn.closed:
            Config.validate()
            self._conn = psycopg2.connect(
                Config.DATABASE_URL,
                connect_timeout=15,
                options="-c statement_timeout=30000",  # 30s timeout
            )
            self._conn.autocommit = True
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── Embedding query ──────────────────────────────────────────────────────────

    def _get_model(self):
        """Lazy-load SentenceTransformer model."""
        if self._model is None:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[retriever] Đang load model bkai-bi-encoder trên device '{device}'...", flush=True)
            t0 = time.time()
            self._model = SentenceTransformer(
                "bkai-foundation-models/vietnamese-bi-encoder",
                device=device,
            )
            print(f"[retriever] Model loaded in {time.time()-t0:.1f}s", flush=True)
        return self._model

    def embed_query(self, query: str) -> List[float]:
        """Chuyển câu hỏi thành vector embedding."""
        model = self._get_model()
        vec = model.encode([query], show_progress_bar=False)[0]
        return vec.astype(np.float32).tolist()

    # ── Các hàm SQL helper ───────────────────────────────────────────────────────

    @staticmethod
    def _rows_to_results(rows, source: str) -> List[RetrievalResult]:
        results = []
        for row in rows:
            (chunk_id, doc_id, chunk_idx, content,
             doc_number, title, legal_type, score) = row
            results.append(RetrievalResult(
                chunk_id=str(chunk_id),
                document_id=doc_id,
                chunk_index=chunk_idx,
                content=content or "",
                doc_number=doc_number or "",
                title=title or "",
                legal_type=legal_type or "",
                score=float(score) if score else 0.0,
                source=source,
                article_hint=extract_article_hint(content or ""),
            ))
        return results

    # ── Vector Search ────────────────────────────────────────────────────────────

    def vector_search(self, query: str, top_k: Optional[int] = None, gpt_mode: Optional[str] = None) -> List[RetrievalResult]:
        """
        Tìm kiếm theo embedding similarity (cosine).
        Yêu cầu: chunks đã có embedding trong DB.
        """
        k = top_k or self.top_k
        
        # GPT enhancement
        search_query = query
        if gpt_mode == "hyde" and self.enhancer.is_active:
            search_query = self.enhancer.generate_hyde(query)
        elif gpt_mode == "expand" and self.enhancer.is_active:
            search_query = self.enhancer.expand_query(query)
            
        embedding = self.embed_query(search_query)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        sql = """
            SELECT
                dc.id,
                dc.document_id,
                dc.chunk_index,
                dc.content,
                d.document_number,
                d.title,
                d.legal_type,
                1 - (dc.embedding <=> %s::vector) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, (embedding_str, embedding_str, k))
            rows = cur.fetchall()

        return self._rows_to_results(rows, "vector")

    # ── Full-Text Search ─────────────────────────────────────────────────────────

    def fts_search(self, query: str, top_k: Optional[int] = None, gpt_mode: Optional[str] = None) -> List[RetrievalResult]:
        """
        Tìm kiếm Full-Text bằng PostgreSQL ts_rank (BM25-style).
        Không cần embedding, hoạt động ngay khi chỉ có content.
        """
        k = top_k or self.top_k
        
        # GPT query expansion
        search_query = query
        if gpt_mode in ("expand", "hyde") and self.enhancer.is_active:
            search_query = self.enhancer.expand_query(query)

        sql = """
            SELECT
                dc.id,
                dc.document_id,
                dc.chunk_index,
                dc.content,
                d.document_number,
                d.title,
                d.legal_type,
                ts_rank_cd(
                    to_tsvector('simple', dc.content),
                    plainto_tsquery('simple', %s)
                ) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE to_tsvector('simple', dc.content) @@ plainto_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s;
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, (search_query, search_query, k))
            rows = cur.fetchall()

        return self._rows_to_results(rows, "fts")

    # ── Hybrid Search (RRF) ──────────────────────────────────────────────────────

    def hybrid_search(self, query: str, top_k: Optional[int] = None, gpt_mode: Optional[str] = None) -> List[RetrievalResult]:
        """
        Kết hợp vector + FTS bằng Reciprocal Rank Fusion (RRF).

        RRF score = Σ  weight_i / (rank_i + k)

        Ưu tiên các chunk xuất hiện cao trên cả hai danh sách.
        """
        k = top_k or self.top_k
        fetch_k = k * 3   # lấy nhiều hơn để merge

        # For hybrid, vector uses the specified gpt_mode (e.g. HyDE), while FTS uses expand if GPT is active
        vector_gpt_mode = gpt_mode
        fts_gpt_mode = "expand" if gpt_mode else None

        vector_results = self.vector_search(query, top_k=fetch_k, gpt_mode=vector_gpt_mode)
        fts_results    = self.fts_search(query,    top_k=fetch_k, gpt_mode=fts_gpt_mode)

        # Map chunk_id → RetrievalResult
        chunk_map: dict[str, RetrievalResult] = {}
        rrf_scores: dict[str, float] = {}

        def add_ranked_list(results: List[RetrievalResult], weight: float):
            for rank, r in enumerate(results, start=1):
                cid = r.chunk_id
                if cid not in chunk_map:
                    chunk_map[cid] = r
                    rrf_scores[cid] = 0.0
                rrf_scores[cid] += weight / (rank + self.rrf_k)

        add_ranked_list(vector_results, self.vector_weight)
        add_ranked_list(fts_results,    self.fts_weight)

        # Sắp xếp theo RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for cid in sorted_ids[:k]:
            r = chunk_map[cid]
            results.append(RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                content=r.content,
                doc_number=r.doc_number,
                title=r.title,
                legal_type=r.legal_type,
                score=rrf_scores[cid],
                source="hybrid",
                article_hint=r.article_hint,
            ))
        return results

    def are_chunks_duplicate(self, c1: RetrievalResult, c2: RetrievalResult) -> bool:
        """Kiểm tra xem hai chunk có trùng lặp nội dung đáng kể hay không."""
        if c1.document_id == c2.document_id and c1.chunk_index == c2.chunk_index:
            return True
            
        body1 = c1.content.split("| Nội dung:")[-1].strip()
        body2 = c2.content.split("| Nội dung:")[-1].strip()
        
        def normalize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return set(text.split())
            
        words1 = normalize(body1)
        words2 = normalize(body2)
        if not words1 or not words2:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union)
        
        if jaccard > 0.8:
            return True
            
        if c1.article_hint and c2.article_hint and c1.article_hint == c2.article_hint:
            if "điều" in c1.article_hint.lower() or "khoản" in c1.article_hint.lower():
                if jaccard > 0.6:
                    return True
        return False

    # ── Entry point chung ────────────────────────────────────────────────────────

    def rerank(self, query: str, results: List[RetrievalResult], top_k: Optional[int] = None) -> List[RetrievalResult]:
        if not results:
            return []
            
        if getattr(self, "_reranker", None) is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[retriever] Loading reranker model itdainb/PhoRanker on device '{device}'...", flush=True)
            t0 = time.time()
            self._reranker = CrossEncoder("itdainb/PhoRanker", device=device)
            if device == "cuda" and hasattr(self._reranker.model, "half"):
                self._reranker.model.half()
            print(f"[retriever] Reranker model loaded in {time.time()-t0:.1f}s", flush=True)
            
        segmented_query = query
        try:
            import underthesea
            segmented_query = underthesea.word_tokenize(query, format="text")
        except Exception as e:
            pass
            
        pairs = []
        for r in results:
            pairs.append([segmented_query, r.content])
            
        t_rerank = time.time()
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        print(f"[retriever] Reranking took {time.time() - t_rerank:.2f}s for {len(pairs)} pairs.")
        
        legal_bonus = {
            "Luật": 0.15,
            "Nghị định": 0.10,
            "Thông tư": 0.00
        }
        
        for r, score in zip(results, scores):
            final_score = float(score)
            bonus = legal_bonus.get(r.legal_type, 0.0)
            r.score = final_score + bonus
            r.source = f"rerank({r.source})"
            
        results.sort(key=lambda x: x.score, reverse=True)
        
        k = top_k or self.top_k
        return results[:k]

    def retrieve(
        self,
        query: str,
        mode: Literal["vector", "fts", "hybrid"] = "hybrid",
        top_k: Optional[int] = None,
        rerank: bool = False,
    ) -> List[RetrievalResult]:
        """
        Truy xuất kết quả theo mode.

        Args:
            query : câu hỏi pháp lý tiếng Việt
            mode  : "vector" | "fts" | "hybrid" (mặc định hybrid)
            top_k : số kết quả (override top_k của instance)
            rerank: bật reranking bằng PhoRanker
        """
        t0 = time.time()
        k = top_k or self.top_k
        
        if rerank:
            fetch_k = 20
        else:
            fetch_k = max(30, k * 4)
        
        if mode == "vector":
            results = self.vector_search(query, top_k=fetch_k)
        elif mode == "fts":
            results = self.fts_search(query, top_k=fetch_k)
        else:
            results = self.hybrid_search(query, top_k=fetch_k)

        # Lọc trùng lặp
        unique_results = []
        for r in results:
            is_dup = False
            for ur in unique_results:
                if self.are_chunks_duplicate(r, ur):
                    is_dup = True
                    break
            if not is_dup:
                unique_results.append(r)
                
        if rerank:
            results = self.rerank(query, unique_results, top_k=k)
        else:
            results = unique_results[:k]

        elapsed = time.time() - t0
        print(f"[retriever] mode={mode} | rerank={rerank} | {len(results)} kết quả | {elapsed:.2f}s")
        return results

    # ── Dedup + aggregate theo document ─────────────────────────────────────────

    @staticmethod
    def aggregate_by_document(results: List[RetrievalResult]) -> dict:
        """
        Nhóm kết quả theo document, trả về dict:
          { document_id: { "meta": {...}, "chunks": [...], "max_score": float } }
        """
        docs: dict = {}
        for r in results:
            if r.document_id not in docs:
                docs[r.document_id] = {
                    "doc_number": r.doc_number,
                    "title": r.title,
                    "legal_type": r.legal_type,
                    "chunks": [],
                    "max_score": 0.0,
                    "articles": set(),
                }
            docs[r.document_id]["chunks"].append(r)
            docs[r.document_id]["max_score"] = max(
                docs[r.document_id]["max_score"], r.score
            )
            if r.article_hint:
                docs[r.document_id]["articles"].add(r.article_hint)

        # Convert articles set → sorted list
        for doc in docs.values():
            doc["articles"] = sorted(doc["articles"], key=lambda x: int(re.search(r'\d+', x).group()))
        return docs