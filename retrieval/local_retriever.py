"""
local_retriever.py
==================
Retriever chạy hoàn toàn LOCAL trên SQLite (local_chunks.db).
Không cần kết nối Supabase/internet.

Hỗ trợ:
  - FTS: SQLite FTS5 BM25 native (ultra-fast)
  - Vector: cosine similarity trên BLOB embeddings
  - Hybrid: RRF kết hợp FTS + vector

Dùng để test offline khi Supabase không accessible.
"""

import os
import sys
import re
import io
import time
import sqlite3
import numpy as np
from typing import List, Optional, Literal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

# ─── Reuse RetrievalResult từ retriever.py ────────────────────────────────────
from retrieval.retriever import RetrievalResult, extract_article_hint


# ─── Parse metadata từ chunk content header ───────────────────────────────────
# Format: "Văn bản: Nghị định 39/2018/NĐ-CP về hướng dẫn ... (39/2018/NĐ-CP)"
_CONTENT_HEADER_RE = re.compile(
    r'V\u0103n b\u1ea3n:\s*(.+?)\s*\(([^)]+)\)',
    re.MULTILINE | re.UNICODE
)

def _parse_meta_from_content(content: str) -> dict:
    """Parse document metadata từ dòng header trong chunk content."""
    m = _CONTENT_HEADER_RE.search(content[:400])
    if not m:
        return {"document_number": "", "title": "", "legal_type": ""}

    full_title = m.group(1).strip()
    doc_number = m.group(2).strip()

    # Tách legal_type (loại văn bản)
    legal_type_match = re.match(
        r'^(Lu\u1eadt|B\u1ed9 lu\u1eadt|Ngh\u1ecb \u0111\u1ecbnh|Th\u00f4ng t\u01b0|'
        r'Quy\u1ebft \u0111\u1ecbnh|Ngh\u1ecb quy\u1ebft|Ph\u00e1p l\u1ec7nh|'
        r'C\u00f4ng v\u0103n|Th\u00f4ng b\u00e1o|Ch\u1ec9 th\u1ecb)\b',
        full_title, re.UNICODE
    )
    legal_type = legal_type_match.group(1) if legal_type_match else ""

    # Title = xóa "<legal_type> <doc_number>" ở đầu
    title = full_title
    if legal_type and doc_number:
        title = re.sub(
            rf'^{re.escape(legal_type)}\s+{re.escape(doc_number)}\s*',
            '', title
        ).strip()

    return {
        "document_number": doc_number,
        "title": title,
        "legal_type": legal_type,
    }


def _tokenize(text: str) -> List[str]:
    """Tokenize đơn giản cho fallback search."""
    tokens = re.split(r'[\s\.,;:!\?\"\'()\[\]{}\-/\\]+', text.lower())
    return [t for t in tokens if len(t) >= 2]


class LocalRetriever:
    """
    Retriever hoàn toàn local, không cần internet.

    Args:
        db_path      : đường dẫn SQLite
        top_k        : số kết quả
        vector_weight: trọng số vector trong RRF hybrid
        fts_weight   : trọng số FTS trong RRF hybrid
        rrf_k        : hằng số RRF
    """

    def __init__(
        self,
        db_path: str = LOCAL_DB_PATH,
        top_k: int = 10,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.db_path = db_path
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight
        self.rrf_k = rrf_k
        self._conn: Optional[sqlite3.Connection] = None
        self._model = None

    # ── Kết nối ─────────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Embedding model ─────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("[local] Loading bkai-bi-encoder...", flush=True)
            t0 = time.time()
            self._model = SentenceTransformer(
                "bkai-foundation-models/vietnamese-bi-encoder", device="cpu"
            )
            print(f"[local] Model loaded in {time.time()-t0:.1f}s", flush=True)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([query], show_progress_bar=False)[0]
        return vec.astype(np.float32)

    # ── FTS5 check ──────────────────────────────────────────────────────────────

    def _has_fts5_index(self) -> bool:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts5';"
        )
        result = cur.fetchone()
        if result is None:
            return False
        # Verify schema is correct (no content_rowid='id' bug)
        cur.execute("SELECT sql FROM sqlite_master WHERE name='chunks_fts5';")
        sql = cur.fetchone()
        if sql:
            fts_sql = sql[0] or ""
            # Bad schema: content_rowid='id' where id is TEXT
            if "content_rowid='id'" in fts_sql:
                return False
        return True

    # ── FTS Search ──────────────────────────────────────────────────────────────

    def fts_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        FTS search với SQLite FTS5 BM25 native.
        Fallback sang LIKE search nếu FTS5 index chưa được tạo đúng.

        QUAN TRỌNG: FTS5 sử dụng unicode61 tokenizer giữ nguyên dấu tiếng Việt.
        Query cần nhập có dấu đầy đủ để match tốt nhất.
        """
        k = top_k or self.top_k
        conn = self._get_conn()
        cur = conn.cursor()

        if self._has_fts5_index():
            try:
                # FTS5 với BM25 ranking
                # Sanitize query: loại bỏ ký tự đặc biệt không hợp lệ với FTS5 MATCH
                fts_query = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', query)
                fts_query = re.sub(r'\s+', ' ', fts_query).strip()
                if not fts_query:
                    raise ValueError("Empty FTS query after sanitization")

                cur.execute("""
                    SELECT
                        dc.id, dc.document_id, dc.chunk_index, dc.content,
                        -bm25(chunks_fts5) AS score
                    FROM chunks_fts5
                    JOIN document_chunks dc ON dc.rowid = chunks_fts5.rowid
                    WHERE chunks_fts5 MATCH ?
                    ORDER BY bm25(chunks_fts5)
                    LIMIT ?;
                """, (fts_query, k))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    cid = row[0]
                    doc_id = row[1]
                    chunk_idx = row[2]
                    content = row[3] or ""
                    score = float(row[4]) if row[4] else 0.0
                    meta = _parse_meta_from_content(content)
                    results.append(RetrievalResult(
                        chunk_id=str(cid),
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=content,
                        doc_number=meta["document_number"],
                        title=meta["title"],
                        legal_type=meta["legal_type"],
                        score=score,
                        source="fts",
                        article_hint=extract_article_hint(content),
                    ))
                if results:
                    return results
                # 0 results: try LIKE fallback (query may be missing diacritics)
                print("[local][FTS5] 0 results from FTS5 (possibly missing diacritics). Trying LIKE fallback...")
            except Exception as e:
                print(f"[local][FTS5] Error: {e}, falling back to LIKE search")

        # LIKE fallback search (slower, for queries without diacritics)
        print("[local][FTS] Using LIKE fallback search...")

        keywords = [w for w in query.split() if len(w) >= 3][:5]
        if not keywords:
            return []

        like_pattern = f"%{keywords[0]}%"
        cur.execute("""
            SELECT id, document_id, chunk_index, content
            FROM document_chunks
            WHERE content LIKE ?
            LIMIT ?;
        """, (like_pattern, k * 3))
        rows = cur.fetchall()

        results = []
        for row in rows:
            cid, doc_id, chunk_idx, content = row[0], row[1], row[2], row[3] or ""
            score = float(sum(1 for kw in keywords if kw in content))
            if score == 0:
                continue
            meta = _parse_meta_from_content(content)
            results.append(RetrievalResult(
                chunk_id=str(cid),
                document_id=doc_id,
                chunk_index=chunk_idx,
                content=content,
                doc_number=meta["document_number"],
                title=meta["title"],
                legal_type=meta["legal_type"],
                score=score,
                source="fts",
                article_hint=extract_article_hint(content),
            ))
        results.sort(key=lambda x: -x.score)
        return results[:k]

    # ── Vector Search ────────────────────────────────────────────────────────────

    def vector_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Cosine similarity search trên embeddings trong SQLite (scan toàn bộ)."""
        k = top_k or self.top_k
        query_vec = self.embed_query(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, document_id, chunk_index, content, embedding "
            "FROM document_chunks WHERE embedding IS NOT NULL;"
        )

        scores = []
        for row in cur:
            emb_bytes = row["embedding"]
            if not emb_bytes:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            score = float(np.dot(query_vec, emb) / (query_norm * norm))
            scores.append((row["id"], row["document_id"], row["chunk_index"],
                           row["content"], score))

        scores.sort(key=lambda x: -x[4])
        results = []
        for cid, doc_id, chunk_idx, content, score in scores[:k]:
            meta = _parse_meta_from_content(content or "")
            results.append(RetrievalResult(
                chunk_id=str(cid),
                document_id=doc_id,
                chunk_index=chunk_idx,
                content=content or "",
                doc_number=meta["document_number"],
                title=meta["title"],
                legal_type=meta["legal_type"],
                score=score,
                source="vector",
                article_hint=extract_article_hint(content or ""),
            ))
        return results

    # ── Hybrid (RRF) ────────────────────────────────────────────────────────────

    def hybrid_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Kết hợp FTS + vector bằng Reciprocal Rank Fusion."""
        k = top_k or self.top_k
        fetch_k = k * 3

        fts_results    = self.fts_search(query,    top_k=fetch_k)
        vector_results = self.vector_search(query, top_k=fetch_k)

        chunk_map: dict = {}
        rrf_scores: dict = {}

        def add_ranked(results: List[RetrievalResult], weight: float):
            for rank, r in enumerate(results, start=1):
                cid = r.chunk_id
                if cid not in chunk_map:
                    chunk_map[cid] = r
                    rrf_scores[cid] = 0.0
                rrf_scores[cid] += weight / (rank + self.rrf_k)

        add_ranked(fts_results,    self.fts_weight)
        add_ranked(vector_results, self.vector_weight)

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

    # ── Entry point ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        mode: Literal["vector", "fts", "hybrid"] = "fts",
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        t0 = time.time()
        if mode == "vector":
            results = self.vector_search(query, top_k=top_k)
        elif mode == "fts":
            results = self.fts_search(query, top_k=top_k)
        else:
            results = self.hybrid_search(query, top_k=top_k)
        elapsed = time.time() - t0
        print(f"[local] mode={mode} | {len(results)} results | {elapsed:.2f}s")
        return results

    @staticmethod
    def aggregate_by_document(results: List[RetrievalResult]) -> dict:
        """Nhóm kết quả theo document_id."""
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
        for doc in docs.values():
            try:
                doc["articles"] = sorted(
                    doc["articles"],
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
            except Exception:
                doc["articles"] = sorted(doc["articles"])
        return docs


# ─── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    query = sys.argv[1] if len(sys.argv) > 1 else "doanh nghiệp nhỏ và vừa"
    mode  = sys.argv[2] if len(sys.argv) > 2 else "fts"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"Query : {query}")
    print(f"Mode  : {mode}")
    print(f"Top-K : {top_k}")
    print("-" * 60)

    r = LocalRetriever(top_k=top_k)
    results = r.retrieve(query, mode=mode, top_k=top_k)
    r.close()

    for i, res in enumerate(results, 1):
        print(f"\n[{i}] {res.doc_number} | {res.legal_type}")
        print(f"    Tieu de : {res.title[:70]}")
        print(f"    Chunk#{res.chunk_index} | {res.article_hint or '-'} | score={res.score:.4f}")
        print(f"    Content : {res.content[:200]}...")
