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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

# ─── Reuse RetrievalResult từ retriever.py ────────────────────────────────────
from dataclasses import dataclass, field

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
    source: str
    article_hint: Optional[str] = None
    best_chunk_content: Optional[str] = None

    def format_relevant_doc(self) -> str:
        return f"{self.doc_number}|{self.legal_type} {self.doc_number} {self.title}"

    def format_relevant_article(self) -> str:
        doc_str = self.format_relevant_doc()
        article = self.article_hint or "Toàn bộ"
        return f"{doc_str}|{article}"

_ARTICLE_PATTERN = re.compile(
    r"(?:^|\s)(Điều\s+\d+[a-zA-Z]?)",
    re.MULTILINE | re.UNICODE
)

def extract_article_hint(content: str) -> Optional[str]:
    m = _ARTICLE_PATTERN.search(content)
    if m:
        return m.group(1).strip()
    return None



# ─── Parse metadata từ chunk content header ───────────────────────────────────
# Format: "Văn bản: Nghị định 39/2018/NĐ-CP về hướng dẫn ... (39/2018/NĐ-CP)"
_CONTENT_HEADER_RE = re.compile(
    r'V\u0103n b\u1ea3n:\s*(.+?)\s*\(([^)]+)\)',
    re.MULTILINE | re.UNICODE | re.DOTALL
)

def _parse_meta_from_content(content: str) -> dict:
    """Parse document metadata từ dòng header trong chunk content."""
    m = _CONTENT_HEADER_RE.search(content[:400])
    if not m:
        return {"document_number": "", "title": "", "legal_type": ""}

    full_title = m.group(1).strip()
    full_title = re.sub(r'\s+', ' ', full_title)
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


class LegalRetriever:
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
        vector_weight: float = 0.3,
        fts_weight: float = 0.7,
        rrf_k: int = 60,
        use_postgres: bool = False,
        pg_conn_str: Optional[str] = None
    ):
        self.db_path = db_path
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight
        self.rrf_k = rrf_k
        self.use_postgres = use_postgres
        
        if pg_conn_str is None:
            import os
            try:
                from dotenv import load_dotenv
                # Load .env from project root if it exists
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                env_path = os.path.join(project_root, '.env')
                if os.path.exists(env_path):
                    load_dotenv(dotenv_path=env_path)
                else:
                    load_dotenv()
            except ImportError:
                pass
            pg_conn_str = os.getenv("DATABASE_URL")
            if not pg_conn_str:
                pg_conn_str = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"
                
        self.pg_conn_str = pg_conn_str
        self._conn: Optional[sqlite3.Connection] = None
        self._pg_conn = None
        self._model = None
        self._reranker = None

    # ── Kết nối ─────────────────────────────────────────────────────────────────

    def _get_pg_conn(self):
        if self._pg_conn is None:
            import psycopg2
            try:
                self._pg_conn = psycopg2.connect(self.pg_conn_str)
            except psycopg2.OperationalError as e:
                local_fallback = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"
                if self.pg_conn_str != local_fallback:
                    print(f"[local] Failed to connect to pg_conn_str ({self.pg_conn_str}): {e}. Falling back to localhost PostgreSQL...", flush=True)
                    try:
                        self._pg_conn = psycopg2.connect(local_fallback)
                        self.pg_conn_str = local_fallback
                        return self._pg_conn
                    except Exception as fallback_err:
                        print(f"[local] Fallback to localhost PostgreSQL also failed: {fallback_err}", flush=True)
                raise e
        return self._pg_conn

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            
            # SQLite performance tuning
            try:
                cur = self._conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=OFF;")
                cur.execute("PRAGMA cache_size=-2000000;") # 2GB cache
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA mmap_size=3000000000;") # 3GB memory map
            except Exception as e:
                print(f"[local] Warning: Failed to apply SQLite PRAGMAs: {e}")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._pg_conn:
            self._pg_conn.close()
            self._pg_conn = None

    # ── Embedding model ─────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[local] Loading bkai-bi-encoder on device '{device}'...", flush=True)
            t0 = time.time()
            self._model = SentenceTransformer(
                "bkai-foundation-models/vietnamese-bi-encoder", device=device
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

    def fts_search_pg(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        def sqlite_phrase_to_pg_tsquery(phrase: str) -> str:
            phrase = phrase.strip('"')
            words = [w.strip() for w in phrase.split() if w.strip()]
            if len(words) == 1:
                return f"'{words[0]}'"
            return "(" + " <-> ".join(f"'{w}'" for w in words) + ")"

        k = top_k or self.top_k
        conn = self._get_pg_conn()
        cur = conn.cursor()
        results = []

        # 1. Try structured AND/OR query
        q_lower = query.lower()
        subjects = {
            "incubator_coworking": {
                "keywords": ["ươm tạo", "làm việc chung"],
                "syns": ['"ươm tạo"', '"cơ sở ươm tạo"', '"khu làm việc chung"', '"làm việc chung"']
            },
            "sme": {
                "keywords": ["doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "sme", "siêu nhỏ"],
                "syns": ['"doanh nghiệp nhỏ và vừa"', '"nhỏ và vừa"', '"doanh nghiệp nhỏ"', '"doanh nghiệp siêu nhỏ"']
            },
            "labor": {
                "keywords": ["nhân viên", "lao động", "người lao động", "thử việc", "ký hợp đồng", "sa thải", "đuổi việc", "nghỉ việc", "lương"],
                "syns": ['"người lao động"', '"lao động"', '"nhân viên"', '"hợp đồng lao động"']
            }
        }
        
        topics = {
            "tax_land": {
                "keywords": ["thuế", "đất đai", "mặt bằng", "tiền thuê"],
                "syns": ['"thuế"', '"đất đai"', '"mặt bằng"', '"thuê đất"', '"ưu đãi thuế"', '"tiền sử dụng đất"']
            },
            "bidding": {
                "keywords": ["đấu thầu", "nhà thầu", "gói thầu"],
                "syns": ['"đấu thầu"', '"nhà thầu"', '"lựa chọn nhà thầu"', '"gói thầu"', '"xếp hạng hồ sơ"']
            },
            "degrees_holding": {
                "keywords": ["giữ", "bằng", "văn bằng", "chứng chỉ", "giấy tờ", "bằng cấp"],
                "syns": ['"giữ"', '"thu giữ"', '"tạm giữ"', '"văn bằng"', '"chứng chỉ"', '"giấy tờ tùy thân"', '"bản chính"', '"bằng cấp"']
            }
        }

        active_subjects = []
        for s_name, s_info in subjects.items():
            if any(kw in q_lower for kw in s_info["keywords"]):
                active_subjects.append(s_info["syns"])
                
        active_topics = []
        for t_name, t_info in topics.items():
            if any(kw in q_lower for kw in t_info["keywords"]):
                active_topics.append(t_info["syns"])
                
        clauses = []
        if active_subjects:
            flat_subjects = []
            for s in active_subjects:
                flat_subjects.extend(s)
            flat_subjects = list(dict.fromkeys(flat_subjects))
            pg_terms = [sqlite_phrase_to_pg_tsquery(t) for t in flat_subjects]
            clauses.append("(" + " | ".join(pg_terms) + ")")
            
        if active_topics:
            flat_topics = []
            for t in active_topics:
                flat_topics.extend(t)
            flat_topics = list(dict.fromkeys(flat_topics))
            pg_terms = [sqlite_phrase_to_pg_tsquery(t) for t in flat_topics]
            clauses.append("(" + " | ".join(pg_terms) + ")")

        if clauses:
            fts_expr = " & ".join(clauses)
            print(f"[local][FTS-PG] Structured query: {fts_expr}", flush=True)
            try:
                cur.execute("""
                    SELECT c.id, c.document_id, c.chunk_index, c.content, 
                           ts_rank(to_tsvector('simple', c.content), to_tsquery('simple', %s)) AS score,
                           d.document_number, d.title, d.legal_type
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s;
                """, (fts_expr, fts_expr, k))
                rows = cur.fetchall()
                for row in rows:
                    cid, doc_id, chunk_idx, content, score, doc_number, title, legal_type = row
                    results.append(RetrievalResult(
                        chunk_id=str(cid),
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=content,
                        doc_number=doc_number,
                        title=title,
                        legal_type=legal_type,
                        score=float(score) if score is not None else 0.0,
                        source="fts",
                        article_hint=extract_article_hint(content),
                    ))
            except Exception as e:
                print(f"[local][FTS-PG] Structured query failed: {e}", flush=True)
                try:
                    conn.rollback()
                except Exception:
                    pass
                cur = conn.cursor()

        # 2. Try fallback OR query
        if not results:
            GRAMMAR_STOPWORDS = {
                "và", "hoặc", "nhưng", "vì", "nên", "của", "các", "những", "là", "có", 
                "trong", "tại", "để", "theo", "được", "bị", "cho", "ra", "vào", "lên", 
                "xuống", "do", "từ", "đến", "bằng", "with", "với", "về", "như", "thì", 
                "mà", "khi", "gì", "này", "đó", "kia", "nọ", "thế", "vậy", "nào", "ai", 
                "đâu", "cái", "con", "chiếc", "nếu", "sẽ", "phải", "sao", "thế"
            }
            words = [w.lower().strip() for w in query.split()]
            terms = []
            for w in words:
                w = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', w).strip()
                if w and len(w) >= 3 and w not in GRAMMAR_STOPWORDS:
                    terms.append(f"'{w}'")
            
            if terms:
                fallback_expr = " | ".join(terms)
                print(f"[local][FTS-PG] Fallback OR query: {fallback_expr}", flush=True)
                try:
                    cur.execute("""
                        SELECT c.id, c.document_id, c.chunk_index, c.content, 
                               ts_rank(to_tsvector('simple', c.content), to_tsquery('simple', %s)) AS score,
                               d.document_number, d.title, d.legal_type
                        FROM document_chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
                        ORDER BY score DESC
                        LIMIT %s;
                    """, (fallback_expr, fallback_expr, k))
                    rows = cur.fetchall()
                    for row in rows:
                        cid, doc_id, chunk_idx, content, score, doc_number, title, legal_type = row
                        results.append(RetrievalResult(
                            chunk_id=str(cid),
                            document_id=doc_id,
                            chunk_index=chunk_idx,
                            content=content,
                            doc_number=doc_number,
                            title=title,
                            legal_type=legal_type,
                            score=float(score) if score is not None else 0.0,
                            source="fts",
                            article_hint=extract_article_hint(content),
                        ))
                except Exception as e:
                    print(f"[local][FTS-PG] Fallback OR query failed: {e}", flush=True)
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    cur = conn.cursor()

        # 3. Try ILIKE fallback search
        if not results:
            print(f"[local][FTS-PG] FTS queries returned 0 results. Trying ILIKE fallback search...", flush=True)
            keywords = [w for w in query.split() if len(w) >= 3][:5]
            if keywords:
                like_pattern = f"%{keywords[0]}%"
                try:
                    cur.execute("""
                        SELECT c.id, c.document_id, c.chunk_index, c.content,
                               d.document_number, d.title, d.legal_type
                        FROM document_chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.content ILIKE %s
                        LIMIT %s;
                    """, (like_pattern, k * 3))
                    rows = cur.fetchall()
                    for row in rows:
                        cid, doc_id, chunk_idx, content, doc_number, title, legal_type = row
                        score = float(sum(1 for kw in keywords if kw.lower() in content.lower()))
                        results.append(RetrievalResult(
                            chunk_id=str(cid),
                            document_id=doc_id,
                            chunk_index=chunk_idx,
                            content=content,
                            doc_number=doc_number,
                            title=title,
                            legal_type=legal_type,
                            score=score,
                            source="fts",
                            article_hint=extract_article_hint(content),
                        ))
                    results.sort(key=lambda x: -x.score)
                    results = results[:k]
                except Exception as e:
                    print(f"[local][FTS-PG] ILIKE fallback failed: {e}", flush=True)

        cur.close()
        return results

    def fts_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        FTS search với SQLite FTS5 BM25 native hoặc PostgreSQL.
        Sử dụng cấu trúc AND/OR thông minh dựa trên phân tích ngữ nghĩa để tránh gây nhiễu.
        Fallback sang LIKE search nếu FTS5 index chưa được tạo đúng.
        """
        if self.use_postgres:
            return self.fts_search_pg(query, top_k)

        k = top_k or self.top_k
        
        conn = self._get_conn()
        cur = conn.cursor()

        # Helper function to execute two-step FTS search to avoid SQLite slow JOIN behavior
        def _execute_fts_query(match_expr: str, limit: int) -> List[RetrievalResult]:
            try:
                cur.execute("""
                    SELECT rowid, -bm25(chunks_fts5) AS score
                    FROM chunks_fts5
                    WHERE chunks_fts5 MATCH ?
                    ORDER BY bm25(chunks_fts5)
                    LIMIT ?;
                """, (match_expr, limit))
                rows = cur.fetchall()
                if not rows:
                    return []
                    
                rowids = [r[0] for r in rows]
                scores = {r[0]: float(r[1]) if r[1] is not None else 0.0 for r in rows}
                
                placeholders = ",".join("?" for _ in rowids)
                cur.execute(f"""
                    SELECT rowid, id, document_id, chunk_index, content
                    FROM document_chunks
                    WHERE rowid IN ({placeholders})
                """, rowids)
                details = cur.fetchall()
                
                results_list = []
                for row in details:
                    r_id, cid, doc_id, chunk_idx, content = row
                    score = scores.get(r_id, 0.0)
                    meta = _parse_meta_from_content(content or "")
                    results_list.append(RetrievalResult(
                        chunk_id=str(cid),
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=content or "",
                        doc_number=meta["document_number"],
                        title=meta["title"],
                        legal_type=meta["legal_type"],
                        score=score,
                        source="fts",
                        article_hint=extract_article_hint(content or ""),
                    ))
                results_list.sort(key=lambda x: x.score, reverse=True)
                return results_list
            except Exception as e:
                print(f"[local][FTS5] MATCH query error: {e}")
                return []

        if self._has_fts5_index():
            # Xây dựng câu truy vấn FTS cấu trúc AND/OR
            q_lower = query.lower()
            
            # Phân nhóm thực thể (Subjects) và Chủ đề/Hành động (Topics/Actions)
            subjects = {
                "incubator_coworking": {
                    "keywords": ["ươm tạo", "làm việc chung"],
                    "syns": ['"ươm tạo"', '"cơ sở ươm tạo"', '"khu làm việc chung"', '"làm việc chung"']
                },
                "sme": {
                    "keywords": ["doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "sme", "siêu nhỏ"],
                    "syns": ['"doanh nghiệp nhỏ và vừa"', '"nhỏ và vừa"', '"doanh nghiệp nhỏ"', '"doanh nghiệp siêu nhỏ"']
                },
                "labor": {
                    "keywords": ["nhân viên", "lao động", "người lao động", "thử việc", "ký hợp đồng", "sa thải", "đuổi việc", "nghỉ việc", "lương"],
                    "syns": ['"người lao động"', '"lao động"', '"nhân viên"', '"hợp đồng lao động"']
                }
            }
            
            topics = {
                "tax_land": {
                    "keywords": ["thuế", "đất đai", "mặt bằng", "tiền thuê"],
                    "syns": ['"thuế"', '"đất đai"', '"mặt bằng"', '"thuê đất"', '"ưu đãi thuế"', '"tiền sử dụng đất"']
                },
                "bidding": {
                    "keywords": ["đấu thầu", "nhà thầu", "gói thầu"],
                    "syns": ['"đấu thầu"', '"nhà thầu"', '"lựa chọn nhà thầu"', '"gói thầu"', '"xếp hạng hồ sơ"']
                },
                "degrees_holding": {
                    "keywords": ["giữ", "bằng", "văn bằng", "chứng chỉ", "giấy tờ", "bằng cấp"],
                    "syns": ['"giữ"', '"thu giữ"', '"tạm giữ"', '"văn bằng"', '"chứng chỉ"', '"giấy tờ tùy thân"', '"bản chính"', '"bằng cấp"']
                }
            }

            active_subjects = []
            for s_name, s_info in subjects.items():
                if any(kw in q_lower for kw in s_info["keywords"]):
                    active_subjects.append(s_info["syns"])
                    
            active_topics = []
            for t_name, t_info in topics.items():
                if any(kw in q_lower for kw in t_info["keywords"]):
                    active_topics.append(t_info["syns"])
                    
            clauses = []
            if active_subjects:
                flat_subjects = []
                for s in active_subjects:
                    flat_subjects.extend(s)
                flat_subjects = list(dict.fromkeys(flat_subjects))
                clauses.append("(" + " OR ".join(flat_subjects) + ")")
                
            if active_topics:
                flat_topics = []
                for t in active_topics:
                    flat_topics.extend(t)
                flat_topics = list(dict.fromkeys(flat_topics))
                clauses.append("(" + " OR ".join(flat_topics) + ")")

            fts_expr = ""
            if clauses:
                fts_expr = " AND ".join(clauses)
                print(f"[local][FTS5] Structured query: {fts_expr}")
                results = _execute_fts_query(fts_expr, k)
                if results:
                    return results
                print(f"[local][FTS5] Structured query returned 0 results. Trying fallback OR...")

            # Fallback sang OR đơn giản của các từ khóa có nghĩa (loại bỏ stopword)
            GRAMMAR_STOPWORDS = {
                "và", "hoặc", "nhưng", "vì", "nên", "của", "các", "những", "là", "có", 
                "trong", "tại", "để", "theo", "được", "bị", "cho", "ra", "vào", "lên", 
                "xuống", "do", "từ", "đến", "bằng", "with", "với", "về", "như", "thì", 
                "mà", "khi", "gì", "này", "đó", "kia", "nọ", "thế", "vậy", "nào", "ai", 
                "đâu", "cái", "con", "chiếc", "nếu", "sẽ", "phải", "sao", "thế"
            }
            words = [w.lower().strip() for w in query.split()]
            terms = []
            for w in words:
                w = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', w).strip()
                if w and len(w) >= 3 and w not in GRAMMAR_STOPWORDS:
                    terms.append(f'"{w}"')
            
            if terms:
                fallback_expr = " OR ".join(terms)
                print(f"[local][FTS5] Fallback OR query: {fallback_expr}")
                results = _execute_fts_query(fallback_expr, k)
                if results:
                    return results

            print("[local][FTS5] 0 results from FTS5. Trying LIKE fallback...")

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

    # ── FAISS Vector Search ───────────────────────────────────────────────────────

    def _get_faiss_index(self):
        """Lấy hoặc tự động build FAISS index từ SQLite database."""
        import faiss
        index_path = os.path.join(os.path.dirname(self.db_path), "local_chunks.index")
        
        if not hasattr(self, "_faiss_index") or self._faiss_index is None:
            if os.path.exists(index_path):
                print(f"[local] Loading FAISS index from {index_path}...", flush=True)
                t0 = time.time()
                self._faiss_index = faiss.read_index(index_path)
                print(f"[local] FAISS index loaded in {time.time()-t0:.2f}s", flush=True)
            else:
                print(f"[local] FAISS index not found. Building FAISS index from {self.db_path} (one-time setup)...", flush=True)
                t0 = time.time()
                conn = self._get_conn()
                cur = conn.cursor()
                cur.execute("SELECT rowid, embedding FROM document_chunks WHERE embedding IS NOT NULL;")
                rowids = []
                embeddings = []
                for rowid, emb_bytes in cur:
                    if not emb_bytes:
                        continue
                    emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    rowids.append(rowid)
                    embeddings.append(emb)
                
                if not embeddings:
                    raise ValueError("No embeddings found in database to build FAISS index!")
                
                embeddings = np.array(embeddings, dtype=np.float32)
                rowids = np.array(rowids, dtype=np.int64)
                
                # Chuẩn hóa vector L2 để tìm kiếm Cosine Similarity bằng Inner Product
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                
                dim = embeddings.shape[1]
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIDMap(quantizer)
                index.add_with_ids(embeddings, rowids)
                
                # Lưu index để các lần sau dùng ngay
                faiss.write_index(index, index_path)
                self._faiss_index = index
                print(f"[local] FAISS index built and saved in {time.time()-t0:.2f}s", flush=True)
                
        return self._faiss_index

    def vector_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Tìm kiếm tương đồng vector bằng FAISS index (tải và tìm cực nhanh)."""
        k = top_k or self.top_k
        
        query_vec = self.embed_query(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
            
        # Chuẩn hóa query vector và định dạng cho FAISS
        query_vec = query_vec / query_norm
        query_vec = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        
        index = self._get_faiss_index()
        scores, indices = index.search(query_vec, k)
        scores = scores[0]
        indices = indices[0]
        
        valid_indices = [int(idx) for idx in indices if idx != -1]
        if not valid_indices:
            return []
            
        if self.use_postgres:
            # 1. Truy vấn SQLite để lấy (document_id, chunk_index) tương ứng với SQLite rowid
            conn_sq = self._get_conn()
            cur_sq = conn_sq.cursor()
            placeholders_sq = ",".join("?" for _ in valid_indices)
            cur_sq.execute(f"SELECT rowid, document_id, chunk_index FROM document_chunks WHERE rowid IN ({placeholders_sq});", valid_indices)
            rowid_to_key = {row[0]: (row[1], row[2]) for row in cur_sq.fetchall()}
            cur_sq.close()
            
            keys = [rowid_to_key[idx] for idx in valid_indices if idx in rowid_to_key]
            if not keys:
                return []
                
            # 2. Truy vấn PostgreSQL bằng (document_id, chunk_index) để lấy thông tin chi tiết
            conn_pg = self._get_pg_conn()
            cur_pg = conn_pg.cursor()
            placeholders_pg = ",".join("(%s, %s)" for _ in keys)
            params = []
            for doc_id, chunk_idx in keys:
                params.extend([doc_id, chunk_idx])
                
            cur_pg.execute(f"""
                SELECT c.id, c.document_id, c.chunk_index, c.content,
                       d.document_number, d.title, d.legal_type
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE (c.document_id, c.chunk_index) IN ({placeholders_pg});
            """, params)
            
            rows = cur_pg.fetchall()
            cur_pg.close()
            key_to_row = {(row[1], row[2]): row for row in rows}
            
            results = []
            for idx, score in zip(indices, scores):
                idx = int(idx)
                if idx not in rowid_to_key:
                    continue
                key = rowid_to_key[idx]
                if key not in key_to_row:
                    continue
                row = key_to_row[key]
                cid = row[0]
                doc_id = row[1]
                chunk_idx = row[2]
                content = row[3] or ""
                doc_number = row[4]
                title = row[5]
                legal_type = row[6]
                
                results.append(RetrievalResult(
                    chunk_id=str(cid),
                    document_id=doc_id,
                    chunk_index=chunk_idx,
                    content=content,
                    doc_number=doc_number,
                    title=title,
                    legal_type=legal_type,
                    score=float(score),
                    source="vector",
                    article_hint=extract_article_hint(content),
                ))
            return results
        else:
            # Truy vấn SQLite để lấy thông tin các chunks tương ứng với rowid
            conn = self._get_conn()
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in valid_indices)
            cur.execute(f"""
                SELECT rowid, id, document_id, chunk_index, content
                FROM document_chunks
                WHERE rowid IN ({placeholders});
            """, valid_indices)
            
            rows = cur.fetchall()
            row_map = {row["rowid"]: row for row in rows}
            
            results = []
            for idx, score in zip(indices, scores):
                idx = int(idx)
                if idx not in row_map:
                    continue
                row = row_map[idx]
                cid = row["id"]
                doc_id = row["document_id"]
                chunk_idx = row["chunk_index"]
                content = row["content"] or ""
                
                meta = _parse_meta_from_content(content)
                results.append(RetrievalResult(
                    chunk_id=str(cid),
                    document_id=doc_id,
                    chunk_index=chunk_idx,
                    content=content,
                    doc_number=meta["document_number"],
                    title=meta["title"],
                    legal_type=meta["legal_type"],
                    score=float(score),
                    source="vector",
                    article_hint=extract_article_hint(content),
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

    def rerank(self, query: str, results: List[RetrievalResult], top_k: Optional[int] = None) -> List[RetrievalResult]:
        if not results:
            return []
            
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[local] Loading reranker model BAAI/bge-reranker-v2-m3 on device '{device}'...", flush=True)
            t0 = time.time()
            model_path = os.path.join(os.path.dirname(__file__), "bge-reranker-v2-m3")
            automodel_args = {}
            if device == "cuda":
                automodel_args = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True
                }
            self._reranker = CrossEncoder(model_path, device=device, automodel_args=automodel_args)
            print(f"[local] Reranker model loaded in {time.time()-t0:.1f}s", flush=True)
            
        pairs = []
        for r in results:
            best_text = getattr(r, 'best_chunk_content', None) or r.content
            
            # Thêm metadata boosting để tăng độ chính xác (rất hiệu quả với văn bản pháp luật)
            article_str = f"Điều: {r.article_hint}\n" if r.article_hint else ""
            doc_text = f"Văn bản: {r.legal_type} {r.doc_number} - {r.title}\n{article_str}Nội dung:\n{best_text}"
            
            pairs.append([query, doc_text])
            
        t_rerank = time.time()
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        print(f"[local] Reranking took {time.time() - t_rerank:.2f}s for {len(pairs)} pairs.")
        
        LEGAL_WEIGHT = {
            "Luật": 1.0,
            "Bộ luật": 1.0,
            "Nghị quyết": 0.95,
            "Nghị định": 0.85,
            "Thông tư": 0.75,
        }
        
        for r, score in zip(results, scores):
            rerank_score = float(score)
            # Normalize PhoRanker output to positive if it outputs logits
            if rerank_score < 0 or rerank_score > 1.0:
                import math
                rerank_score = 1 / (1 + math.exp(-max(min(rerank_score, 100), -100)))
                
            weight = LEGAL_WEIGHT.get(r.legal_type, 0.7)
            final_score = rerank_score * weight
            
            r.score = final_score
            r.source = f"rerank({r.source})"
            
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Lọc MMR Diversity: max 2 chunk mỗi document
        filtered_results = []
        doc_count = {}
        for r in results:
            doc_id = r.document_id
            if doc_count.get(doc_id, 0) < 2:
                filtered_results.append(r)
                doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
        
        if top_k:
            return filtered_results[:top_k]
        return filtered_results

    def are_chunks_duplicate(self, c1: RetrievalResult, c2: RetrievalResult) -> bool:
        """Kiểm tra xem hai chunk có trùng lặp nội dung đáng kể hay không."""
        if c1.document_id == c2.document_id and c1.chunk_index == c2.chunk_index:
            return True
            
        # Trích xuất phần nội dung thực tế để so sánh
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
        
        # Jaccard score cao => trùng lặp
        if jaccard > 0.8:
            return True
            
        # Nếu trùng Điều/Khoản luật và độ tương đồng tương đối cao => trùng lặp (ví dụ giữa Luật và VBHN)
        if c1.article_hint and c2.article_hint and c1.article_hint == c2.article_hint:
            if "điều" in c1.article_hint.lower() or "khoản" in c1.article_hint.lower():
                if jaccard > 0.6:
                    return True
        return False

    # ── Entry point ─────────────────────────────────────────────────────────────


    def expand_query(self, query: str) -> str:
        """Query Expansion bằng rule-based dictionary."""
        q_lower = query.lower()
        expanded_terms = []
        
        intents = {
            "hỗ trợ": ["ưu đãi", "chính sách hỗ trợ", "khuyến khích"],
            "ưu đãi": ["miễn", "giảm", "hỗ trợ"],
            "miễn": ["giảm", "không phải nộp", "ưu đãi"],
            "giảm": ["miễn", "ưu đãi thuế"],
            "đất đai": ["tiền thuê đất", "tiền sử dụng đất", "mặt bằng", "đất"],
            "thuế": ["thuế thu nhập doanh nghiệp", "thuế tndn", "miễn thuế", "giảm thuế"]
        }
        
        for k, v in intents.items():
            if k in q_lower:
                expanded_terms.extend(v)
                
        if expanded_terms:
            new_terms = [t for t in expanded_terms if t not in q_lower]
            if new_terms:
                return query + " " + " ".join(new_terms)
        return query

    def expand_to_parent_article(self, results: list) -> list:
        """Parent Retrieval: Lấy toàn bộ nội dung của Điều luật từ các chunk ban đầu."""
        if not results:
            return []
            
        target_articles = {}
        hit_indexes = {}
        for r in results:
            if not r.article_hint:
                continue
            doc_id = r.document_id
            if doc_id not in target_articles:
                target_articles[doc_id] = set()
            target_articles[doc_id].add(r.article_hint)
            
            # Ghi nhớ vị trí chunk_index đã hit
            key = (doc_id, r.article_hint)
            if key not in hit_indexes:
                hit_indexes[key] = set()
            hit_indexes[key].add(r.chunk_index)
            
        if not target_articles:
            return results
            
        doc_ids = list(target_articles.keys())
        if self.use_postgres:
            conn = self._get_pg_conn()
            cur = conn.cursor()
            placeholders = ",".join("%s" for _ in doc_ids)
            cur.execute(f"""
                SELECT document_id, chunk_index, content
                FROM document_chunks
                WHERE document_id IN ({placeholders})
                ORDER BY document_id, chunk_index
            """, doc_ids)
            all_chunks = cur.fetchall()
            cur.close()
            
            all_chunks_compat = []
            for row in all_chunks:
                all_chunks_compat.append({
                    "document_id": row[0],
                    "chunk_index": row[1],
                    "content": row[2]
                })
            all_chunks = all_chunks_compat
        else:
            conn = self._get_conn()
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in doc_ids)
            cur.execute(f"""
                SELECT document_id, chunk_index, content
                FROM document_chunks
                WHERE document_id IN ({placeholders})
                ORDER BY document_id, chunk_index
            """, doc_ids)
            all_chunks = cur.fetchall()
            cur.close()
            
        doc_chunks = {}
        for row in all_chunks:
            did = row["document_id"]
            if did not in doc_chunks:
                doc_chunks[did] = []
            doc_chunks[did].append(row)
            
        expanded_articles = {}
        
        for did, rows in doc_chunks.items():
            current_article = None
            for row in rows:
                content = row["content"] or ""
                # Rely on global extract_article_hint
                hint = extract_article_hint(content)
                if hint:
                    current_article = hint
                
                if current_article and current_article in target_articles[did]:
                    key = (did, current_article)
                    
                    # Windowing limit: Chỉ gộp nếu chunk nằm gần vị trí hit (+/- 2 chunks)
                    # Chống phình to context đối với các Điều omnibus (VD: "sửa đổi, bổ sung một số điều")
                    hits = hit_indexes.get(key, set())
                    is_near_hit = False
                    for h in hits:
                        if abs(h - row["chunk_index"]) <= 2:
                            is_near_hit = True
                            break
                            
                    if is_near_hit:
                        if key not in expanded_articles:
                            expanded_articles[key] = []
                        expanded_articles[key].append(content)
                    
        final_results = []
        seen_keys = set()
        
        for r in results:
            if not r.article_hint:
                final_results.append(r)
                continue
                
            key = (r.document_id, r.article_hint)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            if key in expanded_articles:
                # Lưu lại nội dung của best chunk (chunk ban đầu có điểm cao nhất) để dùng cho Rerank
                r.best_chunk_content = r.content
                
                full_content = "\n...\n".join(expanded_articles[key])
                r.content = full_content
                r.source += "_parent_expanded"
                final_results.append(r)
            else:
                final_results.append(r)
                
        return final_results

    def retrieve(
        self,
        query: str,
        mode: Literal["vector", "fts", "hybrid"] = "fts",
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        t0 = time.time()
        k = top_k or self.top_k
        
        # 1. Query Expansion (Intent Understanding)
        expanded_query = self.expand_query(query)
        
        # 2. Truy xuất danh sách ứng viên (candidate pool)
        if rerank:
            fetch_k = 50
        else:
            fetch_k = max(50, k * 4)
        
        if mode == "vector":
            results = self.vector_search(expanded_query, top_k=fetch_k)
        elif mode == "fts":
            results = self.fts_search(expanded_query, top_k=fetch_k)
        else:
            results = self.hybrid_search(expanded_query, top_k=fetch_k)
            
        # 3. Tiến hành lọc trùng lặp trước để loại bỏ nhiễu
        unique_results = []
        for r in results:
            is_dup = False
            for ur in unique_results:
                if self.are_chunks_duplicate(r, ur):
                     is_dup = True
                     break
            if not is_dup:
                unique_results.append(r)
                
        # 4. Parent Retrieval (Thay thế cho aggregate_chunks_into_articles)
        article_results = self.expand_to_parent_article(unique_results)

        # 5. Rerank và lọc
        if rerank:
            results = self.rerank(query, article_results, top_k=None)
        else:
            article_results.sort(key=lambda x: x.score, reverse=True)
            results = article_results

        # 4. Áp dụng Absolute & Relative Score Thresholds (Dynamic Thresholding)
        if results:
            RERANK_THRESHOLD = 0.15
            best_score = results[0].score
            
            relative_threshold_ratio = 0.5
            
            # Dynamic gap filtering: Nếu top 1 vượt trội hoàn toàn top 2
            if len(results) > 1:
                score_1 = results[0].score
                score_2 = results[1].score
                if (score_1 - score_2) > 0.2:
                    relative_threshold_ratio = 0.8
            
            filtered_by_score = []
            for r in results:
                # Giữ chunk nếu thỏa mãn ngưỡng tuyệt đối và ngưỡng tương đối động
                if r.score >= RERANK_THRESHOLD and r.score >= best_score * relative_threshold_ratio:
                    filtered_by_score.append(r)
            results = filtered_by_score

        # 5. Article-level Dedup (Giảm context thừa)
        seen_articles = set()
        deduped = []
        for r in results:
            article_id = r.format_relevant_article()
            if article_id in seen_articles:
                continue
            seen_articles.add(article_id)
            deduped.append(r)
        results = deduped

        # 6. Intent-aware (Query-aware) Filter
        try:
            import underthesea
            tags = underthesea.pos_tag(query)
            important_terms = [word.lower() for word, tag in tags if tag in ['N', 'Np', 'V', 'A', 'M'] and len(word) > 1]
            if not important_terms:
                important_terms = [w.lower() for w in query.split() if len(w) > 2]
        except Exception:
            important_terms = [w.lower() for w in query.split() if len(w) > 2]
            
        filtered_final = []
        for r in results:
            content_lower = r.content.lower()
            # Chunk phải chứa ít nhất một term quan trọng (intent keywords) từ câu hỏi
            if any(term in content_lower for term in important_terms):
                filtered_final.append(r)
                
        # Fallback nếu filter quá ngặt nghèo drop hết tất cả kết quả
        if not filtered_final and results:
            filtered_final = results
            
        results = filtered_final[:k]
            
        elapsed = time.time() - t0
        print(f"[local] mode={mode} | rerank={rerank} | {len(results)} results | {elapsed:.2f}s")
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

    r = LegalRetriever(top_k=top_k)
    results = r.retrieve(query, mode=mode, top_k=top_k)
    r.close()

    for i, res in enumerate(results, 1):
        print(f"\n[{i}] {res.doc_number} | {res.legal_type}")
        print(f"    Tieu de : {res.title[:70]}")
        print(f"    Chunk#{res.chunk_index} | {res.article_hint or '-'} | score={res.score:.4f}")
        print(f"    Content : {res.content[:200]}...")
