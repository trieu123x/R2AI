"""
batch_retrieve.py
=================
Chạy retrieval trên tập câu hỏi và xuất results.json đúng định dạng nộp bài.
Tích hợp local LLM (Qwen3-8B) để sinh câu trả lời cho trường 'answer'.

Input:  file JSON danh sách câu hỏi (format ban tổ chức cung cấp)
Output: results.json (file nộp bài thi bao gồm tài liệu và câu trả lời)

Cách dùng:
  python retrieval/batch_retrieve.py --input questions.json --local --rerank --llm
"""

import os
import sys
import io
import json
import time
import argparse

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import logging
    logging.set_verbosity_error()
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

from src.retrieval.retriever import LegalRetriever


import re

def has_repetitive_loop(text: str) -> bool:
    """Kiểm tra xem câu trả lời có bị lặp từ/cụm từ vô hạn (loop) hay không."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for i in range(len(lines) - 2):
        if lines[i] == lines[i+1] == lines[i+2]:
            print(f"[validator] Phát hiện lặp dòng liên tiếp: '{lines[i]}'")
            return True
            
    for line in lines:
        words = line.split()
        if len(words) > 15:
            # Check sliding window of size 2 to 8
            for size in range(2, 9):
                for start in range(len(words) - 3 * size):
                    w1 = words[start : start + size]
                    w2 = words[start + size : start + 2 * size]
                    w3 = words[start + 2 * size : start + 3 * size]
                    if w1 == w2 == w3:
                        print(f"[validator] Phát hiện lặp cụm từ vô hạn: {' '.join(w1)}")
                        return True
    return False

DATE_FULL_RE = re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")  # dd/mm/yyyy
DOC_NUM_RE = re.compile(r"\d+/(?:19|20)\d{2}(?:/[A-ZĐƠƯ0-9\-]+)?")  # số/năm[/loại]

def normalize_doc_key(s: str) -> str:
    parts = s.split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else s

def extract_doc_numbers_safe(text: str) -> set:
    # Loại bỏ cụm ngày-tháng-năm đầy đủ trước, tránh bắt nhầm phần đuôi (vd "07/2023" trong "01/07/2023")
    text_wo_dates = DATE_FULL_RE.sub(" ", text)
    return {normalize_doc_key(d) for d in DOC_NUM_RE.findall(text_wo_dates)}

def validate_citations_detailed(answer: str, results: list):
    if has_repetitive_loop(answer):
        return False, "câu trả lời bị lặp từ/cụm từ vô hạn (loop)"

    FEW_SHOT_DOCS = {"12/2022"}
    FEW_SHOT_ARTICLES = {"34"}

    # 1. ĐIỀU LUẬT — giữ nguyên như cũ
    answer_articles = set(re.findall(r"Điều\s+(\d+)", answer, re.IGNORECASE))
    if answer_articles:
        context_articles = set()
        for r in results:
            if r.article_hint:
                context_articles.update(re.findall(r"Điều\s+(\d+)", r.article_hint, re.IGNORECASE))
            context_articles.update(re.findall(r"Điều\s+(\d+)", r.content, re.IGNORECASE))
        invalid_articles = (answer_articles - context_articles) - FEW_SHOT_ARTICLES
        if invalid_articles:
            return False, f"dẫn chiếu sai các Điều không có trong context: {invalid_articles}"

    # 2. SỐ HIỆU VĂN BẢN — FIX: loại trừ ngày tháng, chuẩn hoá để so sánh nhất quán
    answer_docs = extract_doc_numbers_safe(answer)
    if answer_docs:
        context_docs = set()
        for r in results:
            if r.doc_number:
                context_docs.update(extract_doc_numbers_safe(r.doc_number))
            context_docs.update(extract_doc_numbers_safe(r.content))
        invalid_docs = (answer_docs - context_docs) - FEW_SHOT_DOCS
        if invalid_docs:
            return False, f"dẫn chiếu sai các số hiệu Văn bản không có trong context: {invalid_docs}"

    return True, ""

def generate_rule_based_answer(results: list) -> str:
    """Tạo câu trả lời rule-based có cấu trúc rõ ràng dựa trên các tài liệu pháp lý đã tìm thấy."""
    if not results:
        return "Không tìm thấy căn cứ pháp lý liên quan để trả lời câu hỏi này."
    
    answer_parts = []
    answer_parts.append("1. Trả lời trực tiếp: Căn cứ vào các văn bản pháp luật hiện hành được tìm thấy, dưới đây là thông tin trích xuất liên quan đến câu hỏi:")
    
    analysis = []
    cơ_sở = []
    
    for idx, r in enumerate(results[:3], start=1):
        ref = f"{r.legal_type} số {r.doc_number}"
        if r.title:
            ref += f" ({r.title})"
        if r.article_hint:
            ref += f" - {r.article_hint}"
            
        content_snippet = r.content.strip()
        if "Nội dung:" in content_snippet:
            content_snippet = content_snippet.split("Nội dung:", 1)[1].strip()
            
        # Giới hạn độ dài snippet để tránh làm answer quá dài
        if len(content_snippet) > 300:
            content_snippet = content_snippet[:297] + "..."
            
        analysis.append(f"- Căn cứ {idx} quy định: {content_snippet}")
        cơ_sở.append(f"- {r.article_hint or 'Quy định'} {r.legal_type} số {r.doc_number}")

    answer_parts.append("2. Phân tích chi tiết:\n" + "\n".join(analysis))
    answer_parts.append("3. Căn cứ pháp lý:\n" + "\n".join(cơ_sở))
    answer_parts.append("4. Hạn chế của dữ liệu (nếu có): Do phương pháp trích xuất tự động, vui lòng tra cứu văn bản gốc để xem toàn bộ nội dung chi tiết.")
    
    return "\n\n".join(answer_parts)


def clean_vietnamese_text(text: str) -> str:
    # Convert to lowercase and normalize spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[.,\-–/\"\'()“”:_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_doc_number(num: str) -> str:
    if not num:
        return ""
    num = num.lower().replace('đ', 'd').strip()
    return num

def is_doc_number_cited(doc_number: str, answer: str) -> bool:
    if not doc_number:
        return False
        
    ans_clean = clean_doc_number(answer)
    num_clean = clean_doc_number(doc_number)
    
    # 1. Direct match
    if num_clean in ans_clean:
        return True
        
    # 2. Match without the suffix part (e.g. 04/2017/QH14 -> 04/2017)
    parts_slash = num_clean.split('/')
    if len(parts_slash) >= 2:
        base_slash = f"{parts_slash[0]}/{parts_slash[1]}"
        if base_slash in ans_clean:
            return True
            
    # 3. Match without dash suffix (e.g. 04/vbhn-btc -> 04/vbhn, 09/vbhn-vpqh -> 09/vbhn)
    parts_dash = num_clean.split('-')
    if len(parts_dash) >= 2:
        base_dash = parts_dash[0]
        # Only use if base_dash has a slash and is specific enough (e.g. contains at least 4 chars)
        if '/' in base_dash and len(base_dash) >= 4:
            if base_dash in ans_clean:
                return True
                
    return False

def is_doc_name_mentioned(legal_type: str, doc_number: str, title: str, answer: str) -> bool:
    if not title:
        return False
    legal_type = legal_type.strip() if legal_type else ""
    title = title.strip()

    # FIX: title luôn có dạng '{LoaiVB} {SoHieu} {TenThat}'. Phải bỏ doc_number
    # khỏi title trước, nếu không candidate sẽ luôn chứa số hiệu và không bao giờ
    # match được answer chỉ ghi tên suông (vd "Luật Sở hữu trí tuệ 2005").
    core = title.replace(doc_number, "").strip() if doc_number else title
    core = re.sub(r"\s+", " ", core)

    # FIX: sau khi bỏ số hiệu, core thường có dạng lặp '{LoaiVB} {LoaiVB} {TenThuan}'
    # vì legal_type vốn đã nằm sẵn trong title gốc trước số hiệu.
    pure_name = core
    if legal_type and core.lower().startswith(legal_type.lower()):
        pure_name = core[len(legal_type):].strip()

    candidates = [core, pure_name]
    if legal_type:
        candidates.append(f"{legal_type} {pure_name}".strip())

    extended_candidates = []
    for c in candidates:
        extended_candidates.append(c)
        c_no_year = re.sub(r'\b(năm\s+)?20\d{2}\b', '', c, flags=re.IGNORECASE)
        c_no_year = re.sub(r'\b(năm\s+)?19\d{2}\b', '', c_no_year, flags=re.IGNORECASE)
        extended_candidates.append(c_no_year.strip())

    clean_ans = clean_vietnamese_text(answer)
    for c in extended_candidates:
        clean_c = clean_vietnamese_text(c)
        if len(clean_c.split()) >= 3 and clean_c in clean_ans:
            return True
    return False

    if not title:
        return False
        
    legal_type = legal_type.strip() if legal_type else ""
    title = title.strip()
    
    # Build full name candidates
    candidates = []
    
    # Check if title already starts with legal_type
    title_lower = title.lower()
    legal_lower = legal_type.lower()
    
    if legal_lower and title_lower.startswith(legal_lower):
        candidates.append(title)
    else:
        candidates.append(title)
        if legal_type:
            candidates.append(f"{legal_type} {title}")
            
    # Also try stripping year suffix from candidates
    # E.g. "năm 2017", "2017", "năm 2021", etc.
    extended_candidates = []
    for c in candidates:
        extended_candidates.append(c)
        c_no_year = re.sub(r'\b(năm\s+)?20\d{2}\b', '', c, flags=re.IGNORECASE)
        c_no_year = re.sub(r'\b(năm\s+)?19\d{2}\b', '', c_no_year, flags=re.IGNORECASE)
        extended_candidates.append(c_no_year.strip())
        
    # Clean both candidates and answer
    clean_ans = clean_vietnamese_text(answer)
    for c in extended_candidates:
        clean_c = clean_vietnamese_text(c)
        # Only match if the candidate name has at least 3 words to avoid generic short names
        if len(clean_c.split()) >= 3:
            if clean_c in clean_ans:
                return True
    return False

def is_doc_cited(r, answer: str) -> bool:
    if is_doc_number_cited(r.doc_number, answer):
        return True
    if is_doc_name_mentioned(r.legal_type, r.doc_number, r.title, answer):  # thêm r.doc_number
        return True
    if r.legal_type and r.legal_type.lower() == "hiến pháp":
        if "hiến pháp" in answer.lower():
            return True
    return False


def extract_cited_pairs(answer: str) -> list:
    """Trích xuất các cặp (doc_number, article_number) từ câu trả lời."""
    all_docs = list(set(re.findall(r'\b\d+/\d+/[A-ZĐa-zđ0-9\-–/]+\b', answer)))
    all_arts = list(set(int(x) for x in re.findall(r'\bĐiều\s+(\d+)\b', answer, re.IGNORECASE)))
    
    if not all_docs or not all_arts:
        return []
        
    if len(all_docs) == 1:
        # Nếu chỉ có 1 văn bản được nhắc tới, map nó với tất cả các Điều tìm thấy
        return [(all_docs[0], art) for art in all_arts]
        
    # Nếu có nhiều văn bản, dùng proximity matching (khoảng cách giữa số Điều và số văn bản <= 120 ký tự)
    pairs = []
    # Pattern 1: Điều X ... DocNum
    pattern1 = r'\bĐiều\s+(\d+)\b[\s\S]{1,120}?\b(\d+/\d+/[A-ZĐa-zđ0-9\-–/]+)\b'
    for match in re.finditer(pattern1, answer, re.IGNORECASE):
        art_num = int(match.group(1))
        doc_num = match.group(2)
        pairs.append((doc_num, art_num))
        
    # Pattern 2: DocNum ... Điều X
    pattern2 = r'\b(\d+/\d+/[A-ZĐa-zđ0-9\-–/]+)\b[\s\S]{1,120}?\bĐiều\s+(\d+)\b'
    for match in re.finditer(pattern2, answer, re.IGNORECASE):
        doc_num = match.group(1)
        art_num = int(match.group(2))
        pairs.append((doc_num, art_num))
        
    return list(set(pairs))


# Global counters for fallback activations
FALLBACK_DOCS_COUNT = 0
FALLBACK_ARTICLES_COUNT = 0

def build_submission_entry(qid: int, question: str, results: list, answer: str = "") -> dict:
    """Tạo một entry trong results.json theo đúng format cuộc thi với bộ lọc trích dẫn."""
    global FALLBACK_DOCS_COUNT, FALLBACK_ARTICLES_COUNT
    
    # Sao chép nông kết quả ban đầu tránh đột biến danh sách gốc ngoài phạm vi
    results = list(results)
    
    # Tích hợp DB lookup trực tiếp từ SQLite local nếu có câu trả lời sinh ra
    if answer and os.path.exists(DB_PATH):
        try:
            cited_pairs = extract_cited_pairs(answer)
            all_arts = [int(x) for x in re.findall(r'\bĐiều\s+(\d+)\b', answer, re.IGNORECASE)]
            cited_in_results = [r for r in results if is_doc_cited(r, answer)]
            
            # Gộp thêm trích dẫn suy luận từ tài liệu có sẵn trong kết quả
            if len(cited_in_results) == 1 and all_arts:
                for art in all_arts:
                    cited_pairs.append((cited_in_results[0].doc_number, art))
            elif len(cited_in_results) > 1 and all_arts:
                for r in cited_in_results:
                    for art in all_arts:
                        escaped_num = re.escape(clean_doc_number(r.doc_number))
                        escaped_name = re.escape(clean_vietnamese_text(r.title.replace(r.doc_number, "")))
                        if len(escaped_name.split()) >= 3:
                            pattern_str = rf'(?:Điều\s+{art}\b[\s\S]{{1,120}}?(?:{escaped_num}|{escaped_name})|(?:{escaped_num}|{escaped_name})[\s\S]{{1,120}}?Điều\s+{art}\b)'
                            if re.search(pattern_str, answer, re.IGNORECASE):
                                cited_pairs.append((r.doc_number, art))
                                
            cited_pairs = list(set(cited_pairs))
            
            if cited_pairs:
                import sqlite3
                from src.retrieval.retriever import _parse_meta_from_content, RetrievalResult
                
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                
                doc_to_id = {}
                
                for doc_num, art_num in cited_pairs:
                    exists_in_results = False
                    for r in results:
                        if r.doc_number == doc_num and r.article_number == art_num:
                            exists_in_results = True
                            break
                    if exists_in_results:
                        continue
                        
                    if doc_num not in doc_to_id:
                        doc_pattern = f"%({doc_num})%"
                        cur.execute("SELECT document_id FROM document_chunks WHERE content LIKE ? LIMIT 1", (doc_pattern,))
                        row = cur.fetchone()
                        if row:
                            doc_to_id[doc_num] = row["document_id"]
                        else:
                            wildcard = doc_num.replace('đ', '_').replace('Đ', '_').replace('d', '_').replace('D', '_')
                            cur.execute("SELECT document_id FROM document_chunks WHERE content LIKE ? LIMIT 1", (f"%({wildcard})%",))
                            row = cur.fetchone()
                            if row:
                                doc_to_id[doc_num] = row["document_id"]
                            else:
                                doc_to_id[doc_num] = None
                                
                    doc_id = doc_to_id.get(doc_num)
                    if doc_id is not None:
                        cur.execute("""
                            SELECT id, document_id, chunk_index, article_hint, article_number, content 
                            FROM document_chunks 
                            WHERE document_id = ? AND article_number = ?
                            LIMIT 1
                        """, (doc_id, art_num))
                        chunk_row = cur.fetchone()
                        if chunk_row:
                            content = chunk_row["content"] or ""
                            meta = _parse_meta_from_content(content)
                            new_r = RetrievalResult(
                                chunk_id=str(chunk_row["id"]),
                                document_id=chunk_row["document_id"],
                                chunk_index=chunk_row["chunk_index"],
                                content=content,
                                doc_number=meta["document_number"],
                                title=meta["title"],
                                legal_type=meta["legal_type"],
                                score=1.0,
                                source="db_lookup",
                                article_hint=chunk_row["article_hint"],
                                article_number=chunk_row["article_number"]
                            )
                            results.append(new_r)
                            print(f"  [DB Lookup] Recovered missing citation: {doc_num} Điều {art_num}", flush=True)
                conn.close()
        except Exception as e:
            print(f"  [DB Lookup] Exception occurred during lookup: {e}", flush=True)

    docs_seen = set()
    articles_seen = set()
    relevant_docs = []
    relevant_articles = []

    # 1. Nhận diện các tài liệu được trích dẫn thực tế trong câu trả lời
    for r in results:
        doc_str = r.format_relevant_doc()
        art_str = r.format_relevant_article()
        
        if is_doc_cited(r, answer):
            if doc_str not in docs_seen:
                docs_seen.add(doc_str)
                relevant_docs.append(doc_str)
                
            # Kiểm tra xem Điều luật cụ thể có được trích dẫn không
            if r.article_hint:
                m = re.search(r'\d+', r.article_hint)
                if m:
                    art_num = m.group()
                    art_pattern = rf'(?:Điều|khoản)\s+{art_num}\b'
                    if re.search(art_pattern, answer, re.IGNORECASE):
                        if art_str not in articles_seen:
                            articles_seen.add(art_str)
                            relevant_articles.append(art_str)
                else:
                    if art_str not in articles_seen:
                        articles_seen.add(art_str)
                        relevant_articles.append(art_str)
            else:
                if art_str not in articles_seen:
                    articles_seen.add(art_str)
                    relevant_articles.append(art_str)

    # 2. Fallback: Luôn giữ lại ít nhất tài liệu Top 1 để đảm bảo độ bao phủ cơ sở (Base Recall)
    fallback_doc_triggered = False
    fallback_art_triggered = False

    if not relevant_docs and results:
        FALLBACK_DOCS_COUNT += 1
        fallback_doc_triggered = True
        top_r = results[0]
        doc_str = top_r.format_relevant_doc()
        if doc_str not in docs_seen:
            docs_seen.add(doc_str)
            relevant_docs.append(doc_str)
            
    if not relevant_articles:
        fallback_art_triggered = True
        # Fallback lấy các điều thuộc tài liệu được chọn
        for r in results:
            doc_str = r.format_relevant_doc()
            if doc_str in docs_seen:
                art_str = r.format_relevant_article()
                if art_str not in articles_seen:
                    articles_seen.add(art_str)
                    relevant_articles.append(art_str)
                    
    if not relevant_articles and results:
        FALLBACK_ARTICLES_COUNT += 1
        top_r = results[0]
        art_str = top_r.format_relevant_article()
        if art_str not in articles_seen:
            articles_seen.add(art_str)
            relevant_articles.append(art_str)

    if fallback_doc_triggered or fallback_art_triggered:
        print(f"  [Citation Filter] QID {qid}: Fallback activated (doc={fallback_doc_triggered}, art={fallback_art_triggered})", flush=True)

    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch retrieval + LLM Qwen3 → results.json"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="File JSON danh sách câu hỏi [{id, question}, ...]")
    parser.add_argument("--output", "-o", default="results.json",
                        help="File output (mặc định: results.json)")
    parser.add_argument("--mode", "-m", choices=["fts", "vector", "hybrid"],
                        default="hybrid")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--vector-weight", type=float, default=0.5)
    parser.add_argument("--fts-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank", "-r", action="store_true", default=True,
                        help="Kích hoạt Reranker PhoRanker để tăng độ chính xác tìm kiếm (mặc định: BẬT)")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Tắt Reranker PhoRanker")
    parser.add_argument("--llm", action="store_true",
                        help="Kích hoạt mô hình sinh câu trả lời tự động")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-Instruct",
                        help="Tên mô hình LLM trên HuggingFace (mặc định: Qwen/Qwen3-8B-Instruct)")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Kích thước batch khi sinh câu trả lời LLM (mặc định: 3)")
    parser.add_argument("--cache-file", default="stage1_cache.json",
                        help="File lưu cache kết quả Giai đoạn 1 (mặc định: stage1_cache.json)")
    args = parser.parse_args()

    # Load câu hỏi
    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        questions = [questions]

    print(f"[batch] {len(questions)} questions from {args.input}")
    print(f"[batch] Mode: {args.mode} | Top-K: {args.top_k} | Rerank: {args.rerank} | LLM: {args.llm}")
    print("-" * 60)

    retriever = LegalRetriever(
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        fts_weight=args.fts_weight,
        rrf_k=args.rrf_k,
        use_postgres=False,
    )
    print("[batch] Backend: LOCAL (SQLite)")

    import dataclasses
    from src.retrieval.retriever import RetrievalResult

    # 1. Giai đoạn 1: Truy xuất tài liệu cho tất cả các câu hỏi
    print("[batch] === GIAI ĐOẠN 1: TRUY XUẤT TÀI LIỆU (RETRIEVAL) ===")
    retrieved_data = []
    
    if args.cache_file and os.path.exists(args.cache_file):
        print(f"[batch] Tìm thấy file cache '{args.cache_file}'. Đang nạp dữ liệu bỏ qua Giai đoạn 1...")
        try:
            with open(args.cache_file, "r", encoding="utf-8") as f:
                cached_list = json.load(f)
            for item in cached_list:
                qid = item["qid"]
                question = item["question"]
                results_dicts = item["results"]
                elapsed = item["elapsed"]
                results = [RetrievalResult(**r) for r in results_dicts]
                retrieved_data.append((qid, question, results, elapsed))
            print(f"[batch] ✓ Đã nạp {len(retrieved_data)} câu hỏi từ cache thành công.")
        except Exception as e:
            print(f"[batch] ✗ Lỗi nạp file cache: {e}. Sẽ chạy lại truy xuất từ đầu.")
            retrieved_data = []

    if not retrieved_data:
        for idx, q in enumerate(questions, start=1):
            qid = q.get("id", idx)
            question = q.get("question", "")

            if not question.strip():
                print(f"  [{idx:3d}/{len(questions)}] id={qid} ⚠ Câu hỏi rỗng, bỏ qua.")
                continue

            t0 = time.time()
            try:
                results = retriever.retrieve(
                    question, 
                    mode=args.mode, 
                    top_k=args.top_k, 
                    rerank=args.rerank
                )
                elapsed = time.time() - t0
                retrieved_data.append((qid, question, results, elapsed))
                print(f"  [{idx:3d}/{len(questions)}] id={qid} | Reranked/Retrieved | {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{idx:3d}/{len(questions)}] id={qid} ✗ LỖI RETRIEVAL: {e} ({elapsed:.2f}s)", flush=True)
                retrieved_data.append((qid, question, [], elapsed))
                
        # Lưu Cache ngay khi xong
        if args.cache_file:
            print(f"\n[batch] Đang lưu cache Giai đoạn 1 ra '{args.cache_file}'...")
            try:
                cached_list = []
                for qid, question, results, elapsed in retrieved_data:
                    results_dicts = [dataclasses.asdict(r) for r in results]
                    cached_list.append({
                        "qid": qid,
                        "question": question,
                        "results": results_dicts,
                        "elapsed": elapsed
                    })
                with open(args.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached_list, f, ensure_ascii=False, indent=2)
                print(f"[batch] ✓ Đã lưu cache thành công.")
            except Exception as e:
                print(f"[batch] ✗ Lỗi lưu file cache: {e}")

    # Giải phóng hoàn toàn Retriever và các model nhúng, reranker trên GPU
    print("\n[batch] Giải phóng retriever và giải phóng GPU memory...")
    if hasattr(retriever, '_model'):
        del retriever._model
    if hasattr(retriever, '_reranker'):
        del retriever._reranker
    retriever.close()
    del retriever
    
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1) # Chờ 1 giây để GPU giải phóng hoàn toàn
    
    # 2. Giai đoạn 2: Khởi tạo Generator và sinh câu trả lời
    submission = []
    total_time = 0.0
    
    if args.llm:
        print("\n[batch] === GIAI ĐOẠN 2: KHỞI TẠO LLM VÀ SINH CÂU TRẢ LỜI (GENERATION) ===")
        from src.llm.llm_client import QwenGenerator
        generator = QwenGenerator(model_name=args.llm_model)
        # lazy load model lúc này sẽ có toàn bộ VRAM
        generator._lazy_load()
        
        batch_size = args.batch_size
        for i in range(0, len(retrieved_data), batch_size):
            batch_data = retrieved_data[i:i+batch_size]
            
            # Tách các câu hỏi hợp lệ (có results)
            valid_items = []
            for item in batch_data:
                qid, question, results, r_time = item
                if not results:
                    entry = build_submission_entry(qid, question, [], answer="Không tìm thấy tài liệu luật pháp liên quan.")
                    submission.append(entry)
                else:
                    valid_items.append(item)
                    
            if not valid_items:
                continue
                
            batch_queries = [item[1] for item in valid_items]
            batch_results_list = [item[2][:3] for item in valid_items]
            
            print(f"\n[batch] Đang sinh câu trả lời cho batch {i//batch_size + 1} ({len(valid_items)} câu)...", flush=True)
            t0 = time.time()
            try:
                # Chạy inference song song cho toàn bộ batch (lượt 1)
                batch_answers = generator.generate_batch_answers(batch_queries, batch_results_list, max_new_tokens=512)
                elapsed_batch = time.time() - t0
                
                # Kiểm tra và xử lý retry cho từng câu
                for idx_in_batch, item in enumerate(valid_items):
                    qid, question, results, r_time = item
                    answer = batch_answers[idx_in_batch]
                    
                    is_valid, err_msg = validate_citations_detailed(answer, results[:3])
                    extra_time = 0.0
                    
                    if not is_valid:
                        # Rơi vào quá trình retry tuần tự cho câu bị lỗi
                        t_retry = time.time()
                        max_retries = 2
                        warning_msg = f"LƯU Ý LỚN: Ở lượt sinh trước, bạn đã mắc lỗi: {err_msg}. Hãy chú ý sửa lỗi này, không được lặp lại và không dẫn chiếu sai lệch."
                        print(f"  id={qid} ⚠ Lỗi sinh ({err_msg}). Chuyển sang Retry tuần tự...")
                        for attempt in range(1, max_retries): # attempt=1 tức là thử lần 2
                            answer = generator.generate_answer(question, results[:3], max_new_tokens=512, warning_msg=warning_msg)
                            is_valid, err_msg = validate_citations_detailed(answer, results[:3])
                            if is_valid:
                                break
                            else:
                                print(f"  id={qid} ⚠ Lỗi sinh ({err_msg}) lần {attempt+1}. Đang sinh lại...")
                                warning_msg = f"LƯU Ý LỚN: Ở lượt sinh trước, bạn đã mắc lỗi: {err_msg}. Hãy chú ý sửa lỗi này, không được lặp lại và không dẫn chiếu sai lệch."
                        else:
                            print(f"  id={qid} ⚠ Đã thử {max_retries} lần vẫn lỗi. Fallback sang Rule-based.")
                            answer = generate_rule_based_answer(results)
                        extra_time = time.time() - t_retry
                    
                    # Thời gian chia đều cho batch + thời gian lấy kết quả retrieval + thời gian retry (nếu có)
                    elapsed = (elapsed_batch / len(valid_items)) + r_time + extra_time
                    total_time += elapsed
                    
                    entry = build_submission_entry(qid, question, results, answer=answer)
                    submission.append(entry)
                    
                    docs_count = len(entry["relevant_docs"])
                    arts_count = len(entry["relevant_articles"])
                    print(f"  id={qid} | {docs_count} docs, {arts_count} articles | LLM Answer: Yes | {elapsed:.2f}s")
            
            except Exception as e:
                print(f"  ✗ LỖI BATCH LLM: {e}", flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                for item in valid_items:
                    submission.append(build_submission_entry(item[0], item[1], item[2], answer=""))
    else:
        print("\n[batch] === GIAI ĐOẠN 2: SINH CÂU TRẢ LỜI RULE-BASED ===")
        for idx, (qid, question, results, r_time) in enumerate(retrieved_data, start=1):
            t0 = time.time()
            answer = generate_rule_based_answer(results)
            elapsed = time.time() - t0 + r_time
            total_time += elapsed
            
            entry = build_submission_entry(qid, question, results, answer=answer)
            submission.append(entry)
            
            docs_count = len(entry["relevant_docs"])
            arts_count = len(entry["relevant_articles"])
            print(f"  [{idx:3d}/{len(questions)}] id={qid} | "
                  f"{docs_count} docs, {arts_count} articles | Rule Answer | {elapsed:.2f}s")

    # Ghi output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    avg_time = total_time / len(questions) if questions else 0
    print("-" * 60)
    print(f"[batch] ✓ Xong! {len(submission)} câu | avg {avg_time:.2f}s/câu")
    print(f"[batch] [Citation Filter Stats] Fallback Top-1 triggered: Docs={FALLBACK_DOCS_COUNT}, Articles={FALLBACK_ARTICLES_COUNT}")
    print(f"[batch] ✓ Đã ghi → {args.output}")
    print()
    print("Để nén và nộp bài:")
    print(f"  Compress-Archive -Path {args.output} -DestinationPath submission.zip")


if __name__ == "__main__":
    main()
