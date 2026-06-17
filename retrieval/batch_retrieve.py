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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever


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

def validate_citations_detailed(answer: str, results: list):
    """Kiểm tra chi tiết trích dẫn của câu trả lời so với context."""
    if has_repetitive_loop(answer):
        return False, "câu trả lời bị lặp từ/cụm từ vô hạn (loop)"

    # Check Articles
    answer_articles = set(re.findall(r"Điều\s+(\d+)", answer, re.IGNORECASE))
    if answer_articles:
        context_articles = set()
        for r in results:
            if r.article_hint:
                matches = re.findall(r"Điều\s+(\d+)", r.article_hint, re.IGNORECASE)
                context_articles.update(matches)
            matches_content = re.findall(r"Điều\s+(\d+)", r.content, re.IGNORECASE)
            context_articles.update(matches_content)
            
        invalid_articles = answer_articles - context_articles
        if invalid_articles:
            return False, f"dẫn chiếu sai các Điều không có trong context: {invalid_articles}"
            
    # Check Documents
    answer_docs = set(re.findall(r"(\d+/\d+)", answer))
    if answer_docs:
        context_docs = set()
        for r in results:
            if r.doc_number:
                matches = re.findall(r"(\d+/\d+)", r.doc_number)
                context_docs.update(matches)
            matches_content = re.findall(r"(\d+/\d+)", r.content)
            context_docs.update(matches_content)
            
        invalid_docs = answer_docs - context_docs
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


def build_submission_entry(qid: int, question: str, results: list, answer: str = "") -> dict:
    """Tạo một entry trong results.json theo đúng format cuộc thi."""
    docs_seen = set()
    articles_seen = set()
    relevant_docs = []
    relevant_articles = []

    for r in results:
        doc_str = r.format_relevant_doc()
        if doc_str not in docs_seen:
            docs_seen.add(doc_str)
            relevant_docs.append(doc_str)

        if r.article_hint:
            art_str = r.format_relevant_article()
            if art_str not in articles_seen:
                articles_seen.add(art_str)
                relevant_articles.append(art_str)

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
    parser.add_argument("--local", "-l", action="store_true", default=True,
                        help="Dung local SQLite thay vi Supabase (offline mode) (mac dinh: BAT)")
    parser.add_argument("--postgres", action="store_true", default=False,
                        help="Dung Supabase/PostgreSQL (mac dinh: TAT)")
    parser.add_argument("--rerank", "-r", action="store_true", default=True,
                        help="Kích hoạt Reranker PhoRanker để tăng độ chính xác tìm kiếm (mặc định: BẬT)")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Tắt Reranker PhoRanker")
    parser.add_argument("--llm", action="store_true",
                        help="Kích hoạt mô hình sinh câu trả lời tự động")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-Instruct",
                        help="Tên mô hình LLM trên HuggingFace (mặc định: Qwen/Qwen3-8B-Instruct)")
    args = parser.parse_args()

    # Load câu hỏi
    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        questions = [questions]

    print(f"[batch] {len(questions)} questions from {args.input}")
    print(f"[batch] Mode: {args.mode} | Top-K: {args.top_k} | Rerank: {args.rerank} | LLM: {args.llm}")
    print("-" * 60)

    # Chọn retriever
    use_postgres = getattr(args, "postgres", False)
    if use_postgres:
        try:
            retriever = LegalRetriever(
                top_k=args.top_k,
                vector_weight=args.vector_weight,
                fts_weight=args.fts_weight,
                rrf_k=args.rrf_k,
                use_postgres=True,
            )
            retriever._get_pg_conn()
            print("[batch] Backend: LOCAL POSTGRESQL")
        except Exception as e:
            print(f"[batch] Local PostgreSQL failed ({e}), using LOCAL SQLite")
            use_postgres = False

    if not use_postgres:
        retriever = LegalRetriever(
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            fts_weight=args.fts_weight,
            rrf_k=args.rrf_k,
            use_postgres=False,
        )
        print("[batch] Backend: LOCAL (SQLite)")

    # 1. Giai đoạn 1: Truy xuất tài liệu cho tất cả các câu hỏi
    print("[batch] === GIAI ĐOẠN 1: TRUY XUẤT TÀI LIỆU (RETRIEVAL) ===")
    retrieved_data = []
    
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
        from retrieval.qwen_generator import QwenGenerator
        generator = QwenGenerator(model_name=args.llm_model)
        # lazy load model lúc này sẽ có toàn bộ VRAM
        generator._lazy_load()
        
        for idx, (qid, question, results, r_time) in enumerate(retrieved_data, start=1):
            if not results:
                entry = build_submission_entry(qid, question, [], answer="Không tìm thấy tài liệu luật pháp liên quan.")
                submission.append(entry)
                continue
                
            t0 = time.time()
            try:
                max_retries = 2
                warning_msg = None
                answer = ""
                for attempt in range(max_retries):
                    answer = generator.generate_answer(question, results[:5], warning_msg=warning_msg)
                    is_valid, err_msg = validate_citations_detailed(answer, results[:5])
                    if is_valid:
                        break
                    else:
                        print(f"  [{idx:3d}/{len(questions)}] id={qid} ⚠ Phát hiện lỗi sinh ({err_msg}) lần {attempt+1}. Đang sinh lại...")
                        warning_msg = f"LƯU Ý LỚN: Ở lượt sinh trước, bạn đã mắc lỗi: {err_msg}. Hãy chú ý sửa lỗi này, không được lặp lại và không dẫn chiếu sai lệch."
                else:
                    print(f"  [{idx:3d}/{len(questions)}] id={qid} ⚠ Đã thử {max_retries} lần vẫn lỗi. Fallback sang Rule-based.")
                    answer = generate_rule_based_answer(results)
                
                elapsed = time.time() - t0 + r_time
                total_time += elapsed
                
                entry = build_submission_entry(qid, question, results, answer=answer)
                submission.append(entry)
                
                docs_count = len(entry["relevant_docs"])
                arts_count = len(entry["relevant_articles"])
                print(f"  [{idx:3d}/{len(questions)}] id={qid} | "
                      f"{docs_count} docs, {arts_count} articles | LLM Answer: Yes | {elapsed:.2f}s")
                      
            except Exception as e:
                elapsed = time.time() - t0 + r_time
                total_time += elapsed
                print(f"  [{idx:3d}/{len(questions)}] id={qid} ✗ LỖI LLM: {e} ({elapsed:.2f}s)", flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                submission.append(build_submission_entry(qid, question, results, answer=""))
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
    print(f"[batch] ✓ Đã ghi → {args.output}")
    print()
    print("Để nén và nộp bài:")
    print(f"  Compress-Archive -Path {args.output} -DestinationPath submission.zip")


if __name__ == "__main__":
    main()
