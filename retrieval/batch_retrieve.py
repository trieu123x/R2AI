"""
batch_retrieve.py
=================
Chạy retrieval trên tập câu hỏi và xuất results.json đúng định dạng nộp bài.

Input:  file JSON danh sách câu hỏi (format ban tổ chức cung cấp)
Output: results.json (file nộp bài thi)

Cách dùng:
  python retrieval/batch_retrieve.py --input questions.json
  python retrieval/batch_retrieve.py --input questions.json --mode hybrid --top-k 10
  python retrieval/batch_retrieve.py --input questions.json --output my_results.json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import LegalRetriever


def build_submission_entry(qid: int, question: str, results: list) -> dict:
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
        "answer": "",           # để trống, phần LLM sẽ điền sau
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch retrieval → results.json"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="File JSON danh sách câu hỏi [{id, question}, ...]")
    parser.add_argument("--output", "-o", default="results.json",
                        help="File output (mặc định: results.json)")
    parser.add_argument("--mode", "-m", choices=["fts", "vector", "hybrid"],
                        default="hybrid")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    parser.add_argument("--fts-weight", type=float, default=0.4)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--local", "-l", action="store_true",
                        help="Dung local SQLite thay vi Supabase (offline mode)")
    args = parser.parse_args()

    # Load câu hỏi
    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        questions = [questions]

    print(f"[batch] {len(questions)} questions from {args.input}")
    print(f"[batch] Mode: {args.mode} | Top-K: {args.top_k}")
    print("-" * 60)

    # Chọn retriever
    if getattr(args, "local", False):
        from retrieval.local_retriever import LocalRetriever
        retriever = LocalRetriever(
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            fts_weight=args.fts_weight,
            rrf_k=args.rrf_k,
        )
        print("[batch] Backend: LOCAL (SQLite)")
    else:
        try:
            retriever = LegalRetriever(
                top_k=args.top_k,
                vector_weight=args.vector_weight,
                fts_weight=args.fts_weight,
                rrf_k=args.rrf_k,
            )
            retriever._get_conn()
            print("[batch] Backend: SUPABASE")
        except Exception as e:
            print(f"[batch] Supabase failed ({e}), using LOCAL SQLite")
            from retrieval.local_retriever import LocalRetriever
            retriever = LocalRetriever(
                top_k=args.top_k,
                vector_weight=args.vector_weight,
                fts_weight=args.fts_weight,
                rrf_k=args.rrf_k,
            )

    submission = []
    total_time = 0.0

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id", idx)
        question = q.get("question", "")

        if not question.strip():
            print(f"  [{idx:3d}/{len(questions)}] id={qid} ⚠ Câu hỏi rỗng, bỏ qua.")
            continue

        t0 = time.time()
        try:
            results = retriever.retrieve(question, mode=args.mode, top_k=args.top_k)
            elapsed = time.time() - t0
            total_time += elapsed

            entry = build_submission_entry(qid, question, results)
            submission.append(entry)

            docs_count = len(entry["relevant_docs"])
            arts_count = len(entry["relevant_articles"])
            print(f"  [{idx:3d}/{len(questions)}] id={qid} | "
                  f"{docs_count} docs, {arts_count} articles | {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{idx:3d}/{len(questions)}] id={qid} ✗ LỖI: {e} ({elapsed:.2f}s)")
            # Vẫn thêm entry rỗng để không bị thiếu câu
            submission.append({
                "id": qid,
                "question": question,
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": [],
            })

    retriever.close()

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
