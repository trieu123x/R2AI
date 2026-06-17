import os
import sys
import json
import time
import argparse

# Force UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.main_pipeline import LegalRAGPipeline

def build_submission_entry(qid: int, question: str, results: list, answer: str = "") -> dict:
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
    parser = argparse.ArgumentParser(description="Chạy bộ pipeline 10 bước hoàn chỉnh.")
    parser.add_argument("--input", "-i", required=True, help="File JSON danh sách câu hỏi")
    parser.add_argument("--output", "-o", default="pipeline_results.json", help="File output")
    parser.add_argument("--use-llm-rewrite", action="store_true", help="Dùng LLM để rewrite query")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-Instruct", help="Tên mô hình LLM trên HuggingFace")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        questions = [questions]

    print(f"[main] Bắt đầu xử lý {len(questions)} câu hỏi với Pipeline (Model: {args.llm_model}).")
    pipeline = LegalRAGPipeline(use_llm_rewrite=args.use_llm_rewrite, llm_model_name=args.llm_model)

    submission = []
    total_time = 0.0

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id", idx)
        question = q.get("question", "")
        if not question.strip():
            continue

        t0 = time.time()
        try:
            res = pipeline.run(question)
            
            top5 = res["top5_results"]
            answer = res["final_answer"]
            
            elapsed = time.time() - t0
            total_time += elapsed
            
            entry = build_submission_entry(qid, question, top5, answer=answer)
            submission.append(entry)
            
            print(f"  [{idx:3d}/{len(questions)}] id={qid} | docs={len(entry['relevant_docs'])} | time={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{idx:3d}/{len(questions)}] id={qid} ✗ LỖI: {e} ({elapsed:.2f}s)")
            submission.append({
                "id": qid,
                "question": question,
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": [],
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    avg_time = total_time / len(questions) if questions else 0
    print("-" * 60)
    print(f"[main] ✓ Hoàn thành! avg {avg_time:.2f}s/câu")
    print(f"[main] ✓ Ghi vào {args.output}")

if __name__ == "__main__":
    main()
