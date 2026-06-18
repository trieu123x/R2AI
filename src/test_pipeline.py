import os
import sys
import json
import time

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.main_pipeline import LegalRAGPipeline

def main():
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    # Load questions
    data_path = os.path.join(PROJECT_ROOT, "R2AIStage1DATA.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    test_questions = data[:5]
    print(f"[test_pipeline] Loaded {len(test_questions)} questions for testing.")

    # Initialize full Legal RAG Pipeline using the Qwen 0.5B model
    print("[test_pipeline] Initializing LegalRAGPipeline...")
    pipeline = LegalRAGPipeline(
        use_llm_rewrite=False,
        llm_model_name="Qwen/Qwen2.5-0.5B-Instruct"
    )

    results = []
    total_time = 0.0

    print("[test_pipeline] Running pipeline execution loop...")
    for idx, q in enumerate(test_questions, start=1):
        qid = q.get("id", idx)
        question = q.get("question", "")

        t0 = time.time()
        print(f"\n=========================================")
        print(f"[{idx:3d}/{len(test_questions)}] id={qid} | Processing Question:")
        print(f"\"{question}\"")
        print("=========================================")

        try:
            out = pipeline.run(question)
            elapsed = time.time() - t0
            total_time += elapsed

            print(f"Done in {elapsed:.2f}s")
            
            # Format answers and references according to R2AI requirements
            relevant_docs = []
            docs_seen = set()
            for r in out.get("top5_results", []):
                doc_str = r.format_relevant_doc()
                if doc_str not in docs_seen:
                    docs_seen.add(doc_str)
                    relevant_docs.append(doc_str)

            relevant_articles = []
            articles_seen = set()
            for r in out.get("top5_results", []):
                if r.article_hint:
                    art_str = r.format_relevant_article()
                    if art_str not in articles_seen:
                        articles_seen.add(art_str)
                        relevant_articles.append(art_str)

            results.append({
                "id": qid,
                "question": question,
                "answer": out.get("final_answer", ""),
                "relevant_docs": relevant_docs,
                "relevant_articles": relevant_articles
            })

            print(f"Generated Answer:\n{out.get('final_answer')}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ ERROR processing ID {qid}: {e} ({elapsed:.2f}s)")
            import traceback
            traceback.print_exc(file=sys.stdout)
            results.append({
                "id": qid,
                "question": question,
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": []
            })

    # Save to output file
    output_path = os.path.join(PROJECT_ROOT, "test_pipeline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_time = total_time / len(test_questions) if test_questions else 0
    print("\n" + "="*50)
    print(f"[test_pipeline] Completed! Average time: {avg_time:.2f}s per question.")
    print(f"[test_pipeline] Results saved to: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
