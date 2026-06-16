import os
import sys
import json
import time
import io

# Force UTF-8 output on Windows to handle Vietnamese characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.local_retriever import LocalRetriever

def main():
    json_path = os.path.join(PROJECT_ROOT, "R2AIStage1DATA.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        sys.exit(1)
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Get first 20 questions
    questions = data[:20]
    
    print("Initializing LocalRetriever with optimizations...")
    t_start = time.time()
    # Force offline loading to skip HuggingFace connection attempt
    os.environ["HF_HUB_OFFLINE"] = "1"
    retriever = LocalRetriever()
    
    # Warm up to load model and FAISS index
    retriever.retrieve("warm up", mode="hybrid")
    print(f"Initialization completed in {time.time() - t_start:.2f}s")
    
    results = []
    
    print("\nStarting evaluation of 20 legal queries...")
    for item in questions:
        q_id = item["id"]
        query = item["question"]
        print(f"Processing Q{q_id:02d}: {query[:60]}...")
        
        # Test FTS (Full-Text Search)
        t0 = time.time()
        fts_res = retriever.retrieve(query, mode="fts", top_k=5)
        fts_time = time.time() - t0
        
        # Test Vector Search
        t0 = time.time()
        vec_res = retriever.retrieve(query, mode="vector", top_k=5)
        vec_time = time.time() - t0
        
        # Test Hybrid Search
        t0 = time.time()
        hyb_res = retriever.retrieve(query, mode="hybrid", top_k=5)
        hyb_time = time.time() - t0
        
        # Retrieve the best documents/articles information
        top_docs_list = []
        docs_seen = set()
        for r in hyb_res:
            if r.doc_number not in docs_seen:
                docs_seen.add(r.doc_number)
                top_docs_list.append(f"{r.doc_number} ({r.title[:30]}...)")
        
        top_doc = top_docs_list[0] if top_docs_list else "N/A"
            
        results.append({
            "id": q_id,
            "question": query,
            "fts": {
                "time": fts_time,
                "top_results": [r.doc_number for r in fts_res[:3]]
            },
            "vector": {
                "time": vec_time,
                "top_results": [r.doc_number for r in vec_res[:3]]
            },
            "hybrid": {
                "time": hyb_time,
                "top_results": [
                    {
                        "doc_number": r.doc_number,
                        "title": r.title,
                        "article": r.article_hint,
                        "score": r.score
                    } for r in hyb_res
                ]
            },
            "top_hybrid_doc": top_doc
        })
        
    # Print Markdown Summary Table
    print("\n" + "="*100)
    print("                                EVALUATION LATENCY & RESULT SUMMARY")
    print("="*100)
    print(f"| ID  | FTS (s)  | Vector (s) | Hybrid (s) | Best Matching Document |")
    print(f"| :--- | :---: | :---: | :---: | :--- |")
    for r in results:
        q_id = r["id"]
        fts_t = r["fts"]["time"]
        vec_t = r["vector"]["time"]
        hyb_t = r["hybrid"]["time"]
        top_res = r["top_hybrid_doc"]
        print(f"| {q_id:02d}  | {fts_t:8.4f} | {vec_t:10.4f} | {hyb_t:10.4f} | {top_res} |")
    print("="*100)
    
    # Calculate Averages
    avg_fts = sum(r["fts"]["time"] for r in results) / len(results)
    avg_vec = sum(r["vector"]["time"] for r in results) / len(results)
    avg_hyb = sum(r["hybrid"]["time"] for r in results) / len(results)
    print(f"Average FTS Latency   : {avg_fts:.4f}s")
    print(f"Average Vector Latency: {avg_vec:.4f}s")
    print(f"Average Hybrid Latency: {avg_hyb:.4f}s")
    
    # Save detailed results to JSON
    output_path = os.path.join(PROJECT_ROOT, "retrieval", "stage1_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved detailed evaluation results to {output_path}")

if __name__ == "__main__":
    main()
