import os
import sys
import json
import time

# Force UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.main_pipeline import LegalRAGPipeline

def test_samples(file_path: str, num_samples: int = 5):
    print(f"--- ĐANG LOAD {num_samples} CÂU HỎI TỪ FILE: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Lỗi: Format file không phải là list JSON.")
        return

    # Lấy 5 câu hỏi đầu tiên
    samples = data[:num_samples]
    
    # Khởi tạo Pipeline
    pipeline = LegalRAGPipeline(use_llm_rewrite=False, llm_model_name="Qwen/Qwen2.5-0.5B-Instruct")
    
    for idx, item in enumerate(samples, start=1):
        q_id = item.get("id", idx)
        question = item.get("question", "")
        
        print(f"\n========================================================")
        print(f"Câu hỏi {idx} (ID: {q_id}): {question}")
        print(f"========================================================")
        
        t0 = time.time()
        res = pipeline.run(question)
        elapsed = time.time() - t0
        
        # In ra các thông tin từ Pipeline
        print(f"\n[1] Query Rewrite: {res['rewritten_query']}")
        print(f"[2] Số documents tìm thấy (Top 5): {len(res['top5_results'])}")
        print(f"[3] Thời gian xử lý: {elapsed:.2f}s")
        print(f"[4] Câu trả lời sinh ra (Valid: {res['is_valid']}):\n")
        print(res["final_answer"])
        print("\n" + "-"*60)

if __name__ == "__main__":
    data_file = os.path.join(PROJECT_ROOT, "R2AIStage1DATA.json")
    test_samples(data_file, num_samples=5)
