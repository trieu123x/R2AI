import os
import time
import sys
import io
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from src.retrieval.local_retriever import LocalRetriever
    from src.llm.llm_client import QwenGenerator

    query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"

    retriever = LocalRetriever(top_k=5)
    results = retriever.retrieve(query, mode="hybrid", rerank=True)

    print("\nLoading Qwen2.5-0.5B-Instruct...")
    t0 = time.time()
    generator = QwenGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    generator._lazy_load()
    print(f"Loaded in {time.time() - t0:.2f}s")

    print("\nGenerating answer...")
    t0 = time.time()
    answer = generator.generate_answer(query, results[:3])
    print(f"Generated in {time.time() - t0:.2f}s")
    print("\nAnswer:")
    print(answer)
except Exception as e:
    print("EXCEPTION OCCURRED:")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
