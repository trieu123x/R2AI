import json
import subprocess
import sys

# Extract first 3 questions
with open("R2AIStage1DATA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

test_data = data[:5]
with open("test_questions.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("Created test_questions.json. Running batch_retrieve.py...")
result = subprocess.run([
    sys.executable, "retrieval/batch_retrieve.py",
    "--input", "test_questions.json",
    "--output", "test_results.json",
    "--local",
    "--mode", "hybrid",
    "--top-k", "5",
    "--rerank",
    "--llm",
    "--llm-model", "Qwen/Qwen2.5-0.5B-Instruct"
], capture_output=True, text=True, encoding="utf-8")

with open("test_run_output.log", "w", encoding="utf-8") as out_f:
    out_f.write("STDOUT:\n")
    out_f.write(result.stdout)
    out_f.write("\nSTDERR:\n")
    out_f.write(result.stderr)

print("Saved output to test_run_output.log")

