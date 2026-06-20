import json
import subprocess
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Extract first 3 questions
with open("R2AIStage1DATA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

test_data = data[:5]
with open("test_questions.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("Created test_questions.json. Running batch_retrieve.py...")
cmd = [
    sys.executable, "-u", "src/retrieval/batch_retrieve.py",
    "--input", "test_questions.json",
    "--output", "test_results.json",
    "--mode", "hybrid",
    "--top-k", "5",
    "--rerank",
    "--llm",
    "--llm-model", "Qwen/Qwen2.5-1.5B-Instruct"
]

with open("test_run_output.log", "w", encoding="utf-8") as out_f:
    out_f.write("STDOUT and STDERR:\n")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )
    
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_f.write(line)
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] batch_retrieve.py failed with exit code: {process.returncode}")
        sys.exit(process.returncode)

print("\nSaved output to test_run_output.log")

