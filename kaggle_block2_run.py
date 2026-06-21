

import subprocess, sys, os

WORKING = "/kaggle/working"

# ── Cấu hình ─────────────────────────────────────────────────────────────────
INPUT_FILE   = "/kaggle/working/database/R2AIStage1DATA.json"
OUTPUT_FILE  = os.path.join(WORKING, "results.json")
CHECKPOINT   = os.path.join(WORKING, "results_checkpoint.json")
LOG_FILE     = os.path.join(WORKING, "batch_run.log")

MODE         = "hybrid"
TOP_K        = 10
LLM_MODEL    = "Qwen/Qwen2.5-7B-Instruct"  # Đổi thành model bạn muốn dùng


# Script retrieval nằm trong working dir sau khi giải nén
SCRIPT = "/kaggle/working/code/retrieval/batch_retrieve.py"

# ── Kiểm tra file đầu vào ────────────────────────────────────────────────────
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Không tìm thấy file câu hỏi: {INPUT_FILE}\n→ Hãy chạy Block 1 trước!")

if not os.path.exists(SCRIPT):
    raise FileNotFoundError(f"Không tìm thấy script: {SCRIPT}\n→ Hãy chạy Block 1 trước!")

import json
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_questions = json.load(f)
print(f"📋 Tổng số câu hỏi: {len(all_questions)}")

# ── Checkpoint: tìm câu đã xử lý ─────────────────────────────────────────────
done_ids = set()
existing_results = []

checkpoint_file = CHECKPOINT if os.path.exists(CHECKPOINT) else (OUTPUT_FILE if os.path.exists(OUTPUT_FILE) else None)
if checkpoint_file:
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        done_ids = {r["id"] for r in existing_results if "id" in r}
        print(f"♻️  Resume từ checkpoint: {len(done_ids)} câu đã xử lý, còn {len(all_questions) - len(done_ids)} câu.")
    except Exception as e:
        print(f"⚠️  Không đọc được checkpoint: {e}. Bắt đầu từ đầu.")

remaining = [q for q in all_questions if q.get("id") not in done_ids]

if not remaining:
    print("✅ Tất cả câu hỏi đã xử lý! Kết quả tại:", OUTPUT_FILE)
    # Ghi output cuối
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    raise SystemExit(0)

# Ghi file câu hỏi còn lại vào temp
TEMP_INPUT = os.path.join(WORKING, "remaining_questions.json")
with open(TEMP_INPUT, "w", encoding="utf-8") as f:
    json.dump(remaining, f, ensure_ascii=False)
print(f"▶️  Đang xử lý {len(remaining)} câu hỏi còn lại...")

# ── Build command ─────────────────────────────────────────────────────────────
cmd = [
    sys.executable, "-u", SCRIPT,
    "--input",     TEMP_INPUT,
    "--output",    CHECKPOINT,   # Ghi vào checkpoint trước
    "--mode",      MODE,
    "--top-k",     str(TOP_K),
    "--rerank",
    "--llm",
    "--llm-model", LLM_MODEL,
]

print(f"\n🚀 Lệnh chạy:\n   {' '.join(cmd)}\n")
print("=" * 70)

# ── Chạy và stream output ─────────────────────────────────────────────────────
with open(LOG_FILE, "a", encoding="utf-8") as log_f:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=WORKING,
        env={**os.environ, "PYTHONPATH": WORKING},
    )

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_f.write(line)

    process.wait()

print("=" * 70)

# ── Merge checkpoint với kết quả cũ ──────────────────────────────────────────
if process.returncode == 0 and os.path.exists(CHECKPOINT):
    try:
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            new_results = json.load(f)

        merged = existing_results + new_results
        # Deduplicate theo id, ưu tiên kết quả mới nhất
        seen = {}
        for r in merged:
            seen[r.get("id")] = r
        final_results = sorted(seen.values(), key=lambda x: x.get("id", 0))

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Hoàn tất! Tổng {len(final_results)} câu → {OUTPUT_FILE}")
        print(f"📦 Để nộp bài:")
        print(f'   import shutil')
        print(f'   shutil.make_archive("/kaggle/working/submission", "zip", "/kaggle/working", "results.json")')

    except Exception as e:
        print(f"\n⚠️  Lỗi khi merge kết quả: {e}")
        print(f"   Kết quả batch mới nhất tại: {CHECKPOINT}")
else:
    print(f"\n❌ Batch chạy lỗi (exit code {process.returncode}). Xem log tại: {LOG_FILE}")
    print(f"   Kết quả đã làm vẫn được lưu tại: {CHECKPOINT}")
    sys.exit(process.returncode)
