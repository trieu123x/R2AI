"""
╔══════════════════════════════════════════════════════════════╗
║  BLOCK 1 — SETUP: Mount 2 dataset + Load BAAI reranker      ║
║                                                              ║
║  Kaggle Datasets cần add (Input):                            ║
║    1. r2ai-code  → upload thư mục src/ trực tiếp (Kaggle    ║
║                    tự extract, src/ có sẵn trong dataset)    ║
║    2. r2ai-data  → chứa: local_chunks.db, local_chunks.index ║
║                           bge-reranker-v2-m3/, R2AIStage1DATA.json ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, shutil, subprocess

# ── 0. Cài đặt thư viện tương thích để tránh lỗi 'prepare_for_model' ───────────
print("📦 Đang cấu hình phiên bản thư viện tương thích (transformers, tokenizers)...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location", "transformers==4.44.2", "tokenizers==0.19.1"],
        check=True
    )
    print("✅ Cài đặt thư viện thành công.")
except Exception as e:
    print(f"⚠️ Cảnh báo lỗi khi cài đặt thư viện: {e}")

WORKING   = "/kaggle/working"
INPUT_BASE = "/kaggle/input"

# ── 1. Tìm dataset theo tên flexible ─────────────────────────────────────────
def find_dataset(names: list, description: str) -> str:
    for name in names:
        path = os.path.join(INPUT_BASE, name)
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
            return path
    available = os.listdir(INPUT_BASE) if os.path.exists(INPUT_BASE) else []
    raise FileNotFoundError(
        f"❌ Không tìm thấy {description}!\n"
        f"   Thử các tên: {names}\n"
        f"   Dataset có sẵn trong /kaggle/input: {available}"
    )

CODE_DATASET_NAMES = ["r2ai-code", "r2aicode", "r2ai-src"]
DATA_DATASET_NAMES = ["r2ai-data", "r2aidata", "r2aiiii", "r2ai-project-data"]

code_dir = find_dataset(CODE_DATASET_NAMES, "Code dataset")
data_dir = find_dataset(DATA_DATASET_NAMES, "Data dataset")

# ── 2. Thêm code dataset vào Python path ─────────────────────────────────────
# Kaggle đã extract → src/ nằm trực tiếp trong code_dir
# Ví dụ: /kaggle/input/r2ai-code/src/retrieval/batch_retrieve.py
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
    print(f"📌 Thêm vào sys.path: {code_dir}")

# Kiểm tra src/ có tồn tại không
src_in_code = os.path.join(code_dir, "src")
if not os.path.exists(src_in_code):
    # Fallback: có thể dataset chứa file ở cấp trên
    print(f"⚠️  Không thấy src/ trong {code_dir}, kiểm tra lại cấu trúc dataset:")
    print("  ", os.listdir(code_dir))
else:
    print(f"✅ Tìm thấy src/ tại: {src_in_code}")

# ── 3. Symlink database files (không copy, tiết kiệm disk) ───────────────────
db_working = os.path.join(WORKING, "database")
os.makedirs(db_working, exist_ok=True)

for fname in ["local_chunks.db", "local_chunks.index"]:
    src = os.path.join(data_dir, fname)
    dst = os.path.join(db_working, fname)
    if not os.path.exists(src):
        print(f"⚠️  Không tìm thấy {fname} trong data dataset")
        continue
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print(f"🔗 Symlink {fname} ({os.path.getsize(src)/1024**3:.2f} GB)")

# ── 4. Symlink file câu hỏi ───────────────────────────────────────────────────
question_src = os.path.join(data_dir, "R2AIStage1DATA.json")
question_dst = os.path.join(WORKING, "R2AIStage1DATA.json")
if os.path.exists(question_src):
    if os.path.islink(question_dst) or os.path.exists(question_dst):
        os.remove(question_dst)
    os.symlink(question_src, question_dst)
    print(f"🔗 Symlink R2AIStage1DATA.json ({os.path.getsize(question_src)/1024:.0f} KB)")
else:
    print("⚠️  Không tìm thấy R2AIStage1DATA.json")

# ── 5. Load BAAI/bge-reranker-v2-m3 ─────────────────────────────────────────
# Model nằm trong data dataset, symlink vào đúng chỗ retriever.py tìm kiếm
RERANKER_MODEL_NAME = "bge-reranker-v2-m3"

# retriever.py tìm model tại: os.path.join(os.path.dirname(__file__), "bge-reranker-v2-m3")
# → /kaggle/input/r2ai-code/src/retrieval/bge-reranker-v2-m3/
reranker_src = os.path.join(data_dir, RERANKER_MODEL_NAME)
reranker_dst = os.path.join(code_dir, "src", "retrieval", RERANKER_MODEL_NAME)

if os.path.exists(reranker_src):
    if os.path.islink(reranker_dst):
        os.remove(reranker_dst)
    elif os.path.isdir(reranker_dst):
        shutil.rmtree(reranker_dst)
    os.symlink(reranker_src, reranker_dst)
    print(f"🔗 Symlink BAAI reranker: {reranker_src} → {reranker_dst}")

    # Kiểm tra model weights tồn tại
    model_weights = os.path.join(reranker_src, "model.safetensors")
    if os.path.exists(model_weights):
        size_gb = os.path.getsize(model_weights) / 1024**3
        print(f"✅ Reranker model weights: {size_gb:.2f} GB — OK")
    else:
        print(f"⚠️  Không thấy model.safetensors trong {reranker_src}")
        print("   Nội dung:", os.listdir(reranker_src))
else:
    print(f"⚠️  Không tìm thấy {RERANKER_MODEL_NAME}/ trong data dataset")
    print("   Nội dung data dataset:", os.listdir(data_dir))

# ── 6. Test import nhanh ──────────────────────────────────────────────────────
print("\n🔍 Kiểm tra import...")
try:
    sys.path.insert(0, code_dir)
    from src.retrieval.retriever import LegalRetriever
    print("✅ LegalRetriever import OK")
except Exception as e:
    print(f"❌ Import lỗi: {e}")

# ── 7. Tóm tắt ───────────────────────────────────────────────────────────────
print("\n📂 Tóm tắt:")
print(f"  Code:     {src_in_code}")
print(f"  DB:       {db_working}")
print(f"  Reranker: {reranker_dst}")
print(f"  Input:    {question_dst}")
print("\n✅ Setup hoàn tất! Chạy Block 2 để bắt đầu inference.")
