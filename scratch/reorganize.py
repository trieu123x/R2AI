import os
import shutil
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log(msg):
    print(f"[reorganize] {msg}", flush=True)

# 1. Create directories
dirs_to_create = [
    "src/ingestion",
    "src/chunking",
    "src/embeddings",
    "src/vectordb",
    "src/retrieval",
    "src/prompts",
    "src/llm",
    "src/utils"
]

for d in dirs_to_create:
    path = os.path.join(PROJECT_ROOT, d)
    os.makedirs(path, exist_ok=True)
    init_file = os.path.join(path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w", encoding="utf-8") as f:
            f.write("# Package initialization\n")
        log(f"Created __init__.py in {d}")

# Helper to read file content
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Helper to write file content
def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Helper to move folder if it exists
def move_folder(src, dst):
    src_path = os.path.join(PROJECT_ROOT, src)
    dst_path = os.path.join(PROJECT_ROOT, dst)
    if os.path.exists(src_path):
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.move(src_path, dst_path)
        log(f"Moved folder {src} -> {dst}")

# 2. Extract prompts to src/prompts/prompt_templates.py
log("Creating src/prompts/prompt_templates.py...")
prompt_templates_content = """# Prompt templates for R2AI generator and rewriter

SYSTEM_PROMPT = (
    "Bạn là một trợ lý pháp lý Việt Nam chuyên nghiệp, chính xác và đáng tin cậy.\\n\\n"
    "Nhiệm vụ của bạn:\\n"
    "- Trả lời câu hỏi dựa trên CÁC CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP. Tuyệt đối không tự suy diễn hoặc giả định các nội dung không có trong tài liệu.\\n"
    "- Hãy viết hoàn toàn bằng tiếng Việt chuẩn mực. Tuyệt đối KHÔNG trộn lẫn tiếng Trung (chữ Hán), tiếng Anh hoặc bất kỳ từ ngữ nước ngoài nào khác.\\n"
    "- Tuyệt đối KHÔNG được tự bịa ra các số hiệu văn bản, điều luật không có trong các căn cứ pháp lý.\\n"
    "- Tuyệt đối KHÔNG sử dụng tên văn bản, điều luật hoặc các từ ngữ trong Ví dụ mẫu (Few-shot) như 'Nghị định X', 'quyền công đoàn', 'thành lập công đoàn' để trả lời cho câu hỏi thực tế.\\n"
    "- Cực kỳ cẩn thận với các từ phủ định hoặc hành vi cấm: các cụm từ như 'nghiêm cấm', 'không được', 'xử phạt đối với hành vi' có nghĩa là hành vi đó BỊ CẤM, tuyệt đối không được viết thành 'người sử dụng lao động được phép thực hiện'.\\n"
    "- Không chỉ nêu tên văn bản chung chung. Hãy giải thích cụ thể nội dung quyền lợi, nghĩa vụ, ưu đãi hoặc mức phạt được quy định.\\n\\n"
    "Bạn BẮT BUỘC phải trình bày câu trả lời của mình nghiêm ngặt theo cấu trúc 4 phần sau (sử dụng chính xác các tiêu đề này làm tiêu đề dòng):\\n"
    "1. Trả lời trực tiếp: Trả lời trực tiếp vào câu hỏi, nêu rõ kết luận chính hoặc hành vi và hệ quả.\\n"
    "2. Phân tích chi tiết: Diễn giải chi tiết nội dung quy định, mức phạt bằng tiền cụ thể, quyền lợi/nghĩa vụ chi tiết được quy định trong các tài liệu tham khảo.\\n"
    "3. Căn cứ pháp lý: Liệt kê chi tiết các điều khoản, điều luật cụ thể được sử dụng làm căn cứ từ tài liệu tham khảo (Ví dụ: 'Điều 17 Bộ luật Lao động 2019', 'Điều 15 Nghị định 12/2022/NĐ-CP').\\n"
    "4. Hạn chế của dữ liệu (nếu có): Nêu rõ nếu các căn cứ pháp lý được cung cấp thiếu thông tin hoặc không đủ cơ sở để trả lời đầy đủ một khía cạnh nào đó của câu hỏi."
)

USER_CONTENT_TEMPLATE = (
    "TÀI LIỆU THAM KHẢO CUNG CẤP:\\n"
    "=========================================\\n"
    "{context}\\n"
    "=========================================\\n\\n"
    "{warning_msg}"
    "CÂU HỎI: {query}\\n\\n"
    "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên.\\n"
)

FEW_SHOT_USER = (
    "TÀI LIỆU THAM KHẢO CUNG CẤP:\\n"
    "=========================================\\n"
    "[Căn cứ 1]\\n"
    "Độ liên quan: 0.9876\\n"
    "Văn bản: Nghị định X năm 2022 về xử phạt hành chính\\n"
    "Điều: Điều 20\\n"
    "Nội dung:\\n"
    "Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập công đoàn.\\n"
    "=========================================\\n\\n"
    "CÂU HỎI: Hành vi cản trở người lao động thành lập công đoàn bị phạt bao nhiêu tiền?\\n\\n"
    "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên."
)

FEW_SHOT_ASSISTANT = (
    "1. Trả lời trực tiếp: Hành vi cản trở người lao động thành lập công đoàn sẽ bị xử phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng.\\n\\n"
    "2. Phân tích chi tiết:\\n"
    "- Về hành vi vi phạm: Người sử dụng lao động có hành vi cản trở người lao động thành lập công đoàn bị nghiêm cấm theo quy định.\\n"
    "- Về mức phạt: Hành vi vi phạm này sẽ bị xử phạt hành chính với mức phạt tiền cụ thể từ 10.000.000 đồng đến 20.000.000 đồng.\\n\\n"
    "3. Căn cứ pháp lý:\\n"
    "- Điều 20 Nghị định X năm 2022 về xử phạt hành chính.\\n\\n"
    "4. Hạn chế của dữ liệu (nếu có): Tài liệu được cung cấp chỉ đề cập đến mức phạt tiền hành chính đối với người sử dụng lao động, không nêu các hình thức xử phạt bổ sung khác hoặc trách nhiệm hình sự (nếu có)."
)
"""
write_file(os.path.join(PROJECT_ROOT, "src/prompts/prompt_templates.py"), prompt_templates_content)

# 3. Create src/embeddings/embedder.py
log("Creating src/embeddings/embedder.py...")
embedder_content = """import os
import time
import numpy as np

class VietnameseBiEncoder:
    def __init__(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[embedder] Loading bi-encoder on device '{device}'...", flush=True)
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name, device=device)
            print(f"[embedder] Model loaded in {time.time()-t0:.1f}s", flush=True)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([query], show_progress_bar=False)[0]
        return vec.astype(np.float32)
"""
write_file(os.path.join(PROJECT_ROOT, "src/embeddings/embedder.py"), embedder_content)

# 4. Create src/vectordb/vector_store.py
log("Creating src/vectordb/vector_store.py...")
vector_store_content = """import os
import time
import sqlite3
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

class VectorStore:
    def __init__(self, db_path: str = LOCAL_DB_PATH):
        self.db_path = db_path
        self._conn = None
        self._faiss_index = None

    def get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            try:
                cur = self._conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=OFF;")
                cur.execute("PRAGMA cache_size=-2000000;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA mmap_size=3000000000;")
            except Exception as e:
                print(f"[vector_store] Warning: Failed to apply SQLite PRAGMAs: {e}")
        return self._conn

    def get_faiss_index(self):
        import faiss
        index_path = os.path.join(os.path.dirname(self.db_path), "local_chunks.index")
        if self._faiss_index is None:
            if os.path.exists(index_path):
                t0 = time.time()
                self._faiss_index = faiss.read_index(index_path)
                print(f"[vector_store] FAISS index loaded in {time.time()-t0:.2f}s", flush=True)
            else:
                print(f"[vector_store] FAISS index not found. Building index...", flush=True)
                t0 = time.time()
                conn = self.get_conn()
                cur = conn.cursor()
                cur.execute("SELECT rowid, embedding FROM document_chunks WHERE embedding IS NOT NULL;")
                rowids = []
                embeddings = []
                for rowid, emb_bytes in cur:
                    if not emb_bytes:
                        continue
                    emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    rowids.append(rowid)
                    embeddings.append(emb)
                if not embeddings:
                    raise ValueError("No embeddings found in database!")
                embeddings = np.array(embeddings, dtype=np.float32)
                rowids = np.array(rowids, dtype=np.int64)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
                dim = embeddings.shape[1]
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIDMap(quantizer)
                index.add_with_ids(embeddings, rowids)
                faiss.write_index(index, index_path)
                self._faiss_index = index
                print(f"[vector_store] FAISS index built in {time.time()-t0:.2f}s", flush=True)
        return self._faiss_index

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
"""
write_file(os.path.join(PROJECT_ROOT, "src/vectordb/vector_store.py"), vector_store_content)

# Move setup_fts5.py to src/vectordb
fts5_src = os.path.join(PROJECT_ROOT, "retrieval/setup_fts5.py")
if os.path.exists(fts5_src):
    shutil.move(fts5_src, os.path.join(PROJECT_ROOT, "src/vectordb/setup_fts5.py"))
    log("Moved setup_fts5.py -> src/vectordb/setup_fts5.py")

# 5. Move chunking files
log("Moving chunking files...")
if os.path.exists(os.path.join(PROJECT_ROOT, "src/legal_chunker.py")):
    shutil.move(os.path.join(PROJECT_ROOT, "src/legal_chunker.py"), os.path.join(PROJECT_ROOT, "src/chunking/chunker.py"))
if os.path.exists(os.path.join(PROJECT_ROOT, "src/process_chunks.py")):
    shutil.move(os.path.join(PROJECT_ROOT, "src/process_chunks.py"), os.path.join(PROJECT_ROOT, "src/chunking/process_chunks.py"))
if os.path.exists(os.path.join(PROJECT_ROOT, "src/test_chunking.py")):
    shutil.move(os.path.join(PROJECT_ROOT, "src/test_chunking.py"), os.path.join(PROJECT_ROOT, "src/chunking/test_chunking.py"))

# 6. Move loader (inject.py) to src/ingestion/loader.py
log("Moving loader files...")
inject_src = os.path.join(PROJECT_ROOT, "retrieval/inject.py")
if os.path.exists(inject_src):
    shutil.move(inject_src, os.path.join(PROJECT_ROOT, "src/ingestion/loader.py"))

# Create ingestion test
test_ingestion_content = """# Basic test for ingestion
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def test_loader_exists():
    loader_path = os.path.join(PROJECT_ROOT, "src/ingestion/loader.py")
    assert os.path.exists(loader_path)
    print("Ingestion loader script exists and is located at:", loader_path)

if __name__ == "__main__":
    test_loader_exists()
"""
write_file(os.path.join(PROJECT_ROOT, "src/ingestion/test_ingestion.py"), test_ingestion_content)

# 7. Move utilities to src/utils
log("Moving utilities...")
utils_moves = [
    ("src/pipeline/context_compressor.py", "src/utils/context_compressor.py"),
    ("src/pipeline/validator.py", "src/utils/validator.py"),
    ("retrieval/down_model.py", "src/utils/down_model.py"),
    ("retrieval/update_reranker.py", "src/utils/update_reranker.py"),
    ("retrieval/update_flagreranker.py", "src/utils/update_flagreranker.py")
]
for src, dst in utils_moves:
    src_path = os.path.join(PROJECT_ROOT, src)
    if os.path.exists(src_path):
        shutil.move(src_path, os.path.join(PROJECT_ROOT, dst))

# 8. Create src/utils/test_utils.py
test_utils_content = """# Test for utilities
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.context_compressor import ContextCompressor
from src.utils.validator import PipelineValidator

def test_utils():
    compressor = ContextCompressor()
    validator = PipelineValidator()
    print("Successfully initialized ContextCompressor and PipelineValidator")

if __name__ == "__main__":
    test_utils()
"""
write_file(os.path.join(PROJECT_ROOT, "src/utils/test_utils.py"), test_utils_content)

# 9. Move model folder bge-reranker-v2-m3
move_folder("retrieval/bge-reranker-v2-m3", "src/retrieval/bge-reranker-v2-m3")

# 10. Re-write and move retrieval/retriever.py and retrieval/local_retriever.py
log("Reorganizing retrieval...")
retrievers = ["retriever.py", "local_retriever.py"]
for rname in retrievers:
    r_path = os.path.join(PROJECT_ROOT, "retrieval", rname)
    if os.path.exists(r_path):
        content = read_file(r_path)
        # Update references in content
        # Update self._model or SentenceTransformer calls to use VietnameseBiEncoder
        # But we can also just let it import and use our new modules or keep it working by updating imports
        # Replace local paths
        content = content.replace('os.path.join(PROJECT_ROOT, "database", "local_chunks.db")', 'os.path.join(PROJECT_ROOT, "database", "local_chunks.db")')
        content = content.replace('model_path = os.path.join(os.path.dirname(__file__), "bge-reranker-v2-m3")', 'model_path = os.path.join(os.path.dirname(__file__), "bge-reranker-v2-m3")')
        content = content.replace('from retrieval.retriever import', 'from src.retrieval.retriever import')
        
        # Write to src/retrieval
        write_file(os.path.join(PROJECT_ROOT, "src/retrieval", rname), content)
        os.remove(r_path)

# Move test_retrieval.py to src/retrieval
if os.path.exists(os.path.join(PROJECT_ROOT, "retrieval/test_retrieval.py")):
    shutil.move(os.path.join(PROJECT_ROOT, "retrieval/test_retrieval.py"), os.path.join(PROJECT_ROOT, "src/retrieval/test_retrieval.py"))

# 11. Reorganize LLM generator
log("Moving generator and rewrites...")
if os.path.exists(os.path.join(PROJECT_ROOT, "retrieval/qwen_generator.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "retrieval/qwen_generator.py"))
    # Let's replace the hardcoded prompt in qwen_generator with imports or keep it but update imports in file
    content = content.replace("from retrieval.retriever import", "from src.retrieval.retriever import")
    write_file(os.path.join(PROJECT_ROOT, "src/llm/llm_client.py"), content)
    os.remove(os.path.join(PROJECT_ROOT, "retrieval/qwen_generator.py"))

if os.path.exists(os.path.join(PROJECT_ROOT, "src/pipeline/generator.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "src/pipeline/generator.py"))
    write_file(os.path.join(PROJECT_ROOT, "src/llm/generator.py"), content)
    os.remove(os.path.join(PROJECT_ROOT, "src/pipeline/generator.py"))

# Move test_llm.py to src/llm
if os.path.exists(os.path.join(PROJECT_ROOT, "test_llm.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "test_llm.py"))
    content = content.replace("from retrieval.local_retriever import LocalRetriever", "from src.retrieval.local_retriever import LocalRetriever")
    content = content.replace("from retrieval.qwen_generator import QwenGenerator", "from src.llm.llm_client import QwenGenerator")
    write_file(os.path.join(PROJECT_ROOT, "src/llm/test_llm.py"), content)
    os.remove(os.path.join(PROJECT_ROOT, "test_llm.py"))

# 12. Move pipeline pieces to src/retrieval/
if os.path.exists(os.path.join(PROJECT_ROOT, "src/pipeline/query_rewrite.py")):
    shutil.move(os.path.join(PROJECT_ROOT, "src/pipeline/query_rewrite.py"), os.path.join(PROJECT_ROOT, "src/retrieval/query_rewrite.py"))
if os.path.exists(os.path.join(PROJECT_ROOT, "src/pipeline/reranker.py")):
    shutil.move(os.path.join(PROJECT_ROOT, "src/pipeline/reranker.py"), os.path.join(PROJECT_ROOT, "src/retrieval/reranker.py"))

# Combine PipelineRetriever into retriever.py wrapper or keep as separate file under src/retrieval
if os.path.exists(os.path.join(PROJECT_ROOT, "src/pipeline/retriever.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "src/pipeline/retriever.py"))
    content = content.replace("from retrieval.retriever import", "from src.retrieval.retriever import")
    write_file(os.path.join(PROJECT_ROOT, "src/retrieval/pipeline_retriever.py"), content)
    os.remove(os.path.join(PROJECT_ROOT, "src/pipeline/retriever.py"))

# 13. Move and update main_pipeline.py to src/main_pipeline.py
if os.path.exists(os.path.join(PROJECT_ROOT, "src/pipeline/main_pipeline.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "src/pipeline/main_pipeline.py"))
    # Update imports
    content = content.replace("from src.pipeline.query_rewrite import QueryRewriter", "from src.retrieval.query_rewrite import QueryRewriter")
    content = content.replace("from src.pipeline.retriever import PipelineRetriever", "from src.retrieval.pipeline_retriever import PipelineRetriever")
    content = content.replace("from src.pipeline.reranker import PipelineReranker", "from src.retrieval.reranker import PipelineReranker")
    content = content.replace("from src.pipeline.context_compressor import ContextCompressor", "from src.utils.context_compressor import ContextCompressor")
    content = content.replace("from src.pipeline.generator import PipelineGenerator", "from src.llm.generator import PipelineGenerator")
    content = content.replace("from src.pipeline.validator import PipelineValidator", "from src.utils.validator import PipelineValidator")
    write_file(os.path.join(PROJECT_ROOT, "src/main_pipeline.py"), content)
    os.remove(os.path.join(PROJECT_ROOT, "src/pipeline/main_pipeline.py"))

# 14. Move batch_retrieve.py to src/retrieval/batch_retrieve.py and update imports
if os.path.exists(os.path.join(PROJECT_ROOT, "retrieval/batch_retrieve.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "retrieval/batch_retrieve.py"))
    content = content.replace("from retrieval.retriever import LegalRetriever", "from src.retrieval.retriever import LegalRetriever")
    content = content.replace("from retrieval.qwen_generator import QwenGenerator", "from src.llm.llm_client import QwenGenerator")
    write_file(os.path.join(PROJECT_ROOT, "src/retrieval/batch_retrieve.py"), content)
    os.remove(os.path.join(PROJECT_ROOT, "retrieval/batch_retrieve.py"))

# 15. Update test_run.py to run src/retrieval/batch_retrieve.py
if os.path.exists(os.path.join(PROJECT_ROOT, "test_run.py")):
    content = read_file(os.path.join(PROJECT_ROOT, "test_run.py"))
    content = content.replace('retrieval/batch_retrieve.py', 'src/retrieval/batch_retrieve.py')
    write_file(os.path.join(PROJECT_ROOT, "test_run.py"), content)

# 16. Clean up directories that are now empty (except for pycache)
log("Cleaning up old directories...")
folders_to_clean = ["retrieval", "src/pipeline"]
for folder in folders_to_clean:
    path = os.path.join(PROJECT_ROOT, folder)
    if os.path.exists(path):
        # Remove any lingering files, pycache, etc.
        shutil.rmtree(path, ignore_errors=True)
        log(f"Removed old folder: {folder}")

# 17. Move other root level test/inspect scripts to scratch/
log("Moving root level scripts to scratch...")
root_files = os.listdir(PROJECT_ROOT)
for fname in root_files:
    if fname.endswith(".py") and fname not in ["test_run.py", "build_ipynb.py"]:
        src_path = os.path.join(PROJECT_ROOT, fname)
        dst_path = os.path.join(PROJECT_ROOT, "scratch", fname)
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)
            log(f"Moved root script {fname} -> scratch/{fname}")

# Move remaining files in retrieval/ if any
retrieval_dir = os.path.join(PROJECT_ROOT, "retrieval")
if os.path.exists(retrieval_dir):
    for fname in os.listdir(retrieval_dir):
        shutil.move(os.path.join(retrieval_dir, fname), os.path.join(PROJECT_ROOT, "scratch", fname))
    shutil.rmtree(retrieval_dir, ignore_errors=True)

log("Reorganization complete!")
