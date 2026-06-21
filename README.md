# R2AI — Legal RAG Assistant (Vietnamese)

Hệ thống RAG (Retrieval-Augmented Generation) dành cho văn bản pháp lý tiếng Việt, phục vụ cuộc thi **R2AI 2026**.  
Sử dụng **SQLite + FTS5** (offline) kết hợp **Supabase PostgreSQL + pgvector** (online) để tra cứu văn bản pháp luật theo hybrid search (BM25 + vector).

---

## 📁 Cấu trúc thư mục

```
R2AI/
│
├── .env                          # Biến môi trường (Supabase URL, DB URL, keys)
│
├── config/
│   └── config.py                 # Đọc .env, cung cấp class Config cho toàn project
│
├── src/                          # Xử lý dữ liệu thô → chunks
│   ├── legal_chunker.py          # Module chunking văn bản pháp lý (core logic)
│   ├── process_chunks.py         # Script chính: lọc, chunk, lưu vào local SQLite
│   └── test_chunking.py          # Kiểm tra chất lượng chunking trên ~5 docs mẫu
│
├── database/                     # Tạo embeddings + đồng bộ lên Supabase
│   ├── schema.sql                # Schema Supabase (documents + document_chunks)
│   ├── generate_local_embeddings_mp.py   # Sinh embeddings bằng CPU (multiprocessing)
│   └── push_chunks_to_supabase.py        # Upload chunks + embeddings lên Supabase
│
├── retrieval/                    # Pipeline truy vấn
│   ├── local_retriever.py        # Retriever offline dùng SQLite + FTS5 + vector
│   ├── retriever.py              # Retriever online dùng Supabase PostgreSQL
│   ├── batch_retrieve.py         # Truy vấn hàng loạt câu hỏi từ file JSON
│   ├── setup_fts5.py             # Tạo FTS5 virtual table trong SQLite
│   └── test_retrieval.py         # CLI tương tác test retrieval (không cần LLM)
│
├── logs/                         # Output, báo cáo sinh ra khi chạy scripts
├── data/                         # Dữ liệu thô (data/raw) và đã xử lý (data/processed)
└── vietnamese-legal-documents/   # Dataset gốc (parquet) — không commit lên git
    ├── metadata/                 # Metadata các văn bản pháp lý
    └── content/                  # Nội dung full-text các văn bản
```

---

## ⚙️ Cài đặt môi trường

### 1. Tạo và kích hoạt môi trường ảo (.venv)

Nên đặt tên thư mục môi trường ảo là `.venv` (tránh đặt tên trùng với file `.env` cấu hình để không bị ghi đè/nhầm lẫn).

* **Bước A: Tạo môi trường ảo**
  ```bash
  python -m venv .venv
  ```

* **Bước B: Kích hoạt môi trường ảo**
  * **Dành cho Command Prompt (CMD):**
    ```cmd
    .venv\Scripts\activate.bat
    ```
  * **Dành cho PowerShell:**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
    *Lưu ý:* Nếu gặp lỗi bảo mật (Execution Policy) trên PowerShell, chạy lệnh sau trước khi kích hoạt:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```

### 2. Cài đặt các thư viện (Python packages)

Sau khi đã kích hoạt môi trường ảo (bạn sẽ thấy `(.venv)` xuất hiện ở đầu dòng lệnh), tiến hành cài đặt các package:
Hệ thống RAG (Retrieval-Augmented Generation) chuyên sâu dành cho văn bản pháp lý tiếng Việt. Hệ thống được thiết kế để chống lại các lỗi rủi ro của AI (hallucination) nhờ cơ chế kiểm duyệt chặt chẽ, kết hợp tìm kiếm đa mô thức (Hybrid Search: BM25 + Vector) và chấm điểm lại (Reranking).

---

## ⚙️ Hướng dẫn cài đặt môi trường

### 1. Cài đặt Python packages
Yêu cầu Python 3.8 trở lên. Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```
*(Hoặc cài thủ công: `pip install sentence-transformers pandas pyarrow psycopg2-binary python-dotenv numpy pyvi faiss-cpu torch transformers accelerate`)*

### 3. Cấu hình `.env`

Tạo file `.env` tại root (đã có sẵn, chỉ cần cập nhật nếu đổi project Supabase):
### 2. Cấu hình biến môi trường (`.env`)
Tạo file `.env` tại thư mục gốc (nếu dùng cơ sở dữ liệu Supabase online). Nếu chỉ chạy local SQLite thì không bắt buộc, nhưng khuyến nghị:

```env
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<anon_key>
SUPABASE_SERVICE_ROLE_KEY=<service_role_key>
DATABASE_URL=postgresql://postgres.<project>:<password>@<host>:5432/postgres
```

---

## 🚀 Hướng dẫn tiền xử lý dữ liệu (Data Pipeline)

Trước khi hệ thống có thể trả lời câu hỏi, bạn cần xử lý dữ liệu thô và xây dựng Database tìm kiếm:

**Bước 1: Lọc, Chunking và lưu vào SQLite cục bộ**
Chạy lệnh dưới đây để cắt văn bản pháp lý thành các đoạn nhỏ (chunk) theo từng Điều khoản:
```bash
python src/process_chunks.py
```
*(Kết quả tạo ra `database/local_chunks.db` chứa dữ liệu văn bản)*

**Bước 2: Thiết lập Full-Text Search (FTS5)**
Tạo bảng tìm kiếm từ khóa ảo (Virtual Table) bên trong SQLite:
```bash
python retrieval/setup_fts5.py
```

**Bước 3: Sinh Vector Embeddings**
Mã hóa văn bản thành Vector ngữ nghĩa (có thể mất nhiều giờ nếu chạy CPU):
```bash
python database/generate_local_embeddings_mp.py
```

---

## 🧠 Hướng dẫn Chạy & Kiểm thử Hệ thống (RAG Pipeline)

### 1. Kiểm thử tìm kiếm cơ bản (Chưa dùng LLM)
Sử dụng CLI tương tác để kiểm tra xem hệ thống có tìm đúng tài liệu hay không (Offline mode):

```bash
python retrieval/test_retrieval.py --local --mode hybrid --query "Điều kiện thành lập doanh nghiệp" --top-k 5
```
Các tham số:
- `--mode`: Chọn `fts` (Từ khóa), `vector` (Ngữ nghĩa) hoặc `hybrid` (Kết hợp cả hai).
- `--benchmark`: Thêm flag này để đánh giá tốc độ giữa các phương pháp.

### 2. Kiểm thử luồng End-to-End (RAG: Tìm kiếm + AI sinh câu trả lời)
Chạy script test cục bộ với 5 câu hỏi mẫu. Hệ thống sẽ kết hợp Hybrid Search, Reranking, Validation và LLM (VD: Qwen) để đưa ra câu trả lời:

```bash
python test_run.py
```
*Lưu ý: Bạn có thể sửa `test_run.py` để đổi model LLM (Ví dụ: `Qwen/Qwen1.5-4B-Chat` hoặc `Qwen/Qwen2.5-0.5B-Instruct`). Lần đầu chạy sẽ cần tải model từ HuggingFace.*

### 3. Chạy hàng loạt để lấy kết quả (Batch Retrieval)
Để tạo file `results.json` nộp bài hoặc đánh giá toàn diện trên tệp câu hỏi lớn:

```bash
python src/retrieval/batch_retrieve.py \
    --input test_questions.json \
    --output test_results.json \
    --mode hybrid \
    --top-k 5 \
    --rerank \
    --llm \
    --llm-model "Qwen/Qwen1.5-4B-Chat"
```

---

## 🛡 Kiến trúc & Cơ chế phòng vệ của Pipeline

Dự án R2AI sử dụng **RAG Pipeline 3 Giai Đoạn** để đảm bảo tính pháp lý tuyệt đối:

1. **Retrieval & Reranking**: Kết hợp FTS5 và Vector (`bkai-bi-encoder`), sau đó dùng Reranker (`bge-reranker-v2-m3`) để đưa các điều luật liên quan nhất lên đầu.
2. **Validator (Rào chắn ảo giác)**: Bất kỳ câu trả lời nào của AI sinh ra đều được quét Regex. Nếu AI tự bịa ra "Điều luật" không tồn tại trong tài liệu cung cấp, hệ thống lập tức chặng lại.
3. **Safe Generation**: 
   - LLM bị giới hạn bởi Prompt cực kỳ khắt khe (Cấm dùng Placeholder, cấm lặp lại nội dung).
   - Nếu bị Validator chặn, LLM sẽ tự động **Retry** (nhận cảnh báo và sửa lỗi).
   - Nếu LLM vẫn vi phạm, hệ thống tự động **Fallback** sang tạo câu trả lời tĩnh (Rule-based) từ văn bản gốc, đảm bảo không bao giờ cung cấp thông tin sai lệch cho người dùng.

*(Xem chi tiết phân tích luồng tại file `PIPELINE.md`)*
