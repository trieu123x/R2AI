# R2AI — Legal RAG Assistant (Vietnamese)

Hệ thống RAG (Retrieval-Augmented Generation) dành cho văn bản pháp lý tiếng Việt, phục vụ cuộc thi **R2AI 2026**.  
Sử dụng **SQLite + FTS5** (offline) kết hợp **FAISS** (offline) để tra cứu văn bản pháp luật theo hybrid search (BM25 + vector) hoàn toàn cục bộ.

---

## 📁 Cấu trúc thư mục

```
R2AI/
│
├── .env                          # Biến môi trường
│
├── config/
│   └── config.py                 # Đọc .env, cung cấp class Config cho toàn project
│
├── src/                          # Mã nguồn hệ thống
│   ├── chunking/
│   │   ├── chunker.py            # Module chunking văn bản pháp lý (core logic)
│   │   └── process_chunks.py     # Script chính: lọc, chunk, lưu vào local SQLite
│   │
│   ├── embeddings/
│   │   └── embedder.py           # Module tạo vector embedding
│   │
│   ├── ingestion/
│   │   └── loader.py             # Module tải dữ liệu
│   │
│   ├── retrieval/
│   │   ├── batch_retrieve.py     # Truy vấn hàng loạt câu hỏi từ file JSON
│   │   ├── pipeline_retriever.py # Pipeline truy xuất kết hợp FTS5 + Vector
│   │   └── retriever.py          # Retriever cục bộ dùng SQLite + FTS5 + FAISS
│   │
│   └── vectordb/
│       └── vector_store.py       # Quản lý SQLite connection và FAISS index
│
├── database/                     # Chứa cơ sở dữ liệu và index cục bộ
│   └── generate_local_embeddings_mp.py   # Sinh embeddings bằng CPU (multiprocessing)
│
├── logs/                         # Output, báo cáo sinh ra khi chạy scripts
└── vietnamese-legal-documents/   # Dataset gốc (parquet) — không commit lên git
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

Yêu cầu Python 3.8 trở lên. Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```
*(Hoặc cài thủ công: `pip install sentence-transformers pandas pyarrow python-dotenv numpy pyvi faiss-cpu torch transformers accelerate`)*

### 3. Cấu hình `.env`

Tạo file `.env` tại thư mục gốc để lưu trữ các biến môi trường cấu hình (nếu có, ví dụ như `OPENAI_API_KEY` nếu bạn sử dụng OpenAI LLM API). Đối với chế độ chạy cục bộ hoàn toàn (offline mode), file `.env` không bắt buộc phải có đầy đủ các thông tin kết nối database online.

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
