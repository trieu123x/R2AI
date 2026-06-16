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

### 1. Cài Python packages

```bash
pip install sentence-transformers pandas pyarrow psycopg2-binary python-dotenv numpy pyvi
```

### 2. Cấu hình `.env`

Tạo file `.env` tại root (đã có sẵn, chỉ cần cập nhật nếu đổi project Supabase):

```env
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<anon_key>
SUPABASE_SERVICE_ROLE_KEY=<service_role_key>
DATABASE_URL=postgresql://postgres.<project>:<password>@<host>:5432/postgres
```

---

## 🚀 Hướng dẫn chạy theo thứ tự

### BƯỚC 1 — Kiểm tra chất lượng chunking (tuỳ chọn)

> Chạy trên ~5 văn bản mẫu, không ghi vào DB. Dùng để kiểm tra logic chunker trước.

```bash
python src/test_chunking.py
```

- **Input**: `vietnamese-legal-documents/metadata/` và `content/` (parquet files)
- **Output**: In ra console + `logs/chunk_test_report.txt`

---

### BƯỚC 2 — Chunking toàn bộ dữ liệu vào SQLite

> Lọc văn bản pháp lý liên quan (doanh nghiệp, thuế, lao động...), chunk theo điều khoản, lưu vào `database/local_chunks.db`.

```bash
python src/process_chunks.py
```

- **Input**: `vietnamese-legal-documents/` (parquet)
- **Output**: `database/local_chunks.db` (~3GB, có ~392,704 chunks)
- **Thời gian**: ~10–20 phút tuỳ máy

---

### BƯỚC 3 — Thiết lập FTS5 cho SQLite (offline search)

> Tạo virtual table FTS5 trong SQLite để hỗ trợ full-text search bằng BM25.

```bash
python retrieval/setup_fts5.py
```

- **Input**: `database/local_chunks.db` (từ Bước 2)
- **Output**: FTS5 table `chunks_fts` được tạo trong cùng file DB

---

### BƯỚC 4 — Sinh embeddings (vector) cho toàn bộ chunks

> Dùng model `bkai-foundation-models/vietnamese-bi-encoder` để sinh embeddings.  
> Hỗ trợ resume (tiếp tục nếu bị ngắt).

```bash
python database/generate_local_embeddings_mp.py
```

- **Input**: `database/local_chunks.db`
- **Output**: Cột `embedding` được điền vào các rows trong `document_chunks`
- **Thời gian**: Rất lâu trên CPU (~vài giờ). Chạy qua đêm nếu cần.

---

### BƯỚC 5 — Upload lên Supabase (online mode, tuỳ chọn)

> Upload toàn bộ chunks + embeddings từ SQLite lên Supabase PostgreSQL (cần internet).

```bash
python database/push_chunks_to_supabase.py
```

- **Input**: `database/local_chunks.db` + kết nối Supabase qua `.env`
- **Output**: Dữ liệu đồng bộ lên Supabase (bảng `documents` + `document_chunks`)
- **Lưu ý**: Đảm bảo đã chạy Bước 3 và 4 trước để có FTS5 + embeddings

---

### BƯỚC 6 — Kiểm tra retrieval pipeline

> CLI tương tác để test tìm kiếm mà không cần LLM.

**Chế độ offline (SQLite):**
```bash
python retrieval/test_retrieval.py --local
```

**Chế độ online (Supabase, tự fallback về local nếu mất mạng):**
```bash
python retrieval/test_retrieval.py
```

**Các tham số hữu ích:**
```bash
# Chỉ định mode tìm kiếm
python retrieval/test_retrieval.py --local --mode fts       # Full-text search (BM25)
python retrieval/test_retrieval.py --local --mode vector    # Vector similarity
python retrieval/test_retrieval.py --local --mode hybrid    # Hybrid RRF (mặc định)

# Tìm kiếm một câu hỏi cụ thể
python retrieval/test_retrieval.py --local --query "Điều kiện thành lập doanh nghiệp" --top-k 5

# Benchmark so sánh tốc độ 3 mode
python retrieval/test_retrieval.py --local --benchmark

# Export kết quả ra JSON
python retrieval/test_retrieval.py --local --query "Hợp đồng lao động" --export result.json
```

---

### BƯỚC 7 — Truy vấn hàng loạt (batch)

> Dùng khi cần test nhiều câu hỏi cùng lúc từ file JSON.

```bash
python retrieval/batch_retrieve.py
```

---

## 🔍 Sơ đồ pipeline

```
vietnamese-legal-documents/ (parquet)
         │
         ▼ Bước 2
    src/process_chunks.py
         │
         ▼
  database/local_chunks.db  ──────────────────────────────┐
         │                                                  │
         ▼ Bước 3                                          │
  retrieval/setup_fts5.py                                  │
  (tạo FTS5 table)                                         │
         │                                                  │
         ▼ Bước 4                                          │
  database/generate_local_embeddings_mp.py                 │
  (sinh vector embeddings)                                  │
         │                                                  │
         ▼ Bước 5 (tuỳ chọn)                              │
  database/push_chunks_to_supabase.py                      │
  (upload lên Supabase)                                     │
         │                                                  │
         ▼ Bước 6                                          ▼
  retrieval/test_retrieval.py ←── offline: local_retriever.py
                              ←── online:  retriever.py (Supabase)
```

---

## 📝 Ghi chú quan trọng

| File | Kích thước | Ghi chú |
|------|-----------|---------|
| `database/local_chunks.db` | ~3 GB | Không commit lên git (đã có trong `.gitignore`) |
| `vietnamese-legal-documents/` | ~vài GB | Dataset gốc, không commit |
| Model `vietnamese-bi-encoder` | ~500 MB | Tự động tải từ HuggingFace lần đầu |

- **Offline hoàn toàn**: Chỉ cần Bước 2–4 + Bước 6 với flag `--local`
- **Online mode**: Cần thêm Bước 5 và kết nối Supabase
- Hệ thống tự động **fallback** về SQLite nếu Supabase không kết nối được
