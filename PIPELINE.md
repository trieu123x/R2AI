# RAG Pipeline Pháp Lý (R2AI)

Dự án R2AI xây dựng một hệ thống **Retrieval-Augmented Generation (RAG)** chuyên biệt, độ chính xác cao để giải đáp các thắc mắc về Luật pháp Việt Nam. Hệ thống tập trung giải quyết triệt để các rủi ro của AI sinh tạo (như ảo giác - hallucination, sai lệch căn cứ pháp lý) bằng cách kết hợp cơ chế Retrieval mạnh mẽ và Validation nghiêm ngặt.

---

## 🏛 Kiến trúc Pipeline Hệ thống (End-to-End)

Toàn bộ quy trình xử lý một câu hỏi đi qua 3 giai đoạn chính:

### Giai đoạn 1: Truy xuất tài liệu (Retrieval & Reranking)
Nhằm tìm ra các văn bản, điều luật chính xác nhất với câu hỏi từ cơ sở dữ liệu (Local SQLite / PostgreSQL).
- **Hybrid Search**: Chạy song song hai công cụ tìm kiếm:
  - *Tìm kiếm từ khóa (FTS5 / BM25)*: Đảm bảo không bỏ sót các thuật ngữ chuyên ngành và từ khóa chính xác.
  - *Tìm kiếm ngữ nghĩa (Vector Search)*: Bắt được ý định của người dùng thông qua mô hình Embedding (`bkai-bi-encoder`).
- **RRF (Reciprocal Rank Fusion)**: Thuật toán trộn và xếp hạng lại kết quả từ hai luồng tìm kiếm trên.
- **Cross-Encoder Reranking**: Đưa Top-K văn bản qua mô hình Reranker (`BAAI/bge-reranker-v2-m3`) để chấm điểm lại, hiểu sâu mối liên hệ giữa câu hỏi và từng đoạn văn bản, chắt lọc ra các tài liệu chuẩn xác nhất để đưa vào ngữ cảnh.

### Giai đoạn 2: Khởi tạo ngữ cảnh & Kiểm duyệt (Validation)
- **Trích xuất bằng chứng (Evidence Extraction)**: Xử lý cắt gọt các chunk văn bản dư thừa, tập trung vào đoạn text chứa số "Điều", "Khoản" liên quan trực tiếp (File `src/llm/llm_client.py`).
- **Validator**: Cỗ máy quét và kiểm duyệt nội dung (File `src/utils/validator.py`). Sử dụng Regex (`r"Điều\s+(\d+)"`) để thiết lập một hàng rào bảo vệ, đảm bảo rằng mọi "Điều luật" mà AI sinh ra **bắt buộc phải tồn tại** trong các văn bản Context đã cấp.

### Giai đoạn 3: Sinh đáp án an toàn (Generation)
Sử dụng các mô hình ngôn ngữ (như `Qwen-Instruct`) để tổng hợp và sinh lời giải với cơ chế phòng ngự nhiều lớp:
1. **Strict Prompting**: 
   - Hệ thống Prompt ép model xuất ra chuẩn định dạng **4 Phần**: (1) Trả lời trực tiếp, (2) Phân tích, (3) Căn cứ pháp lý, (4) Hạn chế.
   - Thiết lập các Ràng buộc phủ định (Negative Constraints) cực kỳ chặt chẽ: Cấm sử dụng Placeholder (`[X]`), cấm lặp ý, cấm chép lại Few-shot, cấm nhầm lẫn ngữ cảnh.
2. **Cơ chế Tự sửa lỗi (Auto-Retry & Self-Correction)**:
   - Câu trả lời của LLM được chuyển lại về `Validator`. Nếu LLM mắc bệnh "ảo giác" (sinh ra Điều không có trong context), hệ thống sẽ gài `warning_msg` vào Prompt và ép model tự nhận lỗi, sinh lại đáp án (Tối đa 2 lần Retry).
3. **Rule-based Fallback**:
   - Trong trường hợp cực đoan, nếu model liên tục thất bại sau các lần Retry, hệ thống sẽ bỏ qua LLM. Trả về đáp án Rule-based (được format tự động từ Context) để đảm bảo tính chính xác 100% về căn cứ pháp lý, tránh rủi ro cao.

---

## 🚀 Hướng dẫn chạy Pipeline (Generation)

### 1. Chạy bài Test cục bộ (Test Pipeline)
Sử dụng script test để chạy đánh giá nhanh trên 5 câu hỏi mẫu. Script sẽ tự động gọi luồng Batch Retrieve với chế độ Hybrid, Reranker và LLM đầy đủ.

```bash
python test_run.py
```
> *Lưu ý: Nếu không có kết nối Internet để tải file, hãy chắc chắn bạn đã tải model LLM (VD: `Qwen/Qwen1.5-4B-Chat`) và để HuggingFace ở chế độ offline.*

### 2. Chạy Pipeline hàng loạt (Batch Retrieval)
Đây là câu lệnh dùng để chạy toàn bộ bộ dữ liệu thi hoặc tạo file kết quả JSON nộp bài:

```bash
python src/retrieval/batch_retrieve.py \
    --input questions.json \
    --output results.json \
    --mode hybrid \
    --top-k 5 \
    --rerank \
    --llm \
    --llm-model "Qwen/Qwen1.5-4B-Chat"
```

**Các tham số chính:**
- `--input`: File JSON chứa danh sách câu hỏi.
- `--mode`: Chọn `fts`, `vector`, hoặc `hybrid` (Khuyên dùng `hybrid`).
- `--top-k`: Số lượng văn bản trả về để xử lý.
- `--rerank`: Bật mô hình chấm điểm lại độ chuẩn xác (PhoRanker / BGE).
- `--llm`: Bật quá trình sinh câu trả lời tự động bằng AI.
- `--llm-model`: Tên model Huggingface muốn dùng (VD: `Qwen/Qwen1.5-4B-Chat` hoặc `Qwen/Qwen2.5-0.5B-Instruct`).
- `--local`: Ưu tiên sử dụng database SQLite cục bộ.
