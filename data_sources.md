# Nguồn Data cho R2AI2026 - AI Legal Assistant

## 1. Tổng quan yêu cầu

Cuộc thi yêu cầu hệ thống RAG xử lý các câu hỏi pháp luật về **doanh nghiệp vừa và nhỏ (SME)** tại Việt Nam, bao gồm:
- **Luật Doanh nghiệp**
- **Thuế**
- **Lao động**
- **Hợp đồng**

Ví dụ submission trong đề bài tiết lộ hai văn bản **bắt buộc phải có**:
- `04/2017/QH14` — Luật Hỗ trợ doanh nghiệp nhỏ và vừa
- `80/2021/NĐ-CP` — Nghị định hướng dẫn thi hành Luật Hỗ trợ DNV&N

---

## 2. Ràng buộc về Data

> ⚠️ **KHÔNG được sử dụng dữ liệu bên ngoài trong bất kỳ bước xử lý nào** (tức là không được gọi API ngoài lúc inference).

- **Được phép**: tự thu thập văn bản pháp luật offline, tải dataset từ HuggingFace về máy trước.
- **Không được phép**: query real-time ra ngoài internet trong pipeline đánh giá.

---

## 3. Dataset trên HuggingFace (Ưu tiên cao)

| Độ ưu tiên | Dataset | Link | Mô tả |
|:-----------:|---------|------|--------|
| ⭐⭐⭐ | `vohuutridung/vietnamese-legal-documents` | https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents | HTML→Markdown, phủ rộng nhất (nhiều loại văn bản từ TW đến địa phương) |
| ⭐⭐⭐ | `undertheseanlp/UTS_VLC` | https://huggingface.co/datasets/undertheseanlp/UTS_VLC | Hiến pháp, Bộ luật, Luật — **sạch nhất**, xác thực từ vbpl.vn |
| ⭐⭐ | `th1nhng0/vietnamese-legal-documents` | https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents | 500k+ tài liệu (luật, nghị định, thông tư, ...) |
| ⭐⭐ | `VLSP2025-LegalSML` | https://huggingface.co/datasets/VLSP2025-LegalSML | Tối ưu cho SLM <4B, legal reasoning, QA format |
| ⭐ | `duyet/vietnamese-legal-instruct` | https://huggingface.co/datasets/duyet/vietnamese-legal-instruct | Instruction QA format — dùng để fine-tune hoặc few-shot prompting |

---

## 4. Văn bản Pháp luật Bắt buộc trong Corpus

### 4.1 Nhóm Doanh nghiệp & SME
| Mã văn bản | Tên văn bản |
|------------|-------------|
| `59/2020/QH14` | Luật Doanh nghiệp 2020 |
| `04/2017/QH14` | Luật Hỗ trợ doanh nghiệp nhỏ và vừa |
| `80/2021/NĐ-CP` | Nghị định 80/2021 — hướng dẫn thi hành Luật Hỗ trợ DNV&N |
| `47/2021/NĐ-CP` | Nghị định hướng dẫn thi hành Luật Doanh nghiệp 2020 |

### 4.2 Nhóm Thuế
| Mã văn bản | Tên văn bản |
|------------|-------------|
| `13/2008/QH12` | Luật Thuế Giá trị gia tăng (GTGT) |
| `14/2008/QH12` | Luật Thuế Thu nhập doanh nghiệp (TNDN) |
| `04/2007/QH12` | Luật Thuế Thu nhập cá nhân (TNCN) |
| `38/2019/QH14` | Luật Quản lý thuế 2019 |

### 4.3 Nhóm Lao động & Bảo hiểm
| Mã văn bản | Tên văn bản |
|------------|-------------|
| `45/2019/QH14` | Bộ Luật Lao động 2019 |
| `58/2014/QH13` | Luật Bảo hiểm xã hội 2014 |
| `25/2008/QH12` | Luật Bảo hiểm y tế |
| `145/2020/NĐ-CP` | Nghị định hướng dẫn BLLĐ 2019 |

### 4.4 Nhóm Hợp đồng & Dân sự
| Mã văn bản | Tên văn bản |
|------------|-------------|
| `91/2015/QH13` | Bộ Luật Dân sự 2015 |
| `36/2005/QH11` | Luật Thương mại 2005 |

---

## 5. Nguồn Scraping Chính thống

| Nguồn | URL | Ghi chú |
|-------|-----|---------|
| CSDL Quốc gia Pháp luật | https://vbpl.vn | Nguồn gốc của mọi văn bản; full-text PDF + HTML |
| Thư viện Pháp luật | https://thuvienphapluat.vn | Full-text, dễ scrape |
| Luật Việt Nam | https://luatvietnam.vn | Có phân tích, tóm tắt |
| Pháp điển Việt Nam | https://phapdien.moj.gov.vn | Tra cứu văn bản đang còn hiệu lực, phân theo lĩnh vực |

---

## 6. Mô hình Embedding & LLM (tuân thủ quy định < 14B, trước 01/03/2026)

### Embedding Model (Vietnamese)
- `bkai-foundation-models/vietnamese-bi-encoder`
- `dangvantuan/vietnamese-embedding`
- `truro7/vn-law-embedding` _(chuyên biệt cho pháp luật VN)_

### LLM Generator (< 14B)
- `Qwen/Qwen2.5-7B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `vilm/vistral-7b-chat`
- `google/gemma-2-9b-it`

---

## 7. Chiến lược Pipeline

```
[Thu thập dữ liệu]
  └─ Download datasets HuggingFace (vohuutridung + UTS_VLC)
  └─ Scrape thêm văn bản còn thiếu từ vbpl.vn

[Tiền xử lý]
  └─ Filter: chỉ giữ lĩnh vực SME, Thuế, Lao động, Hợp đồng
  └─ Parse thành cấu trúc: DocumentID | DocumentName | ArticleName | Content
  └─ Chunk theo điều/khoản (không chunk ngang điều)

[Indexing]
  └─ Embed bằng Vietnamese embedding model
  └─ Index vào FAISS hoặc ChromaDB
  └─ (Tùy chọn) BM25 index song song để hybrid search

[Retrieval]
  └─ Query → Embedding → ANN Search (dense)
  └─ Query → BM25 Search (sparse)
  └─ Hybrid re-rank (RRF hoặc cross-encoder)

[Generation]
  └─ Top-K articles → Prompt → LLM (< 14B)
  └─ Output: answer + relevant_docs + relevant_articles
```

---

## 8. Từ khóa Lọc Data

Dùng các từ khóa sau để filter văn bản liên quan từ corpus lớn:

```
doanh nghiệp nhỏ và vừa
hỗ trợ doanh nghiệp
luật doanh nghiệp
thuế thu nhập doanh nghiệp
thuế giá trị gia tăng
bảo hiểm xã hội
hợp đồng lao động
người lao động
hộ kinh doanh
đăng ký doanh nghiệp
giải thể doanh nghiệp
phá sản
```

---

*Cập nhật: 2026-06-14*
