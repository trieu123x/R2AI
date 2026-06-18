"""
test_chunking.py
Chay thu chunking tren subset nho (~5 docs) de danh gia chat luong.
Khong ghi vao DB that — in ra console va luu vao logs/chunk_test_report.txt
"""
import os
import sys
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
import glob
import pandas as pd
from pyvi import ViTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MAX_CHARS   = 800
TEST_N_DOCS = 5          # số document dùng để thử
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_PATH  = os.path.join(PROJECT_ROOT, "logs", "chunk_test_report.txt")

# ── Filters (giống process_chunks.py) ─────────────────────────────────────────
TARGET_SECTORS = ["Doanh nghiệp", "Lao động - Tiền lương", "Thuế - Phí - Lệ Phí", "Bảo hiểm", "Quyền dân sự"]
KEYWORDS = [
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "luật doanh nghiệp",
    "thuế thu nhập doanh nghiệp", "thuế giá trị gia tăng", "bảo hiểm xã hội",
    "hợp đồng lao động", "người lao động", "hộ kinh doanh",
    "đăng ký doanh nghiệp", "giải thể doanh nghiệp", "phá sản"
]
TARGET_DOC_NUMS = [
    "59/2020/QH14", "04/2017/QH14", "80/2021/NĐ-CP", "47/2021/NĐ-CP",
    "13/2008/QH12", "14/2008/QH12", "04/2007/QH12", "38/2019/QH14",
    "45/2019/QH14", "58/2014/QH13", "25/2008/QH12", "145/2020/NĐ-CP",
    "91/2015/QH13", "36/2005/QH11"
]
NORMATIVE_TYPES = [
    "Luật", "Bộ luật", "Nghị định", "Thông tư", "Thông tư liên tịch",
    "Nghị quyết", "Pháp lệnh", "Hiến pháp", "Văn bản hợp nhất"
]


# ── Load dữ liệu ──────────────────────────────────────────────────────────────
def load_documents(n=TEST_N_DOCS):
    print("[+] Loading metadata parquet...")
    meta_path = os.path.join(PROJECT_ROOT, "vietnamese-legal-documents", "metadata", "data-00000-of-00001.parquet")
    df_meta = pd.read_parquet(meta_path)

    cond_sector  = df_meta['legal_sectors'].fillna('').apply(lambda s: any(sec in s for sec in TARGET_SECTORS))
    cond_keyword = df_meta['title'].fillna('').apply(lambda t: any(kw in t.lower() for kw in KEYWORDS))
    cond_doc_num = df_meta['document_number'].isin(TARGET_DOC_NUMS)
    base_filtered = df_meta[cond_sector | cond_keyword | cond_doc_num]

    is_central = ~base_filtered['issuing_authority'].fillna('').str.contains(
        'Tỉnh|Thành phố|Thành Phố|UBND|HĐND|Huyện|Quận|Thị xã|Thị Xã|Cục|Sở|Chi cục|Ủy ban nhân dân|Hội đồng nhân dân',
        case=False, na=False
    )
    central_filtered = base_filtered[is_central]

    cond_normative = central_filtered['legal_type'].isin(NORMATIVE_TYPES)
    cond_target    = central_filtered['document_number'].isin(TARGET_DOC_NUMS)
    final_filtered = central_filtered[cond_normative | cond_target]
    filtered_ids   = set(final_filtered['id'].tolist())

    print(f"[+] Total qualified documents in metadata: {len(final_filtered)}")
    print("[+] Loading content parquets (first match only)...")

    content_files = sorted(glob.glob(
        os.path.join(PROJECT_ROOT, "vietnamese-legal-documents", "content", "*.parquet")
    ))
    content_rows = []
    for file_path in content_files:
        df_c = pd.read_parquet(file_path)
        matched = df_c[df_c['id'].isin(filtered_ids)]
        content_rows.append(matched)
        if sum(len(x) for x in content_rows) >= n:
            break

    df_all_content = pd.concat(content_rows, ignore_index=True)
    df_merged = pd.merge(final_filtered, df_all_content, on="id", how="inner")
    return df_merged.head(n)


# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_document(doc_id, doc_num, title, content, max_chars=MAX_CHARS):
    chunks = []
    if not content:
        return chunks

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    pattern  = r'\n(?=Điều \d+[\.:\s])'
    articles = re.split(pattern, content)

    for art_idx, art in enumerate(articles):
        art = art.strip()
        if not art:
            continue

        art_header_match = re.match(r'^(Điều \d+[\.:\s]*[^\n]*)', art)
        art_header = art_header_match.group(1) if art_header_match else f"Mục {art_idx}"

        if len(art) <= max_chars:
            prefix   = f"Văn bản: {title} ({doc_num}) | {art_header} | Nội dung: "
            full_text = prefix + art
            chunks.append({
                "article":    art_header,
                "content":    full_text,
                "raw_len":    len(art),
                "total_len":  len(full_text),
                "is_sub":     False,
                "sub_idx":    None,
            })
        else:
            paragraphs = art.split("\n")
            current    = ""
            sub_idx    = 1
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) > max_chars:
                    if current:
                        prefix    = f"Văn bản: {title} ({doc_num}) | {art_header} (Phần {sub_idx}) | Nội dung: "
                        full_text = prefix + current
                        chunks.append({
                            "article":   f"{art_header} (Phần {sub_idx})",
                            "content":   full_text,
                            "raw_len":   len(current),
                            "total_len": len(full_text),
                            "is_sub":    True,
                            "sub_idx":   sub_idx,
                        })
                        sub_idx += 1
                    current = para
                else:
                    current = (current + "\n" + para) if current else para

            if current:
                prefix    = f"Văn bản: {title} ({doc_num}) | {art_header} (Phần {sub_idx}) | Nội dung: "
                full_text = prefix + current
                chunks.append({
                    "article":   f"{art_header} (Phần {sub_idx})",
                    "content":   full_text,
                    "raw_len":   len(current),
                    "total_len": len(full_text),
                    "is_sub":    True,
                    "sub_idx":   sub_idx,
                })
    return chunks


# ── Đánh giá ─────────────────────────────────────────────────────────────────
def evaluate(all_chunks):
    lens = [c['total_len'] for c in all_chunks]
    raw  = [c['raw_len']  for c in all_chunks]

    print("\n" + "="*70)
    print("📊  CHUNKING QUALITY REPORT")
    print("="*70)
    print(f"  Tổng chunks                : {len(all_chunks):,}")
    print(f"  Chunk ngắn nhất (total)    : {min(lens):,} chars")
    print(f"  Chunk dài nhất  (total)    : {max(lens):,} chars")
    print(f"  Trung bình      (total)    : {sum(lens)//len(lens):,} chars")
    print(f"  Chunks quá dài (>MAX_CHARS): {sum(1 for c in all_chunks if c['raw_len'] > MAX_CHARS)}")
    print(f"  Chunks là sub-part         : {sum(1 for c in all_chunks if c['is_sub'])}")
    print(f"  Chunks là full article     : {sum(1 for c in all_chunks if not c['is_sub'])}")

    # Phân phối độ dài
    buckets = {"<200": 0, "200-400": 0, "400-600": 0, "600-800": 0, ">800": 0}
    for c in all_chunks:
        tl = c['total_len']
        if tl < 200:      buckets["<200"]     += 1
        elif tl < 400:    buckets["200-400"]  += 1
        elif tl < 600:    buckets["400-600"]  += 1
        elif tl <= 800:   buckets["600-800"]  += 1
        else:             buckets[">800"]      += 1

    print("\n  Phân phối độ dài (total chars):")
    for k, v in buckets.items():
        bar = "█" * (v * 30 // max(buckets.values()) if buckets.values() else 0)
        print(f"    {k:>10}: {v:>5}  {bar}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    df = load_documents(n=TEST_N_DOCS)
    print(f"\n[+] Testing on {len(df)} documents...\n")

    all_chunks  = []
    report_lines = []

    for _, row in df.iterrows():
        doc_id  = int(row['id'])
        doc_num = row['document_number'] if pd.notna(row['document_number']) else 'N/A'
        title   = row['title']
        content = row['content']

        chunks = chunk_document(doc_id, doc_num, title, content)
        all_chunks.extend(chunks)

        header = f"\n{'─'*70}\n📄  {title}\n    [{doc_num}] | {len(content):,} raw chars → {len(chunks)} chunks"
        print(header)
        report_lines.append(header)

        # In 3 chunk đầu của mỗi doc
        for i, ch in enumerate(chunks[:3]):
            seg = ViTokenizer.tokenize(ch['content'])
            block = (
                f"\n  ── Chunk {i+1}: {ch['article']} (raw={ch['raw_len']} | total={ch['total_len']} chars) ──\n"
                f"  [Raw]:\n  {ch['content'][:300]}{'...' if len(ch['content'])>300 else ''}\n"
                f"  [Segmented (first 200 chars)]:\n  {seg[:200]}...\n"
            )
            print(block)
            report_lines.append(block)

    evaluate(all_chunks)

    # Ghi report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write(f"\n\nTotal chunks across {len(df)} docs: {len(all_chunks)}\n")
    print(f"\n[+] Full report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
