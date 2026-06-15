import pandas as pd
import glob
import os
import re

# Load metadata
print("Loading metadata...")
df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")

target_sectors = ["Doanh nghiệp", "Lao động - Tiền lương", "Thuế - Phí - Lệ Phí", "Bảo hiểm", "Quyền dân sự"]
keywords = [
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "luật doanh nghiệp",
    "thuế thu nhập doanh nghiệp", "thuế giá trị gia tăng", "bảo hiểm xã hội",
    "hợp đồng lao động", "người lao động", "hộ kinh doanh",
    "đăng ký doanh nghiệp", "giải thể doanh nghiệp", "phá sản"
]

target_doc_nums = [
    "59/2020/QH14", "04/2017/QH14", "80/2021/NĐ-CP", "47/2021/NĐ-CP",
    "13/2008/QH12", "14/2008/QH12", "04/2007/QH12", "38/2019/QH14",
    "45/2019/QH14", "58/2014/QH13", "25/2008/QH12", "145/2020/NĐ-CP",
    "91/2015/QH13", "36/2005/QH11"
]

cond_sector = df_meta['legal_sectors'].fillna('').apply(lambda s: any(sec in s for sec in target_sectors))
cond_keyword = df_meta['title'].fillna('').apply(lambda t: any(kw in t.lower() for kw in keywords))
cond_doc_num = df_meta['document_number'].isin(target_doc_nums)

base_filtered = df_meta[cond_sector | cond_keyword | cond_doc_num]

is_central = ~base_filtered['issuing_authority'].fillna('').str.contains(
    'Tỉnh|Thành phố|Thành Phố|UBND|HĐND|Huyện|Quận|Thị xã|Thị Xã|Cục|Sở|Chi cục|Ủy ban nhân dân|Hội đồng nhân dân', 
    case=False, 
    na=False
)
central_filtered = base_filtered[is_central]

normative_types = [
    "Luật", "Bộ luật", "Nghị định", "Thông tư", "Thông tư liên tịch", 
    "Nghị quyết", "Pháp lệnh", "Hiến pháp", "Văn bản hợp nhất"
]

cond_normative = central_filtered['legal_type'].isin(normative_types)
cond_target = central_filtered['document_number'].isin(target_doc_nums)

final_filtered = central_filtered[cond_normative | cond_target]
filtered_ids = set(final_filtered['id'].tolist())

# Load content
content_files = sorted(glob.glob(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\*.parquet"))
content_dfs = []
print("Loading content...")
for file_path in content_files:
    df_chunk = pd.read_parquet(file_path)
    matched_chunk = df_chunk[df_chunk['id'].isin(filtered_ids)]
    content_dfs.append(matched_chunk)

df_all_content = pd.concat(content_dfs, ignore_index=True)
df_merged = pd.merge(final_filtered, df_all_content, on="id", how="inner")

print("Estimating chunks...")
total_raw_chars = 0
total_chunks = 0
total_chunk_chars = 0
max_chars = 800

for idx, row in df_merged.iterrows():
    title = row['title']
    doc_num = row['document_number'] if pd.notna(row['document_number']) else 'N/A'
    content = row['content']
    if not content:
        continue
    
    total_raw_chars += len(content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    
    pattern = r'\n(?=Điều \d+[\.:\s])'
    articles = re.split(pattern, content)
    
    for art_idx, art in enumerate(articles):
        art = art.strip()
        if not art:
            continue
            
        art_header_match = re.match(r'^(Điều \d+[\.:\s]*[^\n]*)', art)
        art_header = art_header_match.group(1) if art_header_match else f"Mục {art_idx}"
        
        context_prefix = f"Văn bản: {title} ({doc_num}) | {art_header} | Nội dung: "
        prefix_len = len(context_prefix)
        
        if len(art) <= max_chars:
            total_chunks += 1
            total_chunk_chars += prefix_len + len(art)
        else:
            paragraphs = art.split("\n")
            current_sub_chunk = ""
            sub_idx = 1
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_sub_chunk) + len(para) > max_chars:
                    if current_sub_chunk:
                        total_chunks += 1
                        total_chunk_chars += prefix_len + len(current_sub_chunk) + 15
                        sub_idx += 1
                    current_sub_chunk = para
                else:
                    current_sub_chunk = (current_sub_chunk + "\n" + para) if current_sub_chunk else para
            
            if current_sub_chunk:
                total_chunks += 1
                total_chunk_chars += prefix_len + len(current_sub_chunk) + 15
                
print("\n--- Chunking Estimation ---")
print(f"Total Raw Characters: {total_raw_chars:,}")
print(f"Total Chunks: {total_chunks:,}")
print(f"Total Chunk Characters (with prepended metadata): {total_chunk_chars:,}")
print(f"Estimated Text size in DB: {total_chunk_chars / (1024 * 1024):.2f} MB")
