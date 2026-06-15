import pandas as pd
import os
import glob

# Load metadata
print("Loading metadata...")
df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")

# Define our filter
# Let's filter by target sectors or keywords in title
target_sectors = ["Doanh nghiệp", "Lao động - Tiền lương", "Thuế - Phí - Lệ Phí", "Bảo hiểm", "Quyền dân sự"]
keywords = [
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "luật doanh nghiệp",
    "thuế thu nhập doanh nghiệp", "thuế giá trị gia tăng", "bảo hiểm xã hội",
    "hợp đồng lao động", "người lao động", "hộ kinh doanh",
    "đăng ký doanh nghiệp", "giải thể doanh nghiệp", "phá sản"
]

# We can also add the exact document numbers
target_doc_nums = [
    "59/2020/QH14", "04/2017/QH14", "80/2021/NĐ-CP", "47/2021/NĐ-CP",
    "13/2008/QH12", "14/2008/QH12", "04/2007/QH12", "38/2019/QH14",
    "45/2019/QH14", "58/2014/QH13", "25/2008/QH12", "145/2020/NĐ-CP",
    "91/2015/QH13", "36/2005/QH11"
]

# Let's write a function to check match
# Option A: Sector matches
cond_sector = df_meta['legal_sectors'].fillna('').apply(lambda s: any(sec in s for sec in target_sectors))
# Option B: Title keyword matches
cond_keyword = df_meta['title'].fillna('').apply(lambda t: any(kw in t.lower() for kw in keywords))
# Option C: Document number matches
cond_doc_num = df_meta['document_number'].isin(target_doc_nums)

filtered_meta = df_meta[cond_sector | cond_keyword | cond_doc_num]
filtered_ids = set(filtered_meta['id'].tolist())

print(f"Filtered metadata count: {len(filtered_meta)}")

# Let's count how many matching contents exist and their total size in characters/bytes
content_files = sorted(glob.glob(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\*.parquet"))

matched_content_count = 0
total_content_bytes = 0

for file_path in content_files:
    print(f"Reading content file {os.path.basename(file_path)}...")
    df_content = pd.read_parquet(file_path)
    matched_df = df_content[df_content['id'].isin(filtered_ids)]
    matched_content_count += len(matched_df)
    total_content_bytes += matched_df['content'].fillna('').str.encode('utf-8').apply(len).sum()

print("\nFinal Stats:")
print(f"Filtered Metadata Count: {len(filtered_meta)}")
print(f"Matched Content Count: {matched_content_count}")
print(f"Total Content Size: {total_content_bytes / (1024 * 1024):.2f} MB")
