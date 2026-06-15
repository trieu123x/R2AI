import pandas as pd
import os
import glob

# Load metadata
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

output = []
output.append(f"Base filtered count (sectors & keywords & targets): {len(base_filtered)}")

# Let's check legal types in the base filtered set
types = base_filtered['legal_type'].value_counts()
output.append("\nLegal types in base filtered:")
for t, c in types.items():
    output.append(f"  - {t}: {c}")

# Let's filter out local government decisions
is_central = ~base_filtered['issuing_authority'].fillna('').str.contains(
    'Tỉnh|Thành phố|Thành Phố|UBND|HĐND|Huyện|Quận|Thị xã|Thị Xã|Cục|Sở|Chi cục|Ủy ban nhân dân|Hội đồng nhân dân', 
    case=False, 
    na=False
)
central_filtered = base_filtered[is_central]
output.append(f"\nCentral filtered count: {len(central_filtered)}")

# Let's look at the types and authorities of the central filtered set
output.append("\nTop issuing authorities in Central filtered:")
for auth, c in central_filtered['issuing_authority'].value_counts().head(20).items():
    output.append(f"  - {auth}: {c}")

# Let's estimate content size for the central filtered set
central_ids = set(central_filtered['id'].tolist())
content_files = sorted(glob.glob(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\*.parquet"))

matched_count = 0
total_bytes = 0

for file_path in content_files:
    df_content = pd.read_parquet(file_path)
    matched_df = df_content[df_content['id'].isin(central_ids)]
    matched_count += len(matched_df)
    total_bytes += matched_df['content'].fillna('').str.encode('utf-8').apply(len).sum()

output.append(f"\nCentral Filtered Match Count: {matched_count}")
output.append(f"Central Filtered Total Content Size: {total_bytes / (1024 * 1024):.2f} MB")

with open("filtering_options_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved output to filtering_options_output.txt")
