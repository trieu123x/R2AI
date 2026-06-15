import pandas as pd
import json

df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")

output = []
output.append(f"Total metadata rows: {len(df_meta)}")

# 1. Check specific document numbers listed in data_sources.md
target_docs = [
    "59/2020/QH14", "04/2017/QH14", "80/2021/NĐ-CP", "47/2021/NĐ-CP",
    "13/2008/QH12", "14/2008/QH12", "04/2007/QH12", "38/2019/QH14",
    "45/2019/QH14", "58/2014/QH13", "25/2008/QH12", "145/2020/NĐ-CP",
    "91/2015/QH13", "36/2005/QH11"
]

matched_by_number = df_meta[df_meta['document_number'].isin(target_docs)]
output.append(f"Matched by exact document number ({len(target_docs)} target docs): {len(matched_by_number)}")
for idx, row in matched_by_number.iterrows():
    output.append(f"  - {row['id']} | {row['document_number']}: {row['title']}")

# 2. Check by keywords in title
keywords = [
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "luật doanh nghiệp",
    "thuế thu nhập doanh nghiệp", "thuế giá trị gia tăng", "bảo hiểm xã hội",
    "hợp đồng lao động", "người lao động", "hộ kinh doanh",
    "đăng ký doanh nghiệp", "giải thể doanh nghiệp", "phá sản"
]

keyword_matches = {}
all_keyword_indices = set()
for kw in keywords:
    matches = df_meta[df_meta['title'].str.contains(kw, case=False, na=False)]
    keyword_matches[kw] = len(matches)
    all_keyword_indices.update(matches.index.tolist())

output.append("\nKeyword matches in title:")
for kw, count in keyword_matches.items():
    output.append(f"  - '{kw}': {count}")
output.append(f"Total unique documents matching keywords in title: {len(all_keyword_indices)}")

# 3. Analyze legal_sectors
sectors = df_meta['legal_sectors'].dropna().str.split(', ').explode().value_counts()
output.append("\nTop 30 legal sectors:")
for sector, count in sectors.head(30).items():
    output.append(f"  - {sector}: {count}")

with open("analysis_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved output to analysis_output.txt")
