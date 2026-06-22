import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Read CSV and JSON
df_csv = pd.read_csv('citation_analysis_detail.csv')
results = json.load(open('results.json', encoding='utf-8'))
df_res = pd.DataFrame(results)

# Merge
df = pd.merge(df_csv, df_res, on='id')

# Target IDs or highest truly missing
sorted_df = df.dropna(subset=['orphan_docs_truly_missing']).copy()
# Split and count number of truly missing docs
sorted_df['n_missing'] = sorted_df['orphan_docs_truly_missing'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0)
sorted_df = sorted_df.sort_values(by='n_missing', ascending=False)

print(f"Total rows with truly missing orphans: {len(sorted_df)}")

# Print top 25
for idx, row in sorted_df.head(25).iterrows():
    print("=" * 80)
    print(f"ID: {row['id']} | Truly Missing: {row['orphan_docs_truly_missing']} (Total: {row['orphan_docs_total']})")
    print(f"Question: {row['question']}")
    print("-" * 40)
    print(f"Answer:\n{row['answer']}")
    print("=" * 80)
    print("\n")
