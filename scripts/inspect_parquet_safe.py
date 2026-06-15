import pandas as pd
import json

output = {}

try:
    df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")
    output["meta_columns"] = df_meta.columns.tolist()
    output["meta_shape"] = df_meta.shape
    output["meta_dtypes"] = {col: str(dtype) for col, dtype in df_meta.dtypes.items()}
    output["meta_sample"] = df_meta.head(5).to_dict(orient="records")
except Exception as e:
    output["meta_error"] = str(e)

try:
    df_content = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\data-00000-of-00011.parquet")
    output["content_columns"] = df_content.columns.tolist()
    output["content_shape"] = df_content.shape
    output["content_dtypes"] = {col: str(dtype) for col, dtype in df_content.dtypes.items()}
    output["content_sample"] = df_content.head(5).to_dict(orient="records")
except Exception as e:
    output["content_error"] = str(e)

with open("inspection_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Saved inspection output to inspection_output.json")
