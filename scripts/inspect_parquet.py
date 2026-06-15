import pandas as pd

try:
    print("Reading metadata parquet...")
    df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")
    print("Metadata columns:", df_meta.columns.tolist())
    print("Metadata shape:", df_meta.shape)
    print("Metadata head:")
    print(df_meta.head(2))
except Exception as e:
    print("Error reading metadata:", e)

try:
    print("\nReading content parquet...")
    df_content = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\data-00000-of-00011.parquet")
    print("Content columns:", df_content.columns.tolist())
    print("Content shape:", df_content.shape)
    print("Content head:")
    print(df_content.head(2))
except Exception as e:
    print("Error reading content:", e)
