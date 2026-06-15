import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import glob
import time
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from config import Config

def main():
    # 1. Validate Configuration
    try:
        Config.validate()
    except Exception as e:
        print(f"[-] Configuration error: {e}")
        return

    # 2. Connect to database and initialize schema
    print("[+] Connecting to Supabase PostgreSQL database...")
    try:
        conn = psycopg2.connect(Config.DATABASE_URL)
        cursor = conn.cursor()
        print("[+] Connected successfully.")
    except Exception as e:
        print(f"[-] Database connection failed: {e}")
        return

    print("[+] Initializing database schema from schema.sql...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        cursor.execute(schema_sql)
        conn.commit()
        print("[+] Schema initialized successfully (pgvector enabled, tables created).")
    except Exception as e:
        print(f"[-] Failed to run schema.sql: {e}")
        conn.rollback()
        conn.close()
        return

    # 3. Load and filter metadata
    print("[+] Loading metadata from parquet...")
    try:
        df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")
        print(f"[+] Loaded {len(df_meta)} metadata records.")
    except Exception as e:
        print(f"[-] Failed to read metadata parquet: {e}")
        conn.close()
        return

    # Apply filters to keep only relevant SME/tax/labor/contracts central regulations
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

    # Keep only central government authorities
    is_central = ~base_filtered['issuing_authority'].fillna('').str.contains(
        'Tỉnh|Thành phố|Thành Phố|UBND|HĐND|Huyện|Quận|Thị xã|Thị Xã|Cục|Sở|Chi cục|Ủy ban nhân dân|Hội đồng nhân dân', 
        case=False, 
        na=False
    )
    central_filtered = base_filtered[is_central]

    # Keep only normative document types OR target document numbers
    normative_types = [
        "Luật", "Bộ luật", "Nghị định", "Thông tư", "Thông tư liên tịch", 
        "Nghị quyết", "Pháp lệnh", "Hiến pháp", "Văn bản hợp nhất"
    ]
    cond_normative = central_filtered['legal_type'].isin(normative_types)
    cond_target = central_filtered['document_number'].isin(target_doc_nums)

    final_filtered_meta = central_filtered[cond_normative | cond_target]
    filtered_ids = set(final_filtered_meta['id'].tolist())
    print(f"[+] Filtered metadata: {len(final_filtered_meta)} records.")

    # 4. Load matching content
    content_files = sorted(glob.glob(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\*.parquet"))
    content_dfs = []
    print("[+] Loading matching content from content parquets...")
    
    for file_path in content_files:
        filename = os.path.basename(file_path)
        print(f"    Reading {filename}...")
        df_chunk = pd.read_parquet(file_path)
        matched_chunk = df_chunk[df_chunk['id'].isin(filtered_ids)]
        content_dfs.append(matched_chunk)
        print(f"    Matched {len(matched_chunk)} records from {filename}.")

    df_all_content = pd.concat(content_dfs, ignore_index=True)
    print(f"[+] Total matching content loaded: {len(df_all_content)} records.")

    # 5. Merge metadata and content
    print("[+] Merging metadata and content...")
    df_merged = pd.merge(final_filtered_meta, df_all_content, on="id", how="inner")
    print(f"[+] Merged dataset size: {len(df_merged)} records.")

    # 6. Upload merged records in batches
    print("[+] Uploading documents to Supabase...")
    
    # We clear the existing table first to prevent duplicate primary key errors during development
    try:
        cursor.execute("TRUNCATE TABLE documents CASCADE;")
        conn.commit()
        print("[+] Cleared existing records in documents table.")
    except Exception as e:
        print(f"[-] Warning: Failed to truncate table: {e}")
        conn.rollback()

    insert_query = """
    INSERT INTO documents (
        id, document_number, title, url, legal_type, 
        legal_sectors, issuing_authority, issuance_date, signers, content
    ) VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        document_number = EXCLUDED.document_number,
        title = EXCLUDED.title,
        url = EXCLUDED.url,
        legal_type = EXCLUDED.legal_type,
        legal_sectors = EXCLUDED.legal_sectors,
        issuing_authority = EXCLUDED.issuing_authority,
        issuance_date = EXCLUDED.issuance_date,
        signers = EXCLUDED.signers,
        content = EXCLUDED.content;
    """

    # Prepare data tuple list
    data_to_insert = []
    for _, row in df_merged.iterrows():
        # Handle nan/null values
        doc_id = int(row['id'])
        doc_num = str(row['document_number']) if pd.notna(row['document_number']) else None
        title = str(row['title']) if pd.notna(row['title']) else "No Title"
        url = str(row['url']) if pd.notna(row['url']) else None
        legal_type = str(row['legal_type']) if pd.notna(row['legal_type']) else None
        sectors = str(row['legal_sectors']) if pd.notna(row['legal_sectors']) else None
        authority = str(row['issuing_authority']) if pd.notna(row['issuing_authority']) else None
        date = str(row['issuance_date']) if pd.notna(row['issuance_date']) else None
        signers = str(row['signers']) if pd.notna(row['signers']) else None
        content = str(row['content']) if pd.notna(row['content']) else None

        data_to_insert.append((
            doc_id, doc_num, title, url, legal_type,
            sectors, authority, date, signers, content
        ))

    batch_size = 100
    total_records = len(data_to_insert)
    start_time = time.time()

    for idx in range(0, total_records, batch_size):
        batch = data_to_insert[idx : idx + batch_size]
        try:
            execute_values(cursor, insert_query, batch)
            conn.commit()
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + len(batch))
            remaining = avg_time * (total_records - (idx + len(batch)))
            print(f"    Uploaded {idx + len(batch)}/{total_records} ({(idx + len(batch))/total_records*100:.1f}%) | Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s")
        except Exception as e:
            print(f"[-] Error inserting batch starting at index {idx}: {e}")
            conn.rollback()
            # If batch upload fails, try inserting individual records in the batch to locate error
            print("    Attempting fallback to row-by-row insert for this batch...")
            for row in batch:
                try:
                    cursor.execute("""
                    INSERT INTO documents (
                        id, document_number, title, url, legal_type, 
                        legal_sectors, issuing_authority, issuance_date, signers, content
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document_number = EXCLUDED.document_number,
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        legal_type = EXCLUDED.legal_type,
                        legal_sectors = EXCLUDED.legal_sectors,
                        issuing_authority = EXCLUDED.issuing_authority,
                        issuance_date = EXCLUDED.issuance_date,
                        signers = EXCLUDED.signers,
                        content = EXCLUDED.content;
                    """, row)
                    conn.commit()
                except Exception as row_err:
                    print(f"    [-] Failed to insert record ID {row[0]}: {row_err}")
                    conn.rollback()

    cursor.close()
    conn.close()
    print(f"\n[+] Finished uploading. Total uploaded: {total_records} documents.")
    print(f"[+] Total duration: {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
