"""
push_to_local_db.py
===================
Đẩy dữ liệu văn bản pháp luật và chunks (kèm embeddings) từ SQLite/Parquet local
vào cơ sở dữ liệu PostgreSQL 'law_vn' trên localhost.
"""

import os
import sys
import io
import time
import queue
import threading
import sqlite3
import argparse
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "database", "schema.sql")
BATCH_SIZE = 5000

def connect_pg(host, port, user, password, dbname):
    print(f"[+] Connecting to PostgreSQL at {host}:{port}/{dbname}...", flush=True)
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=dbname,
        connect_timeout=10,
        options="-c statement_timeout=0"
    )
    conn.autocommit = False
    print("[+] Connected successfully.", flush=True)
    return conn

def ensure_schema(pg_conn):
    print("[+] Verifying schema...", flush=True)
    cursor = pg_conn.cursor()
    
    # Check if vector extension can be enabled
    has_vector = False
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        pg_conn.commit()
        has_vector = True
        print("[+] Extension 'vector' enabled successfully.", flush=True)
    except Exception as e:
        pg_conn.rollback()
        print("[!] Extension 'vector' is not available. Embedding columns will use REAL[] fallback.", flush=True)
        
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    if not has_vector:
        # Fallback: replace vector(768) with real[]
        schema_sql = schema_sql.replace("embedding vector(768)", "embedding real[]")
        # Remove CREATE EXTENSION since it failed
        schema_sql = schema_sql.replace("CREATE EXTENSION IF NOT EXISTS vector;", "-- CREATE EXTENSION IF NOT EXISTS vector;")
        
    try:
        cursor.execute(schema_sql)
        pg_conn.commit()
        print("[+] Schema verified / initialized.", flush=True)
    except Exception as e:
        pg_conn.rollback()
        print(f"[!] Schema initialization failed: {e}", flush=True)
        raise e
    finally:
        cursor.close()

def get_unique_doc_ids_from_sqlite(sqlite_conn):
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT DISTINCT document_id FROM document_chunks;")
    doc_ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    print(f"[+] Local chunks database contains chunks for {len(doc_ids):,} unique documents.", flush=True)
    return doc_ids

def load_and_insert_documents(pg_conn, doc_ids):
    print("[+] Loading document metadata from local parquet...", flush=True)
    meta_path = r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet"
    df_meta = pd.read_parquet(meta_path)
    df_meta = df_meta[df_meta['id'].isin(doc_ids)]
    
    import glob
    content_files = sorted(glob.glob(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\*.parquet"))
    content_dfs = []
    print("[+] Loading local document content parquets...", flush=True)
    for file_path in content_files:
        df_chunk = pd.read_parquet(file_path)
        matched_chunk = df_chunk[df_chunk['id'].isin(doc_ids)]
        content_dfs.append(matched_chunk)
        
    df_all_content = pd.concat(content_dfs, ignore_index=True)
    df_merged = pd.merge(df_meta, df_all_content, on="id", how="inner")
    
    print(f"[+] Found {len(df_merged):,} documents matching chunks in SQLite.", flush=True)
    
    print("[+] Inserting documents into PostgreSQL documents table...", flush=True)
    cursor = pg_conn.cursor()
    
    insert_sql = """
        INSERT INTO documents (id, document_number, title, url, legal_type, legal_sectors, issuing_authority, issuance_date, signers, content)
        VALUES %s
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
    
    records = []
    for idx, row in df_merged.iterrows():
        records.append((
            int(row['id']),
            row.get('document_number'),
            row.get('title'),
            row.get('url'),
            row.get('legal_type'),
            row.get('legal_sectors'),
            row.get('issuing_authority'),
            row.get('issuance_date'),
            row.get('signers'),
            row.get('content')
        ))
        
    # Batch insertion to avoid extremely large queries
    DOC_BATCH_SIZE = 200
    total_docs = len(records)
    print(f"[+] Inserting {total_docs:,} documents in batches of {DOC_BATCH_SIZE}...", flush=True)
    
    for i in range(0, total_docs, DOC_BATCH_SIZE):
        batch = records[i:i+DOC_BATCH_SIZE]
        execute_values(cursor, insert_sql, batch)
        pg_conn.commit()
        print(f"    Inserted {min(i + DOC_BATCH_SIZE, total_docs):>5,}/{total_docs:,} documents...", flush=True)
        
    cursor.close()
    print(f"[+] Successfully inserted/updated {len(df_merged):,} documents.", flush=True)
    return set(df_merged['id'].tolist())

def drop_indexes(pg_conn):
    print("[+] Dropping indexes to accelerate bulk upload...", flush=True)
    cursor = pg_conn.cursor()
    cursor.execute("DROP INDEX IF EXISTS idx_document_chunks_document_id;")
    cursor.execute("DROP INDEX IF EXISTS idx_document_chunks_fts;")
    pg_conn.commit()
    cursor.close()

def recreate_indexes(pg_conn):
    print("[+] Recreating indexes after bulk upload...", flush=True)
    cursor = pg_conn.cursor()
    
    start = time.time()
    print("  Creating idx_document_chunks_document_id...", flush=True)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);")
    pg_conn.commit()
    print(f"  idx_document_chunks_document_id created in {time.time() - start:.1f}s.", flush=True)
    
    start = time.time()
    print("  Creating idx_document_chunks_fts (GIN)...", flush=True)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_fts ON document_chunks USING gin(to_tsvector('simple', content));")
    pg_conn.commit()
    print(f"  idx_document_chunks_fts created in {time.time() - start:.1f}s.", flush=True)
    
    cursor.close()
    print("[+] Indexes recreated successfully.", flush=True)

def clear_chunks_table(pg_conn):
    cursor = pg_conn.cursor()
    print("[+] Truncating existing document_chunks table...", flush=True)
    cursor.execute("TRUNCATE TABLE document_chunks CASCADE;")
    pg_conn.commit()
    cursor.close()

def upload_worker(q, host, port, user, password, dbname, existing_doc_ids, results, thread_id):
    conn = None
    try:
        conn = connect_pg(host, port, user, password, dbname)
        cursor = conn.cursor()
        insert_sql = """
            INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
            VALUES %s
            ON CONFLICT DO NOTHING;
        """
        while True:
            batch = q.get()
            if batch is None:
                q.task_done()
                break
                
            filtered_batch = []
            for doc_id, chunk_idx, content, emb_blob in batch:
                if doc_id not in existing_doc_ids:
                    continue
                
                emb_list = None
                if emb_blob:
                    emb_list = np.frombuffer(emb_blob, dtype=np.float32).tolist()
                    
                filtered_batch.append((doc_id, chunk_idx, content, emb_list))
            
            if filtered_batch:
                try:
                    execute_values(cursor, insert_sql, filtered_batch)
                    conn.commit()
                    results.append(len(filtered_batch))
                except Exception as e:
                    print(f"\n[Thread-{thread_id}] Batch insert failed: {e}. Trying fallback row-by-row...", flush=True)
                    conn.rollback()
                    # Fallback row-by-row
                    row_sql = """
                        INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """
                    ok = fail = 0
                    last_err = None
                    for row in filtered_batch:
                        try:
                            cursor.execute(row_sql, row)
                            conn.commit()
                            ok += 1
                        except Exception as re:
                            conn.rollback()
                            fail += 1
                            last_err = re
                    if fail > 0:
                        print(f"[Thread-{thread_id}] Row-by-row failed {fail} times. Last error: {last_err}", flush=True)
                    results.append(ok)
            q.task_done()
    except Exception as e:
        print(f"\n[Thread-{thread_id}] Connection error: {e}", flush=True)
    finally:
        if conn:
            conn.close()

def upload_chunks(sqlite_conn, host, port, user, password, dbname, existing_doc_ids, total_chunks, num_threads=4):
    print(f"\n[+] Starting parallel upload of chunks — threads={num_threads}, batch_size={BATCH_SIZE}, total={total_chunks:,}", flush=True)
    print("-" * 60, flush=True)

    q = queue.Queue(maxsize=num_threads * 2)
    results = []
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=upload_worker, 
            args=(q, host, port, user, password, dbname, existing_doc_ids, results, i)
        )
        t.start()
        threads.append(t)

    sqlite_cursor = sqlite_conn.cursor()
    select_sql = "SELECT document_id, chunk_index, content, embedding FROM document_chunks ORDER BY document_id, chunk_index"
    sqlite_cursor.execute(select_sql)

    start_time = time.time()
    offset = 0
    log_every = 20000
    last_log_offset = 0

    while True:
        rows = sqlite_cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break
        
        q.put(rows)
        offset += len(rows)
        
        if offset - last_log_offset >= log_every:
            uploaded = sum(results)
            elapsed = time.time() - start_time
            rate = offset / elapsed if elapsed > 0 else 0
            remaining = (total_chunks - offset) / rate if rate > 0 else 0
            pct = offset / total_chunks * 100
            print(
                f"  Queued {offset:>8,} | Uploaded {uploaded:>8,} / {total_chunks:,} ({pct:5.1f}%) | "
                f"{rate:>5.0f} chunks/s | ETA {remaining:>5.0f}s",
                flush=True
            )
            last_log_offset = offset

    for _ in range(num_threads):
        q.put(None)

    for t in threads:
        t.join()

    sqlite_cursor.close()
    
    uploaded = sum(results)
    skipped = total_chunks - uploaded
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"[+] Chunk upload completed!")
    print(f"    Chunks uploaded : {uploaded:,}")
    print(f"    Chunks skipped  : {skipped:,}")
    print(f"    Duration        : {elapsed:.1f}s  ({uploaded/elapsed:.0f} chunks/s avg)")
    print("="*60)
    return uploaded

def verify_upload(pg_conn):
    cursor = pg_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents;")
    doc_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM document_chunks;")
    chunk_count = cursor.fetchone()[0]
    cursor.close()
    print(f"\n[+] Verification: PostgreSQL now has {doc_count:,} documents and {chunk_count:,} chunks.")

def main():
    parser = argparse.ArgumentParser(description="Push local_chunks.db to local PostgreSQL.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", required=True, help="PostgreSQL password")
    parser.add_argument("--dbname", default="law_vn")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(LOCAL_DB_PATH):
        print(f"[-] Local SQLite database not found at: {LOCAL_DB_PATH}")
        return

    # 1. Open SQLite Connection
    sqlite_conn = sqlite3.connect(LOCAL_DB_PATH)
    
    # 2. Get total chunks count
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("SELECT COUNT(*) FROM document_chunks;")
    total_chunks = sqlite_cursor.fetchone()[0]
    sqlite_cursor.close()
    print(f"[+] Local SQLite contains {total_chunks:,} chunks.")

    # 3. Connect to PostgreSQL
    pg_conn = None
    try:
        pg_conn = connect_pg(args.host, args.port, args.user, args.password, args.dbname)
        
        # 4. Initialize schema
        ensure_schema(pg_conn)
        
        # 5. Get unique document IDs from SQLite
        doc_ids = get_unique_doc_ids_from_sqlite(sqlite_conn)
        
        # 6. Load matching documents from Parquet files and insert them into documents table
        inserted_doc_ids = load_and_insert_documents(pg_conn, doc_ids)
        
        # 7. Clear old chunks
        clear_chunks_table(pg_conn)
        
        # 8. Drop indexes for speed
        drop_indexes(pg_conn)
        
        # 9. Upload chunks in parallel
        upload_chunks(
            sqlite_conn, 
            args.host, 
            args.port, 
            args.user, 
            args.password, 
            args.dbname, 
            inserted_doc_ids, 
            total_chunks, 
            args.threads
        )
        
        # 10. Recreate indexes
        recreate_indexes(pg_conn)
        
        # 11. Verify upload
        verify_upload(pg_conn)
        
    except Exception as e:
        print(f"\n[-] Fatal error during database push: {e}")
        import traceback
        traceback.print_exc()
        if pg_conn:
            pg_conn.rollback()
    finally:
        sqlite_conn.close()
        if pg_conn:
            pg_conn.close()
        print("[+] Connections closed.")

if __name__ == "__main__":
    main()
