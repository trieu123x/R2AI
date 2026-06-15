"""
push_chunks_to_supabase.py
Doc chunks tu local SQLite (database/local_chunks.db) va upload len Supabase PostgreSQL.
Uu tien upload documents truoc, sau do upload document_chunks.
"""
import os
import sys
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import queue
import threading
import sqlite3
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from config import Config

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH  = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
BATCH_SIZE     = 5000  # records per batch insert


def connect_supabase():
    print("[+] Connecting to Supabase PostgreSQL...", flush=True)
    conn = psycopg2.connect(
        Config.DATABASE_URL,
        connect_timeout=30,
        options="-c statement_timeout=0",  # disable statement timeout for bulk insert
    )
    conn.autocommit = False
    print("[+] Connected successfully.", flush=True)
    return conn


def get_local_chunks(sqlite_conn):
    print("[+] Counting rows in local SQLite...", flush=True)
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM document_chunks;")
    total = cursor.fetchone()[0]
    print(f"[+] Local SQLite contains {total:,} chunks.", flush=True)
    return total


def get_existing_doc_ids(pg_conn):
    """Lay danh sach document_id da co trong Supabase de skip neu chua co."""
    print("[+] Fetching existing document IDs from Supabase...", flush=True)
    cursor = pg_conn.cursor()
    cursor.execute("SELECT id FROM documents;")
    ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    print(f"[+] Supabase documents table has {len(ids):,} existing records.", flush=True)
    return ids


def ensure_schema(pg_conn):
    """Chay schema.sql neu chua co table."""
    print("[+] Verifying schema...", flush=True)
    schema_path = os.path.join(PROJECT_ROOT, "database", "schema.sql")
    cursor = pg_conn.cursor()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    try:
        cursor.execute(schema_sql)
        pg_conn.commit()
        print("[+] Schema verified / initialized.", flush=True)
    except Exception as e:
        pg_conn.rollback()
        print(f"[!] Schema warning (may already exist): {e}", flush=True)
    finally:
        cursor.close()


def clear_chunks_table(pg_conn):
    """Xoa toan bo chunks cu truoc khi upload lai."""
    cursor = pg_conn.cursor()
    print("[+] Truncating existing document_chunks table...", flush=True)
    cursor.execute("DELETE FROM document_chunks;")
    pg_conn.commit()
    cursor.close()
    print("[+] document_chunks cleared.", flush=True)


def upload_worker(q, existing_doc_ids, results, thread_id):
    conn = None
    try:
        conn = connect_supabase()
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
                    conn.rollback()
                    # Fallback row-by-row
                    row_sql = """
                        INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """
                    ok = fail = 0
                    for row in filtered_batch:
                        try:
                            cursor.execute(row_sql, row)
                            conn.commit()
                            ok += 1
                        except Exception as re:
                            conn.rollback()
                            fail += 1
                    results.append(ok)
            q.task_done()
    except Exception as e:
        print(f"\n[Thread-{thread_id}] Connection error: {e}", flush=True)
    finally:
        if conn:
            conn.close()


def upload_chunks(sqlite_conn, pg_conn, total_chunks, num_threads=6):
    """
    Doc chunks tu SQLite va upload len Supabase song song bang nhieu threads.
    """
    existing_doc_ids = get_existing_doc_ids(pg_conn)

    print(f"\n[+] Starting PARALLEL upload — threads={num_threads}, batch_size={BATCH_SIZE}, total={total_chunks:,}", flush=True)
    print("-" * 60, flush=True)

    q = queue.Queue(maxsize=num_threads * 2)
    results = []
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=upload_worker, args=(q, existing_doc_ids, results, i))
        t.start()
        threads.append(t)

    sqlite_cursor = sqlite_conn.cursor()
    # Check if embedding column exists in SQLite table
    sqlite_cursor.execute("PRAGMA table_info(document_chunks);")
    columns = [col[1] for col in sqlite_cursor.fetchall()]
    
    if "embedding" in columns:
        print("[+] 'embedding' column found in local database, uploading WITH embeddings.")
        select_sql = "SELECT document_id, chunk_index, content, embedding FROM document_chunks ORDER BY document_id, chunk_index"
    else:
        print("[!] 'embedding' column NOT found in local database, uploading WITHOUT embeddings.")
        select_sql = "SELECT document_id, chunk_index, content, NULL FROM document_chunks ORDER BY document_id, chunk_index"

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
        
        # Log progress based on offset loaded into queue
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

    # Send sentinels to stop threads
    for _ in range(num_threads):
        q.put(None)

    for t in threads:
        t.join()

    sqlite_cursor.close()
    
    uploaded = sum(results)
    skipped = total_chunks - uploaded
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"[+] Upload completed!")
    print(f"    Chunks uploaded : {uploaded:,}")
    print(f"    Chunks skipped  : {skipped:,}  (no matching document or duplicates)")
    print(f"    Duration        : {elapsed:.1f}s  ({uploaded/elapsed:.0f} chunks/s avg)")
    print("="*60)
    return uploaded, skipped


def drop_indexes(pg_conn):
    """Drop indexes to speed up bulk uploads."""
    print("[+] Dropping indexes to accelerate bulk upload...", flush=True)
    cursor = pg_conn.cursor()
    cursor.execute("DROP INDEX IF EXISTS idx_document_chunks_document_id;")
    cursor.execute("DROP INDEX IF EXISTS idx_document_chunks_fts;")
    pg_conn.commit()
    cursor.close()
    print("[+] Indexes dropped.", flush=True)


def recreate_indexes(pg_conn):
    """Recreate indexes after bulk upload."""
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


def verify_upload(pg_conn):
    """Kiem tra so luong chunks da upload."""
    cursor = pg_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM document_chunks;")
    count = cursor.fetchone()[0]
    cursor.close()
    print(f"\n[+] Verification: Supabase document_chunks now has {count:,} rows.")
    return count


def main():
    # Validate config
    try:
        Config.validate()
    except Exception as e:
        print(f"[-] Config error: {e}")
        return

    # Check local DB
    if not os.path.exists(LOCAL_DB_PATH):
        print(f"[-] Local SQLite not found at: {LOCAL_DB_PATH}")
        print("    Please run src/process_chunks.py first.")
        return

    # Open connections
    sqlite_conn = sqlite3.connect(LOCAL_DB_PATH)
    total_chunks = get_local_chunks(sqlite_conn)

    if total_chunks == 0:
        print("[-] No chunks found in local database. Exiting.")
        sqlite_conn.close()
        return

    pg_conn = connect_supabase()

    try:
        ensure_schema(pg_conn)
        clear_chunks_table(pg_conn)
        drop_indexes(pg_conn)
        uploaded, skipped = upload_chunks(sqlite_conn, pg_conn, total_chunks)
        recreate_indexes(pg_conn)
        verify_upload(pg_conn)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n[-] Fatal error: {e}")
        pg_conn.rollback()
    finally:
        sqlite_conn.close()
        pg_conn.close()
        print("[+] Connections closed.")


if __name__ == "__main__":
    main()
