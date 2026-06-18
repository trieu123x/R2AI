import os
import sys
import sqlite3
import numpy as np
import psycopg2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
PG_CONN_STR = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"

def main():
    print("==================================================")
    print("   Restoring local_chunks.db from PostgreSQL      ")
    print("==================================================")
    
    # 1. Connect to PostgreSQL
    print("[+] Connecting to PostgreSQL...", flush=True)
    try:
        pg_conn = psycopg2.connect(PG_CONN_STR)
    except Exception as e:
        print(f"[-] Failed to connect to PostgreSQL: {e}", flush=True)
        return

    # 2. Check source counts
    pg_cur_count = pg_conn.cursor()
    pg_cur_count.execute("SELECT COUNT(*) FROM document_chunks;")
    total_chunks = pg_cur_count.fetchone()[0]
    pg_cur_count.close()
    print(f"[+] Found {total_chunks:,} chunks in PostgreSQL.", flush=True)

    # 3. Create SQLite DB
    if os.path.exists(DB_PATH):
        print(f"[!] SQLite DB already exists at {DB_PATH}. Deleting to perform clean restore...", flush=True)
        try:
            os.remove(DB_PATH)
        except Exception as e:
            print(f"[-] Failed to delete existing DB: {e}", flush=True)
            pg_conn.close()
            return

    print(f"[+] Creating SQLite DB at {DB_PATH}...", flush=True)
    sqlite_conn = sqlite3.connect(DB_PATH)
    sqlite_cur = sqlite_conn.cursor()

    # 4. Create SQLite Table
    sqlite_cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id INTEGER,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        );
    """)
    sqlite_conn.commit()

    # 5. Fetch and insert data in batches using a named (server-side) cursor
    print(f"[+] Fetching data from PostgreSQL and inserting into SQLite...", flush=True)
    
    # Named cursor for server-side streaming
    pg_cur = pg_conn.cursor(name="streaming_restore_cursor")
    pg_cur.itersize = 10000  # Fetch 10,000 rows at a time
    pg_cur.execute("SELECT id, document_id, chunk_index, content, embedding FROM document_chunks;")
    
    batch_size = 5000
    batch = []
    inserted_count = 0
    
    try:
        while True:
            rows = pg_cur.fetchmany(batch_size)
            if not rows:
                break
                
            sqlite_batch = []
            for row in rows:
                cid, doc_id, chunk_idx, content, emb = row
                
                # Convert embedding array/list to binary float32 blob
                emb_bytes = None
                if emb is not None:
                    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
                    
                sqlite_batch.append((str(cid), doc_id, chunk_idx, content, emb_bytes))
                
            sqlite_cur.executemany("""
                INSERT INTO document_chunks (id, document_id, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?);
            """, sqlite_batch)
            sqlite_conn.commit()
            
            inserted_count += len(sqlite_batch)
            pct = (inserted_count / total_chunks) * 100
            print(f"    Restored {inserted_count:,}/{total_chunks:,} chunks ({pct:.1f}%)", flush=True)
            
    except Exception as e:
        print(f"[-] Error during data transfer: {e}", flush=True)
        sqlite_conn.close()
        pg_conn.close()
        return
    finally:
        pg_cur.close()

    # 6. Build SQLite FTS5 Index
    print("[+] Building SQLite FTS5 Index...", flush=True)
    try:
        sqlite_cur.execute("DROP TABLE IF EXISTS chunks_fts5;")
        sqlite_cur.execute("""
            CREATE VIRTUAL TABLE chunks_fts5
            USING fts5(
                content,
                content='document_chunks',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 1'
            );
        """)
        sqlite_conn.commit()
        
        sqlite_cur.execute("""
            INSERT INTO chunks_fts5(rowid, content)
            SELECT rowid, content
            FROM document_chunks
            WHERE content IS NOT NULL;
        """)
        sqlite_conn.commit()
        
        # Optimize FTS5
        sqlite_cur.execute("INSERT INTO chunks_fts5(chunks_fts5) VALUES('optimize');")
        sqlite_conn.commit()
        print("[+] SQLite FTS5 Index built and optimized successfully.", flush=True)
        
    except Exception as e:
        print(f"[-] Failed to build FTS5 Index: {e}", flush=True)

    # 7. Close connections
    sqlite_conn.close()
    pg_conn.close()
    print("[+] Done! local_chunks.db has been fully restored.", flush=True)

if __name__ == "__main__":
    main()
