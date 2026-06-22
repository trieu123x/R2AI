import os
import sqlite3
import re
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")

def main():
    print("==================================================")
    print("   Migrating local_chunks.db to add new columns   ")
    print("==================================================")
    
    if not os.path.exists(LOCAL_DB_PATH):
        print(f"[-] Local SQLite DB not found at: {LOCAL_DB_PATH}")
        return
        
    print(f"[+] Connecting to SQLite at {LOCAL_DB_PATH}...")
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    
    # 1. Check existing columns
    cursor.execute("PRAGMA table_info(document_chunks);")
    columns = [col[1] for col in cursor.fetchall()]
    
    # 2. Add columns if they do not exist
    if "article_hint" not in columns:
        print("[+] Adding 'article_hint' column...")
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN article_hint TEXT;")
        conn.commit()
        
    if "article_number" not in columns:
        print("[+] Adding 'article_number' column...")
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN article_number INTEGER;")
        conn.commit()
        
    # 3. Migrate data in-place
    print("[+] Loading chunks from SQLite...")
    cursor.execute("SELECT id, content FROM document_chunks WHERE article_hint IS NULL OR article_number IS NULL;")
    rows = cursor.fetchall()
    total_rows = len(rows)
    print(f"[+] Found {total_rows:,} chunks to migrate.")
    
    if total_rows > 0:
        start_time = time.time()
        batch = []
        batch_size = 5000
        processed = 0
        
        # Precompile regex for extraction
        art_num_re = re.compile(r'Điều\s+(\d+)')
        
        for rid, content in rows:
            if not content:
                continue
                
            # Parse article_hint and article_number
            parts = content.split(" | ")
            if len(parts) >= 3:
                art_header = parts[1].strip()
            else:
                art_header = ""
                
            # Extract number
            art_num_match = art_num_re.match(art_header)
            article_number = int(art_num_match.group(1)) if art_num_match else None
            
            batch.append((art_header, article_number, rid))
            
            if len(batch) == batch_size:
                cursor.executemany("""
                    UPDATE document_chunks 
                    SET article_hint = ?, article_number = ? 
                    WHERE id = ?;
                """, batch)
                conn.commit()
                processed += len(batch)
                batch = []
                
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total_rows - processed
                eta = remaining / rate if rate > 0 else 0
                print(f"    Migrated {processed:,}/{total_rows:,} ({processed/total_rows*100:.1f}%) | Speed: {rate:.0f} rows/s | ETA: {eta:.0f}s")
                
        if batch:
            cursor.executemany("""
                UPDATE document_chunks 
                SET article_hint = ?, article_number = ? 
                WHERE id = ?;
            """, batch)
            conn.commit()
            processed += len(batch)
            print(f"    Migrated {processed:,}/{total_rows:,} (100.0%)")
            
        print(f"[+] Migration complete in {time.time() - start_time:.1f}s.")
        
    # 4. Re-create indexes
    print("[+] Re-creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_article ON document_chunks(document_id, article_number);")
    conn.commit()
    print("[+] Indexes created successfully.")
    
    conn.close()
    print("[+] Done!")

if __name__ == "__main__":
    main()
