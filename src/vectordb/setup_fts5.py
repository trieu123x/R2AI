"""
setup_fts5.py
=============
Tạo SQLite FTS5 virtual table trên local_chunks.db để FTS search cực nhanh.
Chỉ cần chạy một lần. FTS5 dùng BM25 tích hợp sẵn trong SQLite.

Cách dùng:
  python retrieval/setup_fts5.py
  python retrieval/setup_fts5.py --force   # Force rebuild
"""

import os, sys, io, sqlite3, time, argparse

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"[-] DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Kiểm tra FTS5 availability
    try:
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(x);")
        cur.execute("DROP TABLE IF EXISTS _fts5_test;")
        print("[+] SQLite FTS5 available.")
    except Exception as e:
        print(f"[-] FTS5 not available: {e}")
        conn.close()
        return

    # Kiểm tra bảng FTS đã tồn tại và đúng schema chưa
    cur.execute("SELECT name FROM sqlite_master WHERE name='chunks_fts5';")
    fts_exists = cur.fetchone() is not None

    if fts_exists:
        # Kiểm tra schema: phải KHÔNG có content_rowid (dạng rowid join)
        cur.execute("SELECT sql FROM sqlite_master WHERE name='chunks_fts5';")
        fts_sql = cur.fetchone()[0] or ""
        is_correct_schema = "content_rowid" not in fts_sql

        cur.execute("SELECT COUNT(*) FROM chunks_fts5;")
        existing = cur.fetchone()[0]

        if is_correct_schema and existing > 0 and not args.force:
            print(f"[+] FTS5 index ready: {existing:,} rows (correct schema).")
            print(f"    Use --force to rebuild.")
            conn.close()
            return
        else:
            reason = "wrong schema" if not is_correct_schema else ("--force flag" if args.force else "empty index")
            print(f"[!] Rebuilding FTS5 ({reason})...")
            cur.execute("DROP TABLE IF EXISTS chunks_fts5;")
            conn.commit()

    # Đếm chunks
    cur.execute("SELECT COUNT(*) FROM document_chunks WHERE content IS NOT NULL;")
    total = cur.fetchone()[0]
    print(f"[+] Building FTS5 index for {total:,} chunks...")
    print("    Using document_chunks.rowid (SQLite internal integer rowid)...")
    print("    This may take 2-5 minutes...")

    t0 = time.time()

    # Tạo FTS5 với content table trỏ vào document_chunks qua rowid ẩn
    # document_chunks có hidden rowid integer dù id là TEXT
    cur.execute("""
        CREATE VIRTUAL TABLE chunks_fts5
        USING fts5(
            content,
            content='document_chunks',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 1'
        );
    """)
    conn.commit()

    # Populate bằng cách insert rowid và content
    print("[+] Populating FTS5 index...")
    cur.execute("""
        INSERT INTO chunks_fts5(rowid, content)
        SELECT rowid, content
        FROM document_chunks
        WHERE content IS NOT NULL;
    """)
    conn.commit()
    elapsed = time.time() - t0

    # Verify
    cur.execute("SELECT COUNT(*) FROM chunks_fts5;")
    count = cur.fetchone()[0]
    print(f"[+] FTS5 index built: {count:,} rows in {elapsed:.1f}s")

    # Quick test
    print("[+] Testing FTS5 query 'doanh nghiep'...")
    cur.execute("""
        SELECT dc.rowid, dc.id, dc.content
        FROM chunks_fts5 f
        JOIN document_chunks dc ON dc.rowid = f.rowid
        WHERE chunks_fts5 MATCH 'doanh nghiep'
        LIMIT 2;
    """)
    test_rows = cur.fetchall()
    if test_rows:
        print(f"[+] Test OK: {len(test_rows)} results found")
        for r in test_rows:
            print(f"    dc.id={r[1]}, content[:60]={str(r[2])[:60]}")
    else:
        print("[!] Test returned 0 results — check tokenizer compatibility")

    # Optimize
    print("[+] Optimizing FTS5 index...")
    t1 = time.time()
    cur.execute("INSERT INTO chunks_fts5(chunks_fts5) VALUES('optimize');")
    conn.commit()
    print(f"[+] Optimized in {time.time()-t1:.1f}s")

    conn.close()
    print("\n[+] Done! FTS5 index ready.")


if __name__ == "__main__":
    main()
