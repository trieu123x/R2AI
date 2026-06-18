import sys
import io
import os
import sqlite3
import psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = r"c:\Users\admin\Downloads\R2AI"
LOCAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
PG_CONN_STR = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"

def main():
    print("Connecting to SQLite...", flush=True)
    conn_sq = sqlite3.connect(LOCAL_DB_PATH)
    cur_sq = conn_sq.cursor()
    
    print("Connecting to Postgres...", flush=True)
    conn_pg = psycopg2.connect(PG_CONN_STR)
    cur_pg = conn_pg.cursor()
    
    # 1. Check a few document_id and chunk_index matches
    cur_sq.execute("SELECT id, document_id, chunk_index, content FROM document_chunks LIMIT 5;")
    sq_rows = cur_sq.fetchall()
    
    print("SQLite sample chunks:", flush=True)
    for row in sq_rows:
        print(f"  SQLite: id={row[0]}, doc_id={row[1]}, chunk_index={row[2]}", flush=True)
        # Try to find in Postgres
        cur_pg.execute("SELECT id, document_id, chunk_index FROM document_chunks WHERE document_id = %s AND chunk_index = %s;", (row[1], row[2]))
        pg_row = cur_pg.fetchone()
        if pg_row:
            print(f"    FOUND in Postgres: id={pg_row[0]}, doc_id={pg_row[1]}, chunk_index={pg_row[2]}", flush=True)
        else:
            print(f"    NOT FOUND in Postgres: doc_id={row[1]}, chunk_index={row[2]}", flush=True)

    # 2. Test the vector_search query flow manually
    print("\nSimulating vector search SQL flow...", flush=True)
    cur_sq.execute("SELECT rowid, document_id, chunk_index FROM document_chunks WHERE embedding IS NOT NULL LIMIT 5;")
    sq_embs = cur_sq.fetchall()
    rowids = [r[0] for r in sq_embs]
    print(f"SQLite rowids: {rowids}", flush=True)
    
    placeholders_sq = ",".join("?" for _ in rowids)
    cur_sq.execute(f"SELECT rowid, document_id, chunk_index FROM document_chunks WHERE rowid IN ({placeholders_sq});", rowids)
    rowid_to_key = {row[0]: (row[1], row[2]) for row in cur_sq.fetchall()}
    
    keys = list(rowid_to_key.values())
    placeholders_pg = ",".join("(%s, %s)" for _ in keys)
    params = []
    for doc_id, chunk_idx in keys:
        params.extend([doc_id, chunk_idx])
        
    query_pg = f"""
        SELECT c.id, c.document_id, c.chunk_index, c.content,
               d.document_number, d.title, d.legal_type
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE (c.document_id, c.chunk_index) IN ({placeholders_pg});
    """
    print("Running Postgres query:", query_pg, flush=True)
    print("Params:", params, flush=True)
    cur_pg.execute(query_pg, params)
    pg_results = cur_pg.fetchall()
    print(f"Postgres returned {len(pg_results)} rows matching those keys.", flush=True)
    for r in pg_results:
        print(f"  Postgres row: doc_id={r[1]}, chunk_index={r[2]}, title={r[5][:50]}", flush=True)

    conn_sq.close()
    conn_pg.close()

if __name__ == "__main__":
    main()
