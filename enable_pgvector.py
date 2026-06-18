import psycopg2

conn = psycopg2.connect("postgresql://postgres:Trieudh.1@localhost:5432/law_vn")
cur = conn.cursor()

try:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    print("[OK] pgvector extension enabled!")
except Exception as e:
    print(f"[ERR] Could not enable extension: {e}")
    conn.rollback()

cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname='vector';")
row = cur.fetchone()
if row:
    print(f"[OK] Verified: pgvector {row[1]} is active in database.")
else:
    print("[WARN] pgvector extension NOT found. You may need to install pgvector server-side first.")
    print("       Run: pip install pgvector  (cho Python adapter)")
    print("       Hoặc cài pgvector cho PostgreSQL server theo: https://github.com/pgvector/pgvector")

conn.close()
