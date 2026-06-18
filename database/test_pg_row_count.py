import psycopg2

PG_CONN_STR = "postgresql://postgres:Trieudh.1@localhost:5432/law_vn"
conn = psycopg2.connect(PG_CONN_STR)
cur = conn.cursor()

fts_expr = "(('người' <-> 'lao' <-> 'động') | ('lao' <-> 'động') | ('nhân' <-> 'viên') | ('hợp' <-> 'đồng' <-> 'lao' <-> 'động')) & (('giữ') | ('thu' <-> 'giữ') | ('tạm' <-> 'giữ') | ('văn' <-> 'bằng') | ('chứng' <-> 'chỉ') | ('giấy' <-> 'tờ' <-> 'tùy' <-> 'thân') | ('bản' <-> 'chính') | ('bằng' <-> 'cấp'))"

cur.execute("""
    SELECT count(*)
    FROM document_chunks c
    WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
""", (fts_expr,))
count = cur.fetchone()[0]
print("Matching row count for structured query:", count)

conn.close()
