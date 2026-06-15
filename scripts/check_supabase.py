import sys, os
sys.path.append(r'c:\Users\admin\Downloads\R2AI')
from config import Config
import psycopg2
conn = psycopg2.connect(Config.DATABASE_URL)
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM documents;')
docs = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM document_chunks;')
chunks = cur.fetchone()[0]
print(f'documents      : {docs:,}')
print(f'document_chunks: {chunks:,}')
conn.close()
