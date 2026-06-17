import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

# We can query all chunks and parse their document number and title
c.execute("SELECT DISTINCT content LIKE '%Văn bản:%' FROM document_chunks")
print("Has header:", c.fetchall())

# Let's count how many chunks we have
c.execute("SELECT COUNT(*) FROM document_chunks")
print("Total chunks:", c.fetchone()[0])

# Let's inspect some of the headers by retrieving first 500 characters of some chunks
c.execute("SELECT content FROM document_chunks LIMIT 10")
for row in c.fetchall():
    print(repr(row[0][:150]))
