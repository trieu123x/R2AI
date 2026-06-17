import psycopg2
import time

def main():
    print("Connecting to database...")
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="Trieudh.1",
        database="law_vn"
    )
    cursor = conn.cursor()
    
    print("Adding row_id SERIAL column (this may take a few seconds)...")
    t0 = time.time()
    cursor.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS row_id SERIAL;")
    conn.commit()
    print(f"Column added in {time.time() - t0:.2f}s")
    
    print("Creating index on row_id...")
    t0 = time.time()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_row_id ON document_chunks(row_id);")
    conn.commit()
    print(f"Index created in {time.time() - t0:.2f}s")
    
    cursor.close()
    conn.close()
    print("Done successfully!")

if __name__ == "__main__":
    main()
