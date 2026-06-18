import sys
import io
import psycopg2
import numpy as np

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def main():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="Trieudh.1",
        database="law_vn"
    )
    cursor = conn.cursor()

    # 1. Fetch a target chunk to use as query vector
    cursor.execute("""
        SELECT c.document_id, c.chunk_index, d.title, c.content, c.embedding
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
        LIMIT 1;
    """)
    row = cursor.fetchone()
    if not row:
        print("No chunks found with embeddings.")
        conn.close()
        return

    doc_id, chunk_idx, title, content, target_emb = row
    target_emb = np.array(target_emb, dtype=np.float32)
    
    print("=" * 80)
    print(f"TARGET QUERY CHUNK:")
    print(f"Document ID  : {doc_id}")
    print(f"Title        : {title}")
    print(f"Chunk Index  : {chunk_idx}")
    print(f"Content (cut): {content[:150]}...")
    print(f"Vector Dimensions: {len(target_emb)}")
    print(f"First 5 elements : {target_emb[:5]}")
    print("=" * 80)

    # 2. Fetch a batch of other chunks to compare
    print("\nQuerying top 1000 chunks to compute similarity on-the-fly...")
    cursor.execute("""
        SELECT c.document_id, c.chunk_index, d.title, c.content, c.embedding
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL AND (c.document_id != %s OR c.chunk_index != %s)
        LIMIT 1000;
    """, (doc_id, chunk_idx))
    
    candidates = cursor.fetchall()
    
    # 3. Calculate similarity
    results = []
    for c_doc_id, c_chunk_idx, c_title, c_content, c_emb in candidates:
        c_emb = np.array(c_emb, dtype=np.float32)
        score = cosine_similarity(target_emb, c_emb)
        results.append((c_doc_id, c_chunk_idx, c_title, c_content, score))
        
    # Sort by score descending
    results.sort(key=lambda x: x[4], reverse=True)
    
    print("\nTOP 5 MOST SIMILAR CHUNKS:")
    print("-" * 80)
    for i, (r_doc_id, r_chunk_idx, r_title, r_content, score) in enumerate(results[:5], 1):
        print(f"Top {i} (Similarity Score: {score:.4f})")
        print(f"  Doc ID: {r_doc_id} | Chunk: {r_chunk_idx}")
        print(f"  Title : {r_title}")
        print(f"  Text  : {r_content[:150]}...")
        print("-" * 80)
        
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
