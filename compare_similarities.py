import sqlite3
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval.local_retriever import LocalRetriever

query = "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì sẽ bị xử lý như thế nào và phải khắc phục ra sao?"

retriever = LocalRetriever()
q_emb = retriever.embed_query(query)

conn = sqlite3.connect("database/local_chunks.db")
c = conn.cursor()

# Get target chunk embedding
c.execute("SELECT content, embedding FROM document_chunks WHERE id = '479312_37'")
target_row = c.fetchone()
if target_row:
    target_content = target_row[0]
    target_emb = np.frombuffer(target_row[1], dtype=np.float32)
    
    # Cosine similarity
    dot = np.dot(q_emb, target_emb)
    q_norm = np.linalg.norm(q_emb)
    t_norm = np.linalg.norm(target_emb)
    sim = dot / (q_norm * t_norm)
    print(f"Similarity with 479312_37 (Decree 12/2022/NĐ-CP Điều 9): {sim:.4f}")
    
# Let's do the same for 479312_40 (Buộc trả lại bản chính)
c.execute("SELECT content, embedding FROM document_chunks WHERE id = '479312_40'")
target_row2 = c.fetchone()
if target_row2:
    target_content2 = target_row2[0]
    target_emb2 = np.frombuffer(target_row2[1], dtype=np.float32)
    sim2 = np.dot(q_emb, target_emb2) / (np.linalg.norm(q_emb) * np.linalg.norm(target_emb2))
    print(f"Similarity with 479312_40 (Decree 12/2022/NĐ-CP Điều 9 Biện pháp khắc phục): {sim2:.4f}")

# Let's see the similarity for the retrieved top 1 in vector search (ID: 82/2020/NĐ-CP Điều 34)
c.execute("SELECT id, content, embedding FROM document_chunks WHERE content LIKE '%82/2020/NĐ-CP%' AND content LIKE '%Điều 34%' LIMIT 1")
ret_row = c.fetchone()
if ret_row:
    ret_id = ret_row[0]
    ret_content = ret_row[1]
    ret_emb = np.frombuffer(ret_row[2], dtype=np.float32)
    sim_ret = np.dot(q_emb, ret_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ret_emb))
    print(f"Similarity with retrieved top 1 ({ret_id}): {sim_ret:.4f}")
    print("Top 1 Content snippet:", ret_content[:300])
