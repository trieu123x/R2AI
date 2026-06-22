import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import sqlite3
import pandas as pd
import glob
import os
from pyvi import ViTokenizer

def init_local_db():
    """Initialize a local SQLite database to store chunks for offline RAG."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "database", "local_chunks.db")
    print(f"[+] Initializing local SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id INTEGER,
            chunk_index INTEGER,
            article_hint TEXT,
            article_number INTEGER,
            content TEXT,
            segmented_content TEXT,
            embedding BLOB
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_article ON document_chunks(document_id, article_number);")
    conn.commit()
    return conn

def load_documents_local():
    """Load and merge documents from local Parquet files using the active filters."""
    print("[+] Loading metadata from local parquet...")
    df_meta = pd.read_parquet(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\metadata\data-00000-of-00001.parquet")

    target_sectors = ["Doanh nghiệp", "Lao động - Tiền lương", "Thuế - Phí - Lệ Phí", "Bảo hiểm", "Quyền dân sự"]
    keywords = [
        "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "luật doanh nghiệp",
        "thuế thu nhập doanh nghiệp", "thuế giá trị gia tăng", "bảo hiểm xã hội",
        "hợp đồng lao động", "người lao động", "hộ kinh doanh",
        "đăng ký doanh nghiệp", "giải thể doanh nghiệp", "phá sản"
    ]
    target_doc_nums = [
        "59/2020/QH14", "04/2017/QH14", "80/2021/NĐ-CP", "47/2021/NĐ-CP",
        "13/2008/QH12", "14/2008/QH12", "04/2007/QH12", "38/2019/QH14",
        "45/2019/QH14", "58/2014/QH13", "25/2008/QH12", "145/2020/NĐ-CP",
        "91/2015/QH13", "36/2005/QH11"
    ]

    cond_sector = df_meta['legal_sectors'].fillna('').apply(lambda s: any(sec in s for sec in target_sectors))
    cond_keyword = df_meta['title'].fillna('').apply(lambda t: any(kw in t.lower() for kw in keywords))
    cond_doc_num = df_meta['document_number'].isin(target_doc_nums)

    base_filtered = df_meta[cond_sector | cond_keyword | cond_doc_num]

    is_central = ~base_filtered['issuing_authority'].fillna('').str.contains(
        'Tỉnh|Thành phố|Thành Phố|UBND|HĐND|Huyện|Quận|Thị xã|Thị Xã|Cục|Sở|Chi cục|Ủy ban nhân dân|Hội đồng nhân dân', 
        case=False, 
        na=False
    )
    central_filtered = base_filtered[is_central]

    normative_types = [
        "Luật", "Bộ luật", "Nghị định", "Thông tư", "Thông tư liên tịch", 
        "Nghị quyết", "Pháp lệnh", "Hiến pháp", "Văn bản hợp nhất"
    ]
    cond_normative = central_filtered['legal_type'].isin(normative_types)
    cond_target = central_filtered['document_number'].isin(target_doc_nums)

    final_filtered = central_filtered[cond_normative | cond_target]
    filtered_ids = set(final_filtered['id'].tolist())

    content_files = sorted(glob.glob(r"c:\Users\admin\Downloads\R2AI\vietnamese-legal-documents\content\*.parquet"))
    content_dfs = []
    print("[+] Loading local content parquets...")
    for file_path in content_files:
        df_chunk = pd.read_parquet(file_path)
        matched_chunk = df_chunk[df_chunk['id'].isin(filtered_ids)]
        content_dfs.append(matched_chunk)

    df_all_content = pd.concat(content_dfs, ignore_index=True)
    df_merged = pd.merge(final_filtered, df_all_content, on="id", how="inner")
    print(f"[+] Successfully loaded {len(df_merged)} filtered documents.")
    return df_merged

def chunk_document(doc_id, doc_num, title, content, max_chars=800, overlap=100):
    """Chunk a single document using the Hybrid Chunker strategy."""
    chunks = []
    if not content:
        return chunks
        
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    
    # Split by "Điều X."
    pattern = r'\n(?=Điều \d+[\.:\s])'
    articles = re.split(pattern, content)
    
    for art_idx, art in enumerate(articles):
        art = art.strip()
        if not art:
            continue
            
        art_header_match = re.match(r'^(Điều \d+[\.:\s]*[^\n]*)', art)
        art_header = art_header_match.group(1) if art_header_match else f"Mục {art_idx}"
        
        # Extract article number
        art_num_match = re.match(r'Điều\s+(\d+)', art_header)
        article_number = int(art_num_match.group(1)) if art_num_match else None
        
        if len(art) <= max_chars:
            context_prefix = f"Văn bản: {title} ({doc_num}) | {art_header} | Nội dung: "
            full_text = context_prefix + art
            
            chunks.append({
                "document_id": doc_id,
                "chunk_index": len(chunks),
                "article_hint": art_header,
                "article_number": article_number,
                "content": full_text
            })
        else:
            paragraphs = art.split("\n")
            current_sub_chunk = ""
            sub_idx = 1
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_sub_chunk) + len(para) > max_chars:
                    if current_sub_chunk:
                        hint = f"{art_header} (Phần {sub_idx})"
                        context_prefix = f"Văn bản: {title} ({doc_num}) | {hint} | Nội dung: "
                        full_text = context_prefix + current_sub_chunk
                        
                        chunks.append({
                            "document_id": doc_id,
                            "chunk_index": len(chunks),
                            "article_hint": hint,
                            "article_number": article_number,
                            "content": full_text
                        })
                        sub_idx += 1
                    
                    # Apply overlap if specified
                    overlap_text = current_sub_chunk[-overlap:] if overlap > 0 and len(current_sub_chunk) > overlap else current_sub_chunk
                    current_sub_chunk = overlap_text + "\n" + para if overlap_text else para
                else:
                    current_sub_chunk = (current_sub_chunk + "\n" + para) if current_sub_chunk else para
            
            if current_sub_chunk:
                hint = f"{art_header} (Phần {sub_idx})"
                context_prefix = f"Văn bản: {title} ({doc_num}) | {hint} | Nội dung: "
                full_text = context_prefix + current_sub_chunk
                
                chunks.append({
                    "document_id": doc_id,
                    "chunk_index": len(chunks),
                    "article_hint": hint,
                    "article_number": article_number,
                    "content": full_text
                })
                
    return chunks

def main():
    start_time = time.time()
    
    # 1. Fetch documents locally
    df_merged = load_documents_local()
    
    # 2. Init local SQLite
    local_conn = init_local_db()
    local_cursor = local_conn.cursor()
    
    # Clear existing chunks if any
    local_cursor.execute("DELETE FROM document_chunks;")
    local_conn.commit()
    
    print("[+] Starting chunking and word-segmentation process...")
    
    batch_size = 500
    local_insert_buffer = []
    total_chunks_created = 0
    total_docs = len(df_merged)
    
    for idx, row in df_merged.iterrows():
        doc_id = int(row['id'])
        doc_num = row['document_number'] if pd.notna(row['document_number']) else 'N/A'
        title = row['title']
        content = row['content']
        
        # Chunk document
        doc_chunks = chunk_document(doc_id, doc_num, title, content)
        
        for chunk in doc_chunks:
            chunk_id = f"{chunk['document_id']}_{chunk['chunk_index']}"
            segmented = ViTokenizer.tokenize(chunk['content'])
            
            local_insert_buffer.append((
                chunk_id,
                chunk['document_id'],
                chunk['chunk_index'],
                chunk.get('article_hint', ''),
                chunk.get('article_number'),
                chunk['content'],
                segmented
            ))
            
        if len(local_insert_buffer) >= batch_size:
            local_cursor.executemany("""
                INSERT OR REPLACE INTO document_chunks 
                (id, document_id, chunk_index, article_hint, article_number, content, segmented_content)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, local_insert_buffer)
            local_conn.commit()
            total_chunks_created += len(local_insert_buffer)
            local_insert_buffer = []
            
        if (idx + 1) % 500 == 0 or (idx + 1) == total_docs:
            print(f"    Processed {idx + 1}/{total_docs} documents | Created {total_chunks_created + len(local_insert_buffer)} chunks...")
            
    # Flush remaining buffer
    if local_insert_buffer:
        local_cursor.executemany("""
            INSERT OR REPLACE INTO document_chunks 
            (id, document_id, chunk_index, article_hint, article_number, content, segmented_content)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, local_insert_buffer)
        local_conn.commit()
        total_chunks_created += len(local_insert_buffer)
        
    local_cursor.close()
    local_conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n[+] Chunking and segmenting completed!")
    print(f"[+] Total chunks saved to local SQLite: {total_chunks_created:,}")
    print(f"[+] Total duration: {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
