import os
import sys
# Resolve the project root absolute path (D:\Project\R2AI\R2AI)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import re
import time
import sqlite3
import pandas as pd
import glob
from pyvi import ViTokenizer

def init_local_db():
    """Initialize a local SQLite database to store chunks for offline RAG."""
    db_path = os.path.join(PROJECT_ROOT, "database", "local_chunks.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    print(f"[+] Initializing local SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=OFF;")
        cursor.execute("PRAGMA temp_store=MEMORY;")
        cursor.execute("PRAGMA cache_size=-2000000;") # 2GB cache
    except Exception as e:
        print(f"[-] Failed to apply SQLite PRAGMAs: {e}")
        
    # Drop table first to avoid "no column named parent_id" errors if database already exists with old schema
    cursor.execute("DROP TABLE IF EXISTS document_chunks;")
    cursor.execute("""
        CREATE TABLE document_chunks (
            id TEXT PRIMARY KEY,
            document_id INTEGER,
            chunk_index INTEGER,
            parent_id TEXT,
            content TEXT,
            segmented_content TEXT
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON document_chunks(parent_id);")
    conn.commit()
    return conn

def load_documents_local():
    """Load documents from local Parquet files, filtering ONLY by Central Authority and Normative Types."""
    print("[+] Loading metadata from local parquet...")
    meta_path = os.path.join(PROJECT_ROOT, "vietnamese-legal-documents", "metadata", "data-00000-of-00001.parquet")
    df_meta = pd.read_parquet(meta_path)

    # TUYẾN PHÒNG THỦ 1: Chỉ lấy văn bản cấp Trung ương (Theo Slide 12)
    is_central = ~df_meta['issuing_authority'].fillna('').str.contains(
        'Tỉnh|Thành phố|Thành Phố|UBND|HĐND|Huyện|Quận|Thị xã|Thị Xã|Cục|Sở|Chi cục|Ủy ban nhân dân|Hội đồng nhân dân', 
        case=False, 
        na=False
    )
    central_filtered = df_meta[is_central]

    # TUYẾN PHÒNG THỦ 2: Chỉ lấy các loại văn bản quy phạm pháp luật thành văn chuẩn (Theo Slide 11, 12)
    normative_types = [
        "Luật", "Bộ luật", "Nghị định", "Thông tư", "Thông tư liên tịch", 
        "Nghị quyết", "Pháp lệnh", "Hiến pháp", "Văn bản hợp nhất"
    ]
    final_filtered = central_filtered[central_filtered['legal_type'].isin(normative_types)]
    filtered_ids = set(final_filtered['id'].tolist())

    content_glob = os.path.join(PROJECT_ROOT, "vietnamese-legal-documents", "content", "*.parquet")
    content_files = sorted(glob.glob(content_glob))
    content_dfs = []
    print(f"[+] Loading local content parquets for {len(filtered_ids)} target central documents...")
    
    for file_path in content_files:
        df_chunk = pd.read_parquet(file_path)
        matched_chunk = df_chunk[df_chunk['id'].isin(filtered_ids)]
        content_dfs.append(matched_chunk)

    df_all_content = pd.concat(content_dfs, ignore_index=True)
    df_merged = pd.merge(final_filtered, df_all_content, on="id", how="inner")
    
    print(f"[+] Successfully loaded {len(df_merged)} clean central normative documents for full-scale RAG.")
    return df_merged

class LegalDocumentParser:
    def __init__(self, doc_title="", doc_number=""):
        self.doc_title = doc_title
        self.doc_number = doc_number
        
        self.re_phan = re.compile(r'^(PHẦN|Phần)\s+([IVXLCDM\d]+|thứ\s+\w+)([\.:\-\s]+.*)?$', re.IGNORECASE)
        self.re_chuong = re.compile(r'^(CHƯƠNG|Chương)\s+([IVXLCDM\d]+)([\.:\-\s]+.*)?$', re.IGNORECASE)
        self.re_muc = re.compile(r'^(MỤC|Mục)\s+([IVXLCDM\d]+)([\.:\-\s]+.*)?$', re.IGNORECASE)
        self.re_tieu_muc = re.compile(r'^(Tiểu mục|TIỂU MỤC)\s+(\d+|[IVXLCDM]+)([\.:\-\s]+.*)?$', re.IGNORECASE)
        self.re_dieu = re.compile(r'^Điều\s+(\d+[a-zA-Z]*)([\.:\-\s]+.*)?$', re.IGNORECASE)
        # Match Clause "1.", "2.", "10." or "1)" at start of line (strictly requiring . or ) to avoid false positives like "15 ngày...")
        self.re_khoan = re.compile(r'^(\d{1,2})([\.\)])\s*(.*)$')
        self.re_diem = re.compile(r'^([a-zđăâêôơư]{1,2})\)\s*(.*)$', re.IGNORECASE)

    def parse_to_hierarchy(self, content: str) -> list:
        lines = content.split('\n')
        
        current_phan = ""
        current_chuong = ""
        current_muc = ""
        current_tieu_muc = ""
        
        articles = []
        active_article = None
        active_khoan = None
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            if self.re_phan.match(line_str):
                current_phan = line_str
                current_chuong = ""; current_muc = ""; current_tieu_muc = ""
                continue
                
            if self.re_chuong.match(line_str):
                current_chuong = line_str
                current_muc = ""; current_tieu_muc = ""
                continue
                
            if self.re_muc.match(line_str):
                current_muc = line_str
                current_tieu_muc = ""
                continue
                
            if self.re_tieu_muc.match(line_str):
                current_tieu_muc = line_str
                continue
                
            m_dieu = self.re_dieu.match(line_str)
            if m_dieu:
                if active_article:
                    if active_khoan:
                        active_article['elements'].append(active_khoan)
                        active_khoan = None
                    articles.append(active_article)
                
                dieu_num = m_dieu.group(1)
                active_article = {
                    'num': dieu_num,
                    'header': line_str,
                    'phan': current_phan,
                    'chuong': current_chuong,
                    'muc': current_muc,
                    'tieu_muc': current_tieu_muc,
                    'text': '',
                    'elements': []
                }
                continue
                
            if active_article:
                m_khoan = self.re_khoan.match(line_str)
                m_diem = self.re_diem.match(line_str)
                
                if m_khoan:
                    if active_khoan:
                        active_article['elements'].append(active_khoan)
                    
                    khoan_num = m_khoan.group(1)
                    active_khoan = {
                        'type': 'khoan',
                        'num': khoan_num,
                        'text': line_str,
                        'points': []
                    }
                elif m_diem:
                    diem_letter = m_diem.group(1)
                    diem_item = {
                        'type': 'diem',
                        'letter': diem_letter,
                        'text': line_str
                    }
                    if active_khoan:
                        active_khoan['points'].append(diem_item)
                    else:
                        active_article['elements'].append(diem_item)
                else:
                    if active_khoan:
                        if active_khoan['points']:
                            active_khoan['points'][-1]['text'] += "\n" + line_str
                        else:
                            active_khoan['text'] += "\n" + line_str
                    else:
                        if active_article['elements']:
                            last_elem = active_article['elements'][-1]
                            if last_elem['type'] == 'diem':
                                last_elem['text'] += "\n" + line_str
                            else:
                                if last_elem['points']:
                                    last_elem['points'][-1]['text'] += "\n" + line_str
                                else:
                                    last_elem['text'] += "\n" + line_str
                        else:
                            if active_article['text']:
                                active_article['text'] += "\n" + line_str
                            else:
                                active_article['text'] = line_str

        if active_article:
            if active_khoan:
                active_article['elements'].append(active_khoan)
            articles.append(active_article)
            
        return articles

def chunk_document(doc_id, doc_num, title, content, max_chars=800):
    """Chunk a single document down to Khoản/Điểm with strictly monotonic chunk indexing."""
    chunks = []
    if not content:
        return chunks
        
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    parser = LegalDocumentParser(doc_title=title, doc_number=doc_num)
    parsed_articles = parser.parse_to_hierarchy(content)
    doc_info = f"Văn bản: {title} ({doc_num})"
    
    # Dùng biến đếm độc lập để quản lý thứ tự chunk tăng dần ổn định
    chunk_counter = 0

    if not parsed_articles:
        paragraphs = content.split('\n')
        current_chunk = ""
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current_chunk) + len(p) > max_chars:
                if current_chunk:
                    chunks.append({
                        "document_id": doc_id,
                        "chunk_index": chunk_counter,
                        "parent_id": f"{doc_id}_global",
                        "content": f"{doc_info} | Nội dung: {current_chunk}"
                    })
                    chunk_counter += 1
                current_chunk = p
            else:
                current_chunk = (current_chunk + "\n" + p) if current_chunk else p
        if current_chunk:
            chunks.append({
                "document_id": doc_id,
                "chunk_index": chunk_counter,
                "parent_id": f"{doc_id}_global",
                "content": f"{doc_info} | Nội dung: {current_chunk}"
            })
        return chunks

    for art in parsed_articles:
        hierarchy_parts = []
        if art['phan']: hierarchy_parts.append(art['phan'])
        if art['chuong']: hierarchy_parts.append(art['chuong'])
        if art['muc']: hierarchy_parts.append(art['muc'])
        if art['tieu_muc']: hierarchy_parts.append(art['tieu_muc'])
        hierarchy_parts.append(art['header'])
        
        prefix = " | ".join(hierarchy_parts)
        full_prefix = f"{doc_info} | {prefix}"
        
        # Thiết lập ID Mảnh Cha đại diện cho toàn bộ Điều luật lớn
        parent_id = f"{doc_id}_Dieu_{art['num']}"
        
        # 1. Đoạn text mở đầu của Điều luật (nếu có)
        if art['text'] and art['text'].strip():
            chunks.append({
                "document_id": doc_id,
                "chunk_index": chunk_counter,
                "parent_id": parent_id,
                "content": f"{full_prefix} | Nội dung: {art['text'].strip()}"
            })
            chunk_counter += 1
            
        # 2. Xử lý bóc tách sinh các mảnh Khoản và Điểm con
        for elem in art['elements']:
            if elem['type'] == 'khoan':
                khoan_header = f"Khoản {elem['num']}"
                khoan_text = elem['text'].strip()
                
                # Nạp Mảnh con Khoản
                chunks.append({
                    "document_id": doc_id,
                    "chunk_index": chunk_counter,
                    "parent_id": parent_id,
                    "content": f"{full_prefix} | {khoan_header} | Nội dung: {khoan_text}"
                })
                chunk_counter += 1
                
                # Nạp Mảnh con Điểm kẹp bối cảnh của Khoản cha
                for pt in elem['points']:
                    diem_header = f"Điểm {pt['letter']}"
                    diem_text = pt['text'].strip()
                    chunks.append({
                        "document_id": doc_id,
                        "chunk_index": chunk_counter,
                        "parent_id": parent_id,
                        "content": f"{full_prefix} | {khoan_header} | {diem_header} | Bối cảnh {khoan_header}: {khoan_text} | Nội dung {diem_header}: {diem_text}"
                    })
                    chunk_counter += 1
                    
            elif elem['type'] == 'diem':
                diem_header = f"Điểm {elem['letter']}"
                diem_text = elem['text'].strip()
                chunks.append({
                    "document_id": doc_id,
                    "chunk_index": chunk_counter,
                    "parent_id": parent_id,
                    "content": f"{full_prefix} | {diem_header} | Nội dung: {diem_text}"
                })
                chunk_counter += 1
                
    return chunks

def main():
    start_time = time.time()
    
    # 1. Nạp và lọc dữ liệu chuẩn Trung ương
    df_merged = load_documents_local()
    
    # 2. Khởi tạo SQLite
    local_conn = init_local_db()
    local_cursor = local_conn.cursor()
    
    local_cursor.execute("DELETE FROM document_chunks;")
    local_conn.commit()
    
    print("[+] Starting chunking and word-segmentation process...")
    
    batch_size = 5000 # Rút nhỏ batch xuống 5000 để tối ưu hóa bộ nhớ RAM máy local
    local_insert_buffer = []
    total_chunks_created = 0
    total_docs = len(df_merged)
    
    for idx, row in df_merged.iterrows():
        doc_id = int(row['id'])
        doc_num = row['document_number'] if pd.notna(row['document_number']) else 'N/A'
        title = row['title']
        content = row['content']
        
        doc_chunks = chunk_document(doc_id, doc_num, title, content)
        
        for chunk in doc_chunks:
            chunk_id = f"{chunk['document_id']}_{chunk['chunk_index']}"
            segmented = ViTokenizer.tokenize(chunk['content'])
            
            local_insert_buffer.append((
                chunk_id,
                chunk['document_id'],
                chunk['chunk_index'],
                chunk['parent_id'], # Đẩy trường parent_id vào cơ sở dữ liệu
                chunk['content'],
                segmented
            ))
            
        if len(local_insert_buffer) >= batch_size:
            local_cursor.executemany("""
                INSERT OR REPLACE INTO document_chunks (id, document_id, chunk_index, parent_id, content, segmented_content)
                VALUES (?, ?, ?, ?, ?, ?);
            """, local_insert_buffer)
            local_conn.commit()
            total_chunks_created += len(local_insert_buffer)
            local_insert_buffer = []
            
        if (idx + 1) % 100 == 0 or (idx + 1) == total_docs:
            print(f"    Processed {idx + 1}/{total_docs} documents | Created {total_chunks_created + len(local_insert_buffer)} chunks...")
            
    if local_insert_buffer:
        local_cursor.executemany("""
            INSERT OR REPLACE INTO document_chunks (id, document_id, chunk_index, parent_id, content, segmented_content)
            VALUES (?, ?, ?, ?, ?, ?);
        """, local_insert_buffer)
        local_conn.commit()
        total_chunks_created += len(local_insert_buffer)
        
    local_cursor.close()
    local_conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n[✓] Chunking and segmenting completed successfully!")
    print(f"[+] Total chunks saved to local SQLite: {total_chunks_created:,}")
    print(f"[+] Total duration: {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()