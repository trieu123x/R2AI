import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import re
from config import Config

def main():
    print("[+] Connecting to Supabase to fetch documents for estimation...")
    conn = psycopg2.connect(Config.DATABASE_URL)
    cursor = conn.cursor()
    
    # We fetch id, document_number, title, and length of content
    cursor.execute("SELECT id, document_number, title, content FROM documents;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"[+] Loaded {len(rows)} documents from Supabase.")
    
    total_raw_chars = 0
    total_chunks = 0
    total_chunk_chars = 0
    
    max_chars = 800
    
    for row in rows:
        doc_id, doc_num, title, content = row
        if not content:
            continue
        
        total_raw_chars += len(content)
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
            
            # Context prefix
            context_prefix = f"Văn bản: {title} ({doc_num}) | {art_header} | Nội dung: "
            prefix_len = len(context_prefix)
            
            if len(art) <= max_chars:
                total_chunks += 1
                total_chunk_chars += prefix_len + len(art)
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
                            total_chunks += 1
                            total_chunk_chars += prefix_len + len(current_sub_chunk) + 15 # extra for "(Phần X)"
                            sub_idx += 1
                        current_sub_chunk = para
                    else:
                        current_sub_chunk = (current_sub_chunk + "\n" + para) if current_sub_chunk else para
                
                if current_sub_chunk:
                    total_chunks += 1
                    total_chunk_chars += prefix_len + len(current_sub_chunk) + 15
                    
    print("\n--- Chunking Estimation ---")
    print(f"Total Raw Characters: {total_raw_chars:,}")
    print(f"Total Chunks: {total_chunks:,}")
    print(f"Total Chunk Characters (with prepended metadata): {total_chunk_chars:,}")
    print(f"Estimated Text size in DB: {total_chunk_chars / (1024 * 1024):.2f} MB")
    
if __name__ == "__main__":
    main()
