import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import psycopg2
from pyvi import ViTokenizer
from config import Config

def load_sample_document():
    """Load a sample legal document (e.g., SME Support Law 04/2017/QH14) from Supabase."""
    conn = psycopg2.connect(Config.DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, document_number, title, content 
        FROM documents 
        WHERE document_number = '04/2017/QH14' 
        LIMIT 1;
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "document_number": row[1],
            "title": row[2],
            "content": row[3]
        }
    else:
        conn = psycopg2.connect(Config.DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT id, document_number, title, content FROM documents LIMIT 1;")
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return {
                "id": row[0],
                "document_number": row[1],
                "title": row[2],
                "content": row[3]
            }
    return None

def structure_aware_chunking(doc_title, doc_number, content, max_chars=800, overlap=100):
    """
    Split the legal document by 'Điều' (Article). 
    If an article is longer than max_chars, split it into smaller paragraph chunks.
    Prepend parent metadata to each chunk to keep context alive for vector matching.
    """
    chunks = []
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    
    pattern = r'\n(?=Điều \d+[\.:\s])'
    articles = re.split(pattern, content)
    
    for art_idx, art in enumerate(articles):
        art = art.strip()
        if not art:
            continue
            
        art_header_match = re.match(r'^(Điều \d+[\.:\s]*[^\n]*)', art)
        art_header = art_header_match.group(1) if art_header_match else f"Mục {art_idx}"
        
        if len(art) <= max_chars:
            context_prefix = f"Văn bản: {doc_title} ({doc_number}) | {art_header} | Nội dung: "
            full_text = context_prefix + art
            segmented_text = ViTokenizer.tokenize(full_text)
            
            chunks.append({
                "article": art_header,
                "text": full_text,
                "segmented": segmented_text,
                "length": len(art)
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
                        context_prefix = f"Văn bản: {doc_title} ({doc_number}) | {art_header} (Phần {sub_idx}) | Nội dung: "
                        full_text = context_prefix + current_sub_chunk
                        segmented_text = ViTokenizer.tokenize(full_text)
                        
                        chunks.append({
                            "article": f"{art_header} (Phần {sub_idx})",
                            "text": full_text,
                            "segmented": segmented_text,
                            "length": len(current_sub_chunk)
                        })
                        sub_idx += 1
                    current_sub_chunk = para
                else:
                    if current_sub_chunk:
                        current_sub_chunk += "\n" + para
                    else:
                        current_sub_chunk = para
            
            if current_sub_chunk:
                context_prefix = f"Văn bản: {doc_title} ({doc_number}) | {art_header} (Phần {sub_idx}) | Nội dung: "
                full_text = context_prefix + current_sub_chunk
                segmented_text = ViTokenizer.tokenize(full_text)
                
                chunks.append({
                    "article": f"{art_header} (Phần {sub_idx})",
                    "text": full_text,
                    "segmented": segmented_text,
                    "length": len(current_sub_chunk)
                })
                
    return chunks

def main():
    doc = load_sample_document()
    if not doc:
        print("[-] No document found to chunk.")
        return
        
    chunks = structure_aware_chunking(
        doc_title=doc['title'],
        doc_number=doc['document_number'],
        content=doc['content'],
        max_chars=800
    )
    
    output = []
    output.append(f"Document ID: {doc['id']}")
    output.append(f"Title: {doc['title']}")
    output.append(f"Document Number: {doc['document_number']}")
    output.append(f"Raw Content Length: {len(doc['content'])} characters")
    output.append(f"Generated {len(chunks)} final chunks.")
    output.append("\n" + "="*80)
    output.append("SAMPLE CHUNKS PRODUCED BY HYBRID STRATEGY:")
    output.append("="*80)
    
    for idx, chunk in enumerate(chunks[:5]):
        output.append(f"\n--- Chunk {idx+1} | {chunk['article']} | Raw Length: {chunk['length']} chars ---")
        output.append("\n[Raw Text with Context Prefix]:")
        output.append(chunk['text'])
        output.append("\n[Word-Segmented Text for Embedding Model]:")
        output.append(chunk['segmented'])
        output.append("-" * 50)
        
    with open("chunker_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        
    print("Saved output to chunker_output.txt")

if __name__ == "__main__":
    main()
