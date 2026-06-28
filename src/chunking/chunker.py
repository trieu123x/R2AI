import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from pyvi import ViTokenizer

def load_sample_document():
    """Load a hardcoded sample legal document for offline testing."""
    return {
        "id": 1,
        "document_number": "04/2017/QH14",
        "title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
        "content": """LUẬT HỖ TRỢ DOANH NGHIỆP NHỎ VÀ VỪA
Điều 1. Phạm vi điều chỉnh
Luật này quy định về nguyên tắc, nội dung, nguồn lực hỗ trợ doanh nghiệp nhỏ và vừa; trách nhiệm của cơ quan, tổ chức và cá nhân có liên quan đến hỗ trợ doanh nghiệp nhỏ và vừa.
Điều 2. Đối tượng áp dụng
1. Doanh nghiệp được thành lập, tổ chức và hoạt động theo quy định của pháp luật về doanh nghiệp, đáp ứng các tiêu chí xác định doanh nghiệp nhỏ và vừa theo quy định của Luật này.
2. Cơ quan, tổ chức và cá nhân có liên quan đến hỗ trợ doanh nghiệp nhỏ và vừa.
Điều 3. Nguyên tắc hỗ trợ doanh nghiệp nhỏ và vừa
1. Việc hỗ trợ doanh nghiệp nhỏ và vừa phải tôn trọng quy luật thị trường, phù hợp với điều ước quốc tế mà nước Cộng hòa xã hội chủ nghĩa Việt Nam là thành viên.
2. Bảo đảm công khai, minh bạch về nội dung, đối tượng, trình tự, thủ tục, nguồn lực, mức hỗ trợ và kết quả hỗ trợ.
3. Trường hợp doanh nghiệp nhỏ và vừa đồng thời đáp ứng điều kiện của các mức hỗ trợ khác nhau trong cùng một nội dung hỗ trợ thì được lựa chọn mức hỗ trợ có lợi nhất."""
    }

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
