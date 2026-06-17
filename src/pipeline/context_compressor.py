import re
from typing import List, Optional

import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import RetrievalResult

class ContextCompressor:
    """
    Step 7: Context Compression
    Trích xuất đúng nội dung Điều luật (nếu có) và rút gọn context để tối ưu prompt cho LLM.
    """
    def __init__(self, max_length_per_doc=1500):
        self.max_length_per_doc = max_length_per_doc

    def extract_evidence(self, content: str, article_hint: Optional[str]) -> str:
        """Trích xuất duy nhất phần Điều luật liên quan từ chunk nội dung để tránh nhiễu."""
        header = ""
        lines = content.split('\n')
        for line in lines[:3]:
            if "Văn bản:" in line:
                header = line.strip() + "\n"
                break

        if not article_hint:
            return content.strip()

        match_num = re.search(r'\d+', article_hint)
        if not match_num:
            return content.strip()

        art_num = match_num.group()
        
        pattern = re.compile(rf'^\s*Điều\s+{art_num}\b', re.MULTILINE | re.IGNORECASE)
        match = pattern.search(content)
        if not match:
            pattern_fallback = re.compile(rf'Điều\s+{art_num}\b', re.IGNORECASE)
            match = pattern_fallback.search(content)

        if not match:
            return content.strip()

        start_idx = match.start()
        
        next_pattern = re.compile(r'^\s*Điều\s+\d+\b', re.MULTILINE | re.IGNORECASE)
        matches_after = list(next_pattern.finditer(content[start_idx + len(match.group()):]))
        
        if matches_after:
            end_idx = start_idx + len(match.group()) + matches_after[0].start()
            extracted_text = content[start_idx:end_idx].strip()
        else:
            extracted_text = content[start_idx:].strip()

        return (header + extracted_text).strip()

    def compress(self, results: List[RetrievalResult]) -> str:
        """Nén danh sách Top 5 thành chuỗi context gọn gàng."""
        context_parts = []
        for idx, r in enumerate(results, start=1):
            extracted_body = self.extract_evidence(r.content, r.article_hint)
            
            # Tách bỏ header line "Văn bản: ..." nếu có
            lines = extracted_body.split('\n')
            body_lines = []
            for line in lines:
                if not line.strip().startswith("Văn bản:"):
                    body_lines.append(line)
            clean_body = "\n".join(body_lines).strip()
            
            # Truncate nếu quá dài
            if len(clean_body) > self.max_length_per_doc:
                clean_body = clean_body[:self.max_length_per_doc] + "...\n[Đã cắt bớt]"
            
            part = (
                f"[Căn cứ {idx}]\n"
                f"Độ liên quan: {r.score:.4f}\n"
                f"Văn bản: {r.legal_type} số {r.doc_number} {r.title}\n"
                f"Điều: {r.article_hint or 'Toàn bộ'}\n"
                f"Nội dung:\n{clean_body}"
            )
            context_parts.append(part)
        
        return "\n\n=========================================\n\n".join(context_parts)
