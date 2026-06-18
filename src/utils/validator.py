import re
from typing import List, Tuple

import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import RetrievalResult

class PipelineValidator:
    """
    Step 9: Citation Validation
    Kiểm tra xem câu trả lời có lặp từ, hoặc trích dẫn các Điều/Số hiệu văn bản không có trong context hay không.
    """
    def __init__(self):
        pass

    def has_repetitive_loop(self, text: str) -> bool:
        """Kiểm tra xem câu trả lời có bị lặp từ/cụm từ vô hạn (loop) hay không."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for i in range(len(lines) - 2):
            if lines[i] == lines[i+1] == lines[i+2]:
                print(f"[validator] Phát hiện lặp dòng liên tiếp: '{lines[i]}'")
                return True
                
        for line in lines:
            words = line.split()
            if len(words) > 15:
                # Check sliding window of size 2 to 8
                for size in range(2, 9):
                    for start in range(len(words) - 3 * size):
                        w1 = words[start : start + size]
                        w2 = words[start + size : start + 2 * size]
                        w3 = words[start + 2 * size : start + 3 * size]
                        if w1 == w2 == w3:
                            print(f"[validator] Phát hiện lặp cụm từ vô hạn: {' '.join(w1)}")
                            return True
        return False

    def validate(self, answer: str, results: List[RetrievalResult]) -> Tuple[bool, str]:
        """
        Kiểm tra chi tiết trích dẫn của câu trả lời so với context.
        Trả về (is_valid, error_message)
        """
        if self.has_repetitive_loop(answer):
            return False, "câu trả lời bị lặp từ/cụm từ vô hạn (loop)"

        # Check Articles
        answer_articles = set(re.findall(r"Điều\s+(\d+)", answer, re.IGNORECASE))
        if answer_articles:
            context_articles = set()
            for r in results:
                if r.article_hint:
                    matches = re.findall(r"Điều\s+(\d+)", r.article_hint, re.IGNORECASE)
                    context_articles.update(matches)
                matches_content = re.findall(r"Điều\s+(\d+)", r.content, re.IGNORECASE)
                context_articles.update(matches_content)
                
            invalid_articles = answer_articles - context_articles
            if invalid_articles:
                return False, f"dẫn chiếu sai các Điều không có trong context: {invalid_articles}"
                
        # Check Documents
        answer_docs = set(re.findall(r"(\d+/\d+)", answer))
        if answer_docs:
            context_docs = set()
            for r in results:
                if r.doc_number:
                    matches = re.findall(r"(\d+/\d+)", r.doc_number)
                    context_docs.update(matches)
                matches_content = re.findall(r"(\d+/\d+)", r.content)
                context_docs.update(matches_content)
                
            invalid_docs = answer_docs - context_docs
            if invalid_docs:
                return False, f"dẫn chiếu sai các số hiệu Văn bản không có trong context: {invalid_docs}"
                
        return True, ""
