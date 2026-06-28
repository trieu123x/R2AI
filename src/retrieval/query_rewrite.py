import os
import re
import json
import sys
import io
from pyvi import ViTokenizer

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError in print statements
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Bảo đảm project root nằm trong Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class QueryRewriter:
    """
    Step 2: Query Rewrite.
    Hỗ trợ cả Rule-based synonym expansion thông minh (sử dụng pyvi word segmentation) 
    và LLM-based rewrite chống mất thực thể quan trọng (Preservation Prompting).
    Kiểu trả về dạng Dictionary chứa câu gốc và câu đã làm sạch.
    """
    def __init__(self, use_llm=False, llm_generator=None):
        self.use_llm = use_llm
        self.llm_generator = llm_generator
        
        # Đường dẫn tới file synonyms.json
        self.synonyms_path = os.path.join(PROJECT_ROOT, "src", "retrieval", "synonyms.json")
        self.synonyms = {}
        self._load_synonyms()

    def _load_synonyms(self):
        """Đọc từ điển từ file JSON, nếu không tồn tại sẽ tự động sinh bằng AI."""
        if not os.path.exists(self.synonyms_path):
            print("[query_rewrite] Không thấy synonyms.json. Tự động sinh từ điển bằng AI...", flush=True)
            try:
                from src.retrieval.generate_synonyms import main as run_generate_synonyms
                run_generate_synonyms()
            except Exception as e:
                print(f"[query_rewrite] Cảnh báo lỗi khi sinh từ điển: {e}", flush=True)
                
        if os.path.exists(self.synonyms_path):
            try:
                with open(self.synonyms_path, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
                print(f"[query_rewrite] Đã nạp thành công {len(self.synonyms)} nhóm từ đồng nghĩa.", flush=True)
            except Exception as e:
                print(f"[query_rewrite] Lỗi đọc synonyms.json: {e}", flush=True)
                
        # Fallback cứng nếu hoàn toàn lỗi đọc ghi
        if not self.synonyms:
            self.synonyms = {
                "sme": ["doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "doanh nghiệp siêu nhỏ"],
                "sa_thải": ["đuổi việc", "chấm dứt hợp đồng lao động", "kỷ luật sa thải"],
                "thai_sản": ["nghỉ đẻ", "mang thai", "sinh con"],
                "bảo_hiểm_xã_hội": ["bhxh", "bảo hiểm", "đóng bảo hiểm"],
                "giấy_tờ": ["bản chính", "bằng cấp", "chứng chỉ", "văn bằng", "giấy tờ tùy thân"]
            }

    def rewrite(self, query: str) -> dict:
        """
        Thực hiện viết lại query.
        Trả về Dictionary: {original_query, rewritten_query}
        """
        query_clean = query.strip()
        if not query_clean:
            return {"original_query": "", "rewritten_query": ""}

        if self.use_llm and self.llm_generator:
            rewritten_query = self._llm_rewrite(query_clean)
        else:
            rewritten_query = self._rule_based_rewrite(query_clean)
            
        return {
            "original_query": query_clean,
            "rewritten_query": rewritten_query
        }

    def _rule_based_rewrite(self, query: str) -> str:
        """
        Mở rộng từ đồng nghĩa bằng so khớp thực thể tách từ (Word Segmentation).
        Sử dụng pyvi.ViTokenizer để tránh lỗi so khớp sai từ đồng âm/chuỗi con.
        """
        q_lower = query.lower()
        
        # Tách từ câu hỏi người dùng (ví dụ: "đóng bảo hiểm xã hội" -> "đóng bảo_hiểm_xã_hội")
        segmented_query = ViTokenizer.tokenize(q_lower)
        words_in_query = set(segmented_query.split())
        
        expanded_terms = set()
        
        # Khớp chính xác các khóa (Key có dấu gạch dưới như 'bảo_hiểm_xã_hội')
        for key, syns in self.synonyms.items():
            # Nếu khóa sau khi tách từ (ví dụ: sa_thải) trùng với 1 từ trong câu hỏi đã tách từ
            # Hoặc bất kỳ từ đồng nghĩa nào nằm trong câu hỏi gốc
            if key in words_in_query or any(s in q_lower for s in syns):
                expanded_terms.update(syns)
                
        if expanded_terms:
            # Gộp câu hỏi gốc và các từ đồng nghĩa được tìm thấy
            new_terms = [t for t in expanded_terms if t not in q_lower]
            if new_terms:
                return f"{query} " + " ".join(new_terms)
                
        return query

    def _llm_rewrite(self, query: str) -> str:
        """
        Cải tiến Prompt Chống mất ngữ cảnh (Preservation Prompting).
        Ép LLM giữ lại các từ khóa thực thể quan trọng, số hiệu, Điều/Khoản.
        """
        prompt = (
            "Bạn là một chuyên gia luật Việt Nam tối ưu hóa truy vấn tìm kiếm cho hệ thống RAG.\n"
            "Nhiệm vụ: Hãy viết lại câu hỏi sau thành một câu truy vấn tìm kiếm ngắn gọn, chứa các từ khóa pháp lý cốt lõi.\n\n"
            "Các quy tắc bắt buộc:\n"
            "1. GIỮ NGUYÊN các con số, số hiệu văn bản pháp lý (dạng 'XX/YYYY/QHXX' hoặc 'XX/YYYY/NĐ-CP'), Điều khoản cụ thể (ví dụ: 'Điều 3', 'Khoản 1').\n"
            "2. GIỮ LẠI các chủ thể, thực thể pháp lý quan trọng (ví dụ: tên doanh nghiệp nhỏ và vừa, người lao động, bảo hiểm xã hội, thai sản).\n"
            "3. LOẠI BỎ hoàn toàn các từ thừa, văn phong nói và hỏi han (ví dụ: 'cho mình hỏi', 'kiện có được không', 'giúp mình với', 'xin cảm ơn').\n"
            "4. Tuyệt đối KHÔNG tự ý suy diễn hoặc trả lời câu hỏi. Kết quả trả về duy nhất là câu truy vấn tìm kiếm tối ưu (dưới 20 từ).\n\n"
            f"Câu hỏi gốc: {query}\n\n"
            "Truy vấn tìm kiếm tối ưu:"
        )
        if hasattr(self.llm_generator, 'generate_direct'):
            try:
                rewritten = self.llm_generator.generate_direct(prompt, max_new_tokens=50)
                if rewritten and len(rewritten.strip()) > 3:
                    return rewritten.strip()
            except Exception as e:
                print(f"[query_rewrite] Lỗi LLM rewrite, fallback sang rule-based: {e}", flush=True)
                
        return self._rule_based_rewrite(query)
