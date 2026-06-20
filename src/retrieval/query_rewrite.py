import re

class QueryRewriter:
    """
    Step 2: Query Rewrite.
    Mở rộng câu hỏi bằng các từ đồng nghĩa hoặc viết lại câu hỏi rõ ràng hơn.
    """
    def __init__(self, use_llm=False, llm_generator=None):
        self.use_llm = use_llm
        self.llm_generator = llm_generator
        
        # Simple rule-based synonym dictionary
        self.synonyms = {
            "sme": ["doanh nghiệp nhỏ và vừa", "nhỏ và vừa", "doanh nghiệp siêu nhỏ"],
            "sa thải": ["đuổi việc", "chấm dứt hợp đồng lao động", "kỷ luật"],
            "thai sản": ["nghỉ đẻ", "mang thai", "sinh con"],
            "bảo hiểm xã hội": ["bhxh", "bảo hiểm", "đóng bảo hiểm"],
            "giấy tờ": ["bản chính", "bằng cấp", "chứng chỉ", "văn bằng", "giữ giấy tờ"],
            "công đoàn": ["tổ chức công đoàn", "đại diện người lao động"]
        }

    def rewrite(self, query: str) -> str:
        """Thực hiện rewrite query."""
        if self.use_llm and self.llm_generator:
            return self._llm_rewrite(query)
        return self._rule_based_rewrite(query)
        
    def _rule_based_rewrite(self, query: str) -> str:
        q_lower = query.lower()
        expanded_terms = set()
        
        for key, syns in self.synonyms.items():
            if key in q_lower or any(s in q_lower for s in syns):
                expanded_terms.update(syns)
                
        if expanded_terms:
            # Combine original query with expanded terms
            expansion = " ".join(expanded_terms)
            # Avoid too long queries
            return f"{query} {expansion}"
        return query

    def _llm_rewrite(self, query: str) -> str:
        # Giả định dùng LLM để trích xuất keyword
        # Để tiết kiệm thời gian và tài nguyên, ta có thể cài đặt prompt ngắn
        prompt = (
            "Bạn là một chuyên gia luật Việt Nam.\n"
            "Hãy viết lại câu hỏi sau thành một câu truy vấn tìm kiếm ngắn gọn, "
            "chứa các từ khóa pháp lý quan trọng nhất, không dài quá 20 từ.\n\n"
            f"Câu hỏi: {query}\n\n"
            "Truy vấn tìm kiếm:"
        )
        if hasattr(self.llm_generator, 'generate_direct'):
            rewritten = self.llm_generator.generate_direct(prompt, max_new_tokens=50)
            if rewritten and len(rewritten) > 5:
                return rewritten
        return query
