import os
import sys
import json
import requests
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config

class GPTEnhancer:
    """
    Sử dụng OpenAI GPT để cải thiện chất lượng truy vấn vector thông qua:
    1. Query Expansion (Mở rộng câu hỏi thành từ khóa và thuật ngữ pháp lý liên quan).
    2. HyDE (Hypothetical Document Embeddings - Tạo văn bản giả định trả lời câu hỏi).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        self.is_active = True
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            self.is_active = False

    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Optional[str]:
        if not self.is_active:
            return None
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                return res_json["choices"][0]["message"]["content"].strip()
            else:
                print(f"[GPTEnhancer] API Error {response.status_code}: {response.text}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"[GPTEnhancer] Exception during API call: {e}", file=sys.stderr)
            return None

    def expand_query(self, query: str) -> str:
        if not self.is_active:
            return query
            
        system_prompt = (
            "Bạn là một trợ lý RAG chuyên về luật pháp Việt Nam.\n"
            "Hãy phân tích câu hỏi của người dùng và sinh ra một chuỗi từ khóa tìm kiếm mở rộng.\n"
            "Yêu cầu:\n"
            "1. Bao gồm các danh từ, thuật ngữ pháp luật chính thức đồng nghĩa (ví dụ: 'đấu thầu' -> 'lựa chọn nhà thầu', 'nhà thầu', 'gói thầu').\n"
            "2. Viết câu hỏi dưới dạng ngắn gọn, cô đọng.\n"
            "3. Trả về duy nhất chuỗi văn bản mở rộng chứa từ khóa và cụm từ tìm kiếm, KHÔNG giải thích gì thêm."
        )
        
        expanded = self._call_openai(system_prompt, query, temperature=0.2)
        if expanded:
            print(f"[GPTEnhancer] Expanded Query: \"{expanded}\"")
            return expanded
        return query

    def generate_hyde(self, query: str) -> str:
        if not self.is_active:
            return query
            
        system_prompt = (
            "Bạn là một chuyên gia soạn thảo văn bản luật pháp Việt Nam.\n"
            "Hãy viết một đoạn điều khoản pháp lý giả định (ví dụ: một Điều trong Luật hoặc Nghị định) "
            "để trả lời trực tiếp cho câu hỏi của người dùng.\n"
            "Yêu cầu:\n"
            "1. Sử dụng văn phong pháp lý cực kỳ trang trọng, chính xác.\n"
            "2. Bắt đầu bằng 'Điều ...' và trình bày các khoản 1, 2, 3.\n"
            "3. Trả về duy nhất đoạn văn bản pháp lý giả định đó, KHÔNG chào hỏi hay giải thích gì thêm."
        )
        
        hyde_doc = self._call_openai(system_prompt, query, temperature=0.4)
        if hyde_doc:
            print(f"[GPTEnhancer] Generated HyDE Document (first 100 chars): \"{hyde_doc[:100]}...\"")
            return hyde_doc
        return query
