"""
qwen_generator.py
=================
Bộ sinh câu trả lời tự động bằng LLM local (mặc định Qwen/Qwen3-8B-Instruct).
Hỗ trợ:
  - Tự động detect và offload sang GPU bằng device_map="auto"
  - Sử dụng chat template chuẩn của dòng Qwen
  - Tối ưu hóa memory bằng torch.bfloat16 hoặc torch.float16
"""

import os
import sys
import re
import time
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import RetrievalResult

def extract_evidence(content: str, article_hint: Optional[str]) -> str:
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

class QwenGenerator:
    """
    Trình phát sinh câu trả lời pháp lý local sử dụng mô hình Qwen3-8B.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-8B-Instruct"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        """Lazy load tokenizer và model để tối ưu hóa bộ nhớ khi không dùng tới."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"

        print(f"[generator] Đang load model '{self.model_name}' (Offline={is_offline})...", flush=True)
        t0 = time.time()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=is_offline
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            print(f"[generator] Thiết lập mô hình chạy trên dtype={dtype}...", flush=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=is_offline
            )
            print(f"[generator] Model loaded in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[generator] Failed to load model: {e}")
            raise

    def generate_answer(self, query: str, results: List[RetrievalResult], max_new_tokens: int = 1024, warning_msg: Optional[str] = None) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi và các tài liệu luật pháp đã truy xuất.
        """
        if not results:
            return "Không tìm thấy tài liệu luật pháp liên quan để làm căn cứ trả lời."

        self._lazy_load()

        # Xây dựng Structured Context từ các tài liệu đã trích xuất bằng chứng
        context_parts = []
        for idx, r in enumerate(results, start=1):
            extracted_body = extract_evidence(r.content, r.article_hint)
            # Tách bỏ header line "Văn bản: ..." nếu có
            lines = extracted_body.split('\n')
            body_lines = []
            for line in lines:
                if not line.strip().startswith("Văn bản:"):
                    body_lines.append(line)
            clean_body = "\n".join(body_lines).strip()
            
            part = (
                f"[Căn cứ {idx}]\n"
                f"Độ liên quan: {r.score:.4f}\n"
                f"Văn bản: {r.legal_type} số {r.doc_number} {r.title}\n"
                f"Điều: {r.article_hint or 'Toàn bộ'}\n"
                f"Nội dung:\n{clean_body}"
            )
            context_parts.append(part)
        
        context = "\n\n=========================================\n\n".join(context_parts)

        if warning_msg:
            context = warning_msg + "\n\n" + context

        # Hệ thống Prompt hướng dẫn cực kỳ rõ ràng cho Qwen
        system_prompt = (
            "Bạn là một trợ lý pháp lý Việt Nam chuyên nghiệp, chính xác và đáng tin cậy.\n\n"
            "Nhiệm vụ của bạn:\n"
            "- Trả lời câu hỏi dựa trên CÁC CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP. Tuyệt đối không tự suy diễn hoặc giả định các nội dung không có trong tài liệu.\n"
            "- Hãy viết hoàn toàn bằng tiếng Việt chuẩn mực. Tuyệt đối KHÔNG trộn lẫn tiếng Trung (chữ Hán), tiếng Anh hoặc bất kỳ từ ngữ nước ngoài nào khác.\n"
            "- Tuyệt đối KHÔNG được tự bịa ra các số hiệu văn bản, điều luật không có trong các căn cứ pháp lý.\n"
            "- Tuyệt đối KHÔNG sử dụng tên văn bản, điều luật hoặc các từ ngữ trong Ví dụ mẫu (Few-shot) như 'Nghị định X', 'quyền công đoàn', 'thành lập công đoàn' để trả lời cho câu hỏi thực tế.\n"
            "- Cực kỳ cẩn thận với các từ phủ định hoặc hành vi cấm: các cụm từ như 'nghiêm cấm', 'không được', 'xử phạt đối với hành vi' có nghĩa là hành vi đó BỊ CẤM, tuyệt đối không được viết thành 'người sử dụng lao động được phép thực hiện'.\n"
            "- Không chỉ nêu tên văn bản chung chung. Hãy giải thích cụ thể nội dung quyền lợi, nghĩa vụ, ưu đãi hoặc mức phạt được quy định.\n\n"
            "Bạn BẮT BUỘC phải trình bày câu trả lời của mình nghiêm ngặt theo cấu trúc 4 phần sau (sử dụng chính xác các tiêu đề này làm tiêu đề dòng):\n"
            "1. Trả lời trực tiếp: Trả lời trực tiếp vào câu hỏi, nêu rõ kết luận chính hoặc hành vi và hệ quả.\n"
            "2. Phân tích chi tiết: Diễn giải chi tiết nội dung quy định, mức phạt bằng tiền cụ thể, quyền lợi/nghĩa vụ chi tiết được quy định trong các tài liệu tham khảo.\n"
            "3. Căn cứ pháp lý: Liệt kê chi tiết các điều khoản, điều luật cụ thể được sử dụng làm căn cứ từ tài liệu tham khảo (Ví dụ: 'Điều 17 Bộ luật Lao động 2019', 'Điều 15 Nghị định 12/2022/NĐ-CP').\n"
            "4. Hạn chế của dữ liệu (nếu có): Nêu rõ nếu các căn cứ pháp lý được cung cấp thiếu thông tin hoặc không đủ cơ sở để trả lời đầy đủ một khía cạnh nào đó của câu hỏi."
        )

        user_content = (
            f"TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
            f"=========================================\n"
            f"{context}\n"
            f"=========================================\n\n"
            f"CÂU HỎI: {query}\n\n"
            f"Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            # Few-shot example
            {"role": "user", "content": (
                "TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
                "=========================================\n"
                "[Căn cứ 1]\n"
                "Độ liên quan: 0.9876\n"
                "Văn bản: Nghị định X năm 2022 về xử phạt hành chính\n"
                "Điều: Điều 20\n"
                "Nội dung:\n"
                "Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập công đoàn.\n"
                "=========================================\n\n"
                "CÂU HỎI: Hành vi cản trở người lao động thành lập công đoàn bị phạt bao nhiêu tiền?\n\n"
                "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên."
            )},
            {"role": "assistant", "content": (
                "1. Trả lời trực tiếp: Hành vi cản trở người lao động thành lập công đoàn sẽ bị xử phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng.\n\n"
                "2. Phân tích chi tiết:\n"
                "- Về hành vi vi phạm: Người sử dụng lao động có hành vi cản trở người lao động thành lập công đoàn bị nghiêm cấm theo quy định.\n"
                "- Về mức phạt: Hành vi vi phạm này sẽ bị xử phạt hành chính với mức phạt tiền cụ thể từ 10.000.000 đồng đến 20.000.000 đồng.\n\n"
                "3. Căn cứ pháp lý:\n"
                "- Khoản 1 Điều 20 Nghị định X năm 2022.\n\n"
                "4. Hạn chế của dữ liệu (nếu có):\n"
                "- Tài liệu được cung cấp không đề cập đến các hình thức xử phạt bổ sung hay biện pháp khắc phục hậu quả khác đối với hành vi này."
            )},
            {"role": "user", "content": user_content}
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=self._tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
