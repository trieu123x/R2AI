import os
import time
from typing import Optional

class PipelineGenerator:
    """
    Step 8: Qwen3-8B
    Sinh câu trả lời dựa trên query đã xử lý và context đã được nén.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-8B-Instruct"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"

        print(f"[generator] Loading model '{self.model_name}' (Offline={is_offline})...", flush=True)
        t0 = time.time()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=is_offline
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            print(f"[generator] Device={device}, Dtype={dtype}...", flush=True)

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

    def generate_direct(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Sinh kết quả trực tiếp từ prompt (dùng cho Rewrite)."""
        self._lazy_load()
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([chat_prompt], return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def generate_answer(self, query: str, context: str, warning_msg: Optional[str] = None, max_new_tokens: int = 1024) -> str:
        """Sinh câu trả lời cuối cùng dựa trên context."""
        if not context.strip():
            return "Không tìm thấy tài liệu luật pháp liên quan để làm căn cứ trả lời."

        self._lazy_load()

        system_prompt = (
            "Bạn là một chuyên gia phân tích pháp lý cao cấp. Nhiệm vụ của bạn là trích dẫn thông tin từ ngữ cảnh (Context) được cung cấp để trả lời câu hỏi một cách chuẩn xác, khách quan.\n\n"
            "HƯỚNG DẪN TƯ DUY VÀ KHỬ LỖI:\n"
            "1. Trả lời trực tiếp: Đưa ra câu trả lời kết luận ngắn gọn, trực diện trong 1-2 câu. (Ví dụ: Nếu hỏi Có/Không thì khẳng định Có/Không kèm lý do ngắn; nếu hỏi 'Mức nào/Điều kiện gì/Bao lâu' thì chỉ ra ngay con số hoặc điều kiện cốt lõi nhất, KHÔNG ghi chung chung là 'Khẳng định Có').\n"
            "2. TUYỆT ĐỐI KHÔNG LẶP Ý: Không copy lại các câu chữ, nội dung đã viết ở phần trước xuống phần sau. Mỗi thông tin chỉ xuất hiện MỘT LẦN duy nhất trong toàn bộ văn bản.\n"
            "3. CẤM BỊA KÝ TỰ (PLACEHOLDER): Tuyệt đối không sử dụng các ký tự đại diện hoặc giữ chỗ như [X], [Y], [Z], [Nghị định...]. Nếu tài liệu không ghi rõ số điều/khoản/năm, hãy viết cụ thể bằng chữ: 'Theo quy định của văn bản...' hoặc 'Tài liệu không nêu rõ số điều'.\n"
            "4. ĐÚNG PHÂN LOẠI: Thông tin thuộc nhóm nào thì chỉ viết vào nhóm đó (Ví dụ: Tiền hỗ trợ đào tạo/quản trị không được xếp vào nhóm ưu đãi đất đai).\n\n"
            "Bạn BẮT BUỘC phải trình bày câu trả lời nghiêm ngặt theo đúng cấu trúc 4 phần sau (Không được tự ý thêm, bớt hoặc đổi tên phần):\n"
            "1. Trả lời trực tiếp: Đưa ra câu trả lời kết luận ngắn gọn, tổng quan trong từ 1 đến 2 câu (Ví dụ: Khẳng định Có/Không, Mức phạt cụ thể là bao nhiêu, hoặc Thời gian tối đa là bao lâu).\n"
            "2. Phân tích chi tiết: Liệt kê chi tiết các điều kiện, tiêu chuẩn hoặc các bước thực hiện dưới dạng các đầu dòng súc tích. TUYỆT ĐỐI không lặp lại câu kết luận đã viết ở phần 1.\n"
            "3. Căn cứ pháp lý: Chỉ rõ Tên văn bản luật và Số hiệu điều/khoản trích xuất được từ ngữ cảnh. Nếu ngữ cảnh không có, ghi rõ 'Chưa có căn cứ điều khoản cụ thể trong tài liệu'.\n"
            "4. Hạn chế của dữ liệu: Nêu rõ những thông tin mà câu hỏi yêu cầu nhưng tài liệu được cung cấp chưa làm rõ hoặc còn thiếu (Nếu tài liệu đã đầy đủ, ghi 'Không có')."
        )

        user_content = (
            f"TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
            f"=========================================\n"
            f"{context}\n"
            f"=========================================\n\n"
            f"{f'LƯU Ý QUAN TRỌNG TỪ HỆ THỐNG: {warning_msg}\\n\\n' if warning_msg else ''}"
            f"CÂU HỎI: {query}\n\n"
            f"Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên, suy luận logic để phân phối thông tin và trả lời chính xác theo cấu trúc 4 phần đã quy định ở trên."
        )

        messages = [
            {"role": "system", "content": system_prompt},
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
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.15,
            pad_token_id=self._tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
