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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.retriever import RetrievalResult

# ── System Prompt dùng chung cho cả single và batch generation ────────────────
_SYSTEM_PROMPT = (
    "Bạn là một trợ lý pháp lý Việt Nam chuyên nghiệp, chính xác và đáng tin cậy.\n\n"
    "Nhiệm vụ của bạn:\n"
    "- Trả lời câu hỏi dựa trên CÁC CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP. Tuyệt đối không tự suy diễn hoặc giả định các nội dung không có trong tài liệu.\n"
    "- Hãy viết hoàn toàn bằng tiếng Việt chuẩn mực. Tuyệt đối KHÔNG trộn lẫn tiếng Trung (chữ Hán), tiếng Anh hoặc bất kỳ từ ngữ nước ngoài nào khác.\n"
    "- Tuyệt đối KHÔNG được tự bịa ra các số hiệu văn bản, điều luật không có trong các căn cứ pháp lý.\n"
    "- Tuyệt đối KHÔNG sử dụng các ký tự thay thế hoặc giữ chỗ như [X], [Y], [Z]. "
    "Nếu tài liệu không nêu rõ số điều/khoản, hãy trích dẫn bằng nội dung chữ hoặc ghi 'Theo quy định của pháp luật hiện hành'.\n"
    "- Hãy tổng hợp thông tin ngắn gọn, súc tích. Không lặp lại cùng một ý ở các mục khác nhau.\n"
    "- Chỉ trích dẫn thông tin trả lời trực tiếp cho câu hỏi. Không đưa thông tin của đối tượng khác vào nếu câu hỏi hỏi về chủ thể cụ thể.\n"
    "- Tuyệt đối KHÔNG sử dụng tên văn bản, điều luật hoặc các từ ngữ trong Ví dụ mẫu (Few-shot) để trả lời cho câu hỏi thực tế.\n"
    "- Cực kỳ cẩn thận với các từ phủ định hoặc hành vi cấm: 'nghiêm cấm', 'không được', 'xử phạt đối với hành vi' có nghĩa là hành vi đó BỊ CẤM.\n"
    "- Không chỉ nêu tên văn bản chung chung. Hãy giải thích cụ thể nội dung quyền lợi, nghĩa vụ, ưu đãi hoặc mức phạt.\n"
    "- Khi câu hỏi hỏi về xử phạt, BẮT BUỘC phải nêu cả biện pháp khắc phục hậu quả (nếu có trong tài liệu).\n\n"
    "Bạn BẮT BUỘC phải trình bày câu trả lời nghiêm ngặt theo cấu trúc 4 phần sau:\n"
    "1. Trả lời trực tiếp: Trả lời trực tiếp vào câu hỏi, nêu rõ kết luận chính hoặc hành vi và hệ quả.\n"
    "2. Phân tích chi tiết: Diễn giải chi tiết nội dung quy định, mức phạt bằng tiền cụ thể, quyền lợi/nghĩa vụ chi tiết, biện pháp khắc phục hậu quả (nếu có).\n"
    "3. Căn cứ pháp lý: Liệt kê chi tiết các điều khoản cụ thể từ tài liệu tham khảo (Ví dụ: 'Điều 17 Bộ luật Lao động 2019', 'Khoản 1 Điều 8 Nghị định 28/2020/NĐ-CP').\n"
    "4. Hạn chế của dữ liệu (nếu có): Nêu rõ nếu các căn cứ pháp lý thiếu thông tin để trả lời đầy đủ một khía cạnh nào đó của câu hỏi."
)

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

    # Hỗ trợ tìm kiếm cả số và chữ cái hậu tố (Ví dụ: 15, 15a, 15b)
    match_num = re.search(r'(\d+[a-zA-Z]?)', article_hint)
    if not match_num:
        return content.strip()

    art_num = match_num.group(1)
    
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
        is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"

        # Tắt flashinfer JIT (tránh lỗi -lcuda trên Kaggle CUDA 13.x)
        # Không set VLLM_ATTENTION_BACKEND → vLLM tự chọn backend tối ưu cho T4 (sm_75)
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        # Check if we can use vllm
        use_vllm = False
        if torch.cuda.is_available():
            try:
                from vllm import LLM
                use_vllm = True
            except ImportError:
                use_vllm = False

        if use_vllm:
            print(f"[generator] Loading model '{self.model_name}' with vLLM (Offline={is_offline})...", flush=True)
            t0 = time.time()
            try:
                cc_major = torch.cuda.get_device_capability()[0]
                dtype = "bfloat16" if cc_major >= 8 else "float16"
                num_gpus = torch.cuda.device_count()
                
                # Check VRAM to set memory utilization
                free_vram_gb = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"[generator] GPU free VRAM: {free_vram_gb:.1f} GB", flush=True)
                
                gpu_memory_utilization = 0.85
                
                from vllm import LLM
                self._model = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
                    enforce_eager=False
                )
                self._tokenizer = self._model.get_tokenizer()
                self._tokenizer.padding_side = 'left'
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                self._is_vllm = True
                print(f"[generator] Model loaded with vLLM in {time.time() - t0:.1f}s", flush=True)
                return
            except Exception as e:
                print(f"[generator] Failed to load model with vLLM: {e}. Falling back to transformers...", flush=True)

        # Fallback to standard transformers
        self._is_vllm = False
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[generator] Loading model '{self.model_name}' with Transformers (Offline={is_offline})...", flush=True)
        t0 = time.time()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=is_offline,
                use_fast=False
            )
            
            # Cấu hình padding để hỗ trợ batch inference
            self._tokenizer.padding_side = 'left'
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            # Kiểm tra VRAM khả dụng — tránh để accelerate offload xuống disk
            use_gpu = False
            if torch.cuda.is_available():
                free_vram_gb = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"[generator] GPU free VRAM: {free_vram_gb:.1f} GB", flush=True)
                # Qwen2.5-1.5B cần ~3GB, Qwen3-8B cần ~16GB (float16)
                # Nếu ít hơn 2.5GB thì dùng CPU hẳn load và disk offloading
                use_gpu = free_vram_gb >= 2.5

            max_memory = None
            if use_gpu:
                cc_major = torch.cuda.get_device_capability()[0]
                dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
                device_map = "auto"
                
                # Cấu hình max_memory cho đa GPU (Kaggle T4 x 2) để tránh GPU 0 bị quá tải KV cache
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    max_memory = {}
                    for gpu_id in range(num_gpus):
                        if gpu_id == 0:
                            max_memory[gpu_id] = "9GiB"  # Giới hạn GPU 0 ở 9GB, dành 5GB+ cho KV cache
                        else:
                            max_memory[gpu_id] = "13GiB" # GPU 1 chứa phần còn lại
                    max_memory["cpu"] = "32GiB"
                    print(f"[generator] Chế độ: GPU đa card (dtype={dtype}, device_map=auto, max_memory={max_memory})", flush=True)
                else:
                    print(f"[generator] Chế độ: GPU đơn (dtype={dtype}, device_map=auto)", flush=True)
            else:
                dtype = torch.float32
                device_map = "cpu"
                print(f"[generator] Chế độ: CPU (float32) — sẽ chậm hơn nhưng tránh disk offloading", flush=True)

            self._device = "cuda" if use_gpu else "cpu"

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                local_files_only=is_offline,
                attn_implementation="sdpa" if use_gpu else "eager",  # sdpa không tương thích CPU tốt
            )
            print(f"[generator] Model loaded in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[generator] Failed to load model: {e}")
            raise

    def generate_answer(self, query: str, results: List[RetrievalResult], max_new_tokens: int = 1028, warning_msg: Optional[str] = None) -> str:
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

        # Hệ thống Prompt hướng dẫn cực kỳ rõ ràng cho Qwen
        system_prompt = _SYSTEM_PROMPT

        user_content = (
            f"TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
            f"=========================================\n"
            f"{context}\n"
            f"=========================================\n\n"
            f"{f'LƯU Ý QUAN TRỌNG TỪ HỆ THỐNG: {warning_msg}\n\n' if warning_msg else ''}"
            f"CÂU HỎI: {query}\n\n"
            f"Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            # Few-shot example — dùng số điều thật, KHÔNG dùng placeholder [X][Y][Z]
            {"role": "user", "content": (
                "TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
                "=========================================\n"
                "[Căn cứ 1]\n"
                "Độ liên quan: 0.9876\n"
                "Văn bản: Nghị định số 12/2022/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực lao động\n"
                "Điều: Điều 34\n"
                "Nội dung:\n"
                "Điều 34. Vi phạm về quyền thành lập, gia nhập và hoạt động công đoàn của người lao động\n"
                "1. Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập, gia nhập và hoạt động công đoàn.\n"
                "2. Biện pháp khắc phục hậu quả: Buộc người sử dụng lao động thu hồi quyết định, chấm dứt hành vi cản trở và đảm bảo quyền thành lập công đoàn của người lao động.\n"
                "=========================================\n\n"
                "CÂU HỎI: Hành vi cản trở người lao động thành lập công đoàn bị phạt bao nhiêu tiền và phải khắc phục ra sao?\n\n"
                "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên."
            )},
            {"role": "assistant", "content": (
                "1. Trả lời trực tiếp: Hành vi cản trở người lao động thành lập công đoàn sẽ bị phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng và buộc thu hồi quyết định, chấm dứt hành vi vi phạm.\n\n"
                "2. Phân tích chi tiết:\n"
                "- Về mức phạt: Khoản 1 Điều 34 Nghị định 12/2022/NĐ-CP quy định phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập, gia nhập và hoạt động công đoàn.\n"
                "- Về biện pháp khắc phục: Người sử dụng lao động bị buộc thu hồi quyết định vi phạm và bảo đảm quyền thành lập công đoàn của người lao động theo Khoản 2 Điều 34 cùng Nghị định.\n\n"
                "3. Căn cứ pháp lý:\n"
                "- Khoản 1, Khoản 2 Điều 34 Nghị định số 12/2022/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực lao động.\n\n"
                "4. Hạn chế của dữ liệu (nếu có):\n"
                "- Tài liệu không nêu mức phạt đối với tái phạm hoặc vi phạm quy mô lớn."
            )},
            {"role": "user", "content": user_content}
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Tắt Qwen3 thinking mode để tránh loop vô hạn
            )
        except TypeError:
            # Fallback cho tokenizer không hỗ trợ enable_thinking (Qwen2.x)
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if getattr(self, '_is_vllm', False):
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.1,
            )
            outputs = self._model.generate([prompt], sampling_params, use_tqdm=False)
            response = outputs[0].outputs[0].text
            return response.strip()

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self._tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def generate_batch_answers(self, queries: List[str], results_list: List[List[RetrievalResult]], max_new_tokens: int = 1028, warning_msgs: Optional[List[Optional[str]]] = None) -> List[str]:
        """
        Sinh câu trả lời cho một batch câu hỏi cùng lúc để tận dụng tối đa sức mạnh tính toán song song của GPU.
        """
        if not queries:
            return []

        self._lazy_load()

        if warning_msgs is None:
            warning_msgs = [None] * len(queries)

        prompts = []
        for query, results, warning_msg in zip(queries, results_list, warning_msgs):
            if not results:
                # Nếu không có kết quả truy xuất, tạo prompt rỗng hoặc báo lỗi
                context = ""
            else:
                context_parts = []
                for idx, r in enumerate(results, start=1):
                    extracted_body = extract_evidence(r.content, r.article_hint)
                    lines = extracted_body.split('\n')
                    body_lines = [line for line in lines if not line.strip().startswith("Văn bản:")]
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

            user_content = (
                f"TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
                f"=========================================\n"
                f"{context}\n"
                f"=========================================\n\n"
                f"{f'LƯU Ý QUAN TRỌNG TỪ HỆ THỐNG: {warning_msg}\n\n' if warning_msg else ''}"
                f"CÂU HỎI: {query}\n\n"
                f"Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên.\n"
            )

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                # Few-shot example — dùng số điều thật, KHÔNG dùng placeholder [X][Y][Z]
                {"role": "user", "content": (
                    "TÀI LIỆU THAM KHẢO CUNG CẤP:\n"
                    "=========================================\n"
                    "[Căn cứ 1]\n"
                    "Độ liên quan: 0.9876\n"
                    "Văn bản: Nghị định số 12/2022/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực lao động\n"
                    "Điều: Điều 34\n"
                    "Nội dung:\n"
                    "Điều 34. Vi phạm về quyền thành lập, gia nhập và hoạt động công đoàn của người lao động\n"
                    "1. Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập, gia nhập và hoạt động công đoàn.\n"
                    "2. Biện pháp khắc phục hậu quả: Buộc người sử dụng lao động thu hồi quyết định, chấm dứt hành vi cản trở và đảm bảo quyền thành lập công đoàn của người lao động.\n"
                    "=========================================\n\n"
                    "CÂU HỎI: Hành vi cản trở người lao động thành lập công đoàn bị phạt bao nhiêu tiền và phải khắc phục ra sao?\n\n"
                    "Yêu cầu trả lời: Hãy phân tích kỹ tài liệu tham khảo trên và trả lời câu hỏi tuân thủ đúng cấu trúc 4 phần nêu trên."
                )},
                {"role": "assistant", "content": (
                    "1. Trả lời trực tiếp: Hành vi cản trở người lao động thành lập công đoàn sẽ bị phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng và buộc thu hồi quyết định, chấm dứt hành vi vi phạm.\n\n"
                    "2. Phân tích chi tiết:\n"
                    "- Về mức phạt: Khoản 1 Điều 34 Nghị định 12/2022/NĐ-CP quy định phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi cản trở người lao động thành lập, gia nhập và hoạt động công đoàn.\n"
                    "- Về biện pháp khắc phục: Người sử dụng lao động bị buộc thu hồi quyết định vi phạm và bảo đảm quyền thành lập công đoàn của người lao động theo Khoản 2 Điều 34 cùng Nghị định.\n\n"
                    "3. Căn cứ pháp lý:\n"
                    "- Khoản 1, Khoản 2 Điều 34 Nghị định số 12/2022/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực lao động.\n\n"
                    "4. Hạn chế của dữ liệu (nếu có):\n"
                    "- Tài liệu không nêu mức phạt đối với tái phạm hoặc vi phạm quy mô lớn."
                )},
                {"role": "user", "content": user_content}
            ]

            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Tắt Qwen3 thinking mode
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(prompt)

        if getattr(self, '_is_vllm', False):
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.1,
            )
            outputs = self._model.generate(prompts, sampling_params, use_tqdm=False)
            responses = [out.outputs[0].text.strip() for out in outputs]
            return responses

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._model.device)

        # Trên CPU, batch inference với padding lớn rất chậm/bị treo. Chạy tuần tự thay thế.
        if getattr(self, '_device', 'cuda') == 'cpu':
            print(f"[generator] CPU mode: đang sinh tuần tự {len(prompts)} câu...", flush=True)
            results = []
            for i, prompt in enumerate(prompts):
                inp = self._tokenizer([prompt], return_tensors="pt")
                out = self._model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self._tokenizer.eos_token_id
                )
                gen_ids = out[0][inp.input_ids.shape[1]:]
                text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(text.strip())
                print(f"[generator]   [{i+1}/{len(prompts)}] xong", flush=True)
            return results

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self._tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        responses = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]
