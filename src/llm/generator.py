import os
import time
from typing import Optional

class PipelineGenerator:
    """
    Step 8: Qwen3-8B (vLLM optimized)
    Sinh câu trả lời dựa trên query đã xử lý và context đã được nén.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-8B-Instruct"):
        self.model_name = model_name
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return

        import torch
        # Tắt flashinfer JIT compilation để tránh lỗi "-lcuda not found" trên Kaggle CUDA 13.x
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        from vllm import LLM

        is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        
        # Tắt FlashInfer Sampler và ép dùng XFORMERS để tránh lỗi build C++ trên Kaggle T4
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

        print(f"[generator] Loading model '{self.model_name}' with vLLM (Offline={is_offline})...", flush=True)
        t0 = time.time()

        try:
            cc_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
            dtype = "bfloat16" if cc_major >= 8 else "float16"

            self._model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype=dtype,
                gpu_memory_utilization=0.4,
                enforce_eager=False
            )
            print(f"[generator] Model loaded in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[generator] Failed to load model: {e}")
            raise

    def generate_direct(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Sinh kết quả trực tiếp từ prompt (dùng cho Rewrite)."""
        self._lazy_load()
        from vllm import SamplingParams
        
        messages = [{"role": "user", "content": prompt}]
        tokenizer = self._model.get_tokenizer()
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        outputs = self._model.generate([chat_prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def generate_answer(self, query: str, context: str, warning_msg: Optional[str] = None, max_new_tokens: int = 1024) -> str:
        """Sinh câu trả lời cuối cùng dựa trên context."""
        if not context.strip():
            return "Không tìm thấy tài liệu luật pháp liên quan để làm căn cứ trả lời."

        self._lazy_load()
        from vllm import SamplingParams
        from src.prompts.prompt_templates import SYSTEM_PROMPT, USER_CONTENT_TEMPLATE

        system_prompt = SYSTEM_PROMPT

        warning_str = f"LƯU Ý QUAN TRỌNG TỪ HỆ THỐNG: {warning_msg}\n\n" if warning_msg else ""
        user_content = USER_CONTENT_TEMPLATE.format(
            context=context,
            warning_msg=warning_str,
            query=query
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        tokenizer = self._model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.1,
            repetition_penalty=1.15
        )
        
        outputs = self._model.generate([prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        return response.strip()
