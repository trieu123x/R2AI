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
