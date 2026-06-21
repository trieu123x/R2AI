import json
import re

notebook_path = r'c:\Users\admin\Downloads\R2AI\R2AIII.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', '')
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = source
        
        # 1. Modify pip install to include vllm
        if '!pip install' in source_str:
            if 'vllm' not in source_str:
                source_str = source_str.replace('accelerate bitsandbytes', 'accelerate bitsandbytes vllm')
                if isinstance(source, list):
                    # update list
                    new_source = []
                    for line in source:
                        new_source.append(line.replace('accelerate bitsandbytes', 'accelerate bitsandbytes vllm'))
                    cell['source'] = new_source
                else:
                    cell['source'] = source_str
                print("Updated pip install command.")
                
        # 2. Modify PipelineGenerator
        if 'class PipelineGenerator:' in source_str and 'transformers' in source_str:
            new_generator_code = """class PipelineGenerator:
    \"\"\"
    Step 8: Qwen3-8B (vLLM optimized)
    Sinh câu trả lời dựa trên query đã xử lý và context đã được nén.
    \"\"\"
    def __init__(self, model_name: str = "Qwen/Qwen3-8B-Instruct"):
        self.model_name = model_name
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return

        import torch
        from vllm import LLM

        is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"

        print(f"[generator] Loading model '{self.model_name}' with vLLM (Offline={is_offline})...", flush=True)
        t0 = time.time()

        try:
            dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

            # Sử dụng gpu_memory_utilization = 0.4 để chừa VRAM cho Retriever và Reranker (Embedding, CrossEncoder)
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
        \"\"\"Sinh kết quả trực tiếp từ prompt (dùng cho Rewrite).\"\"\"
        self._lazy_load()
        from vllm import SamplingParams
        
        messages = [{"role": "user", "content": prompt}]
        tokenizer = self._model.get_tokenizer()
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        outputs = self._model.generate([chat_prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def generate_answer(self, query: str, context: str, warning_msg: Optional[str] = None, max_new_tokens: int = 1024) -> str:
        \"\"\"Sinh câu trả lời cuối cùng dựa trên context.\"\"\"
        if not context.strip():
            return "Không tìm thấy tài liệu luật pháp liên quan để làm căn cứ trả lời."

        self._lazy_load()
        from vllm import SamplingParams

        system_prompt = SYSTEM_PROMPT

        warning_str = f"LƯU Ý QUAN TRỌNG TỪ HỆ THỐNG: {warning_msg}\\n\\n" if warning_msg else ""
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
"""
            # Replace the old PipelineGenerator class with the new one
            # The old code is from `class PipelineGenerator:` until `# ==========================================`
            # We can use regex to replace the old block
            pattern = re.compile(r'class PipelineGenerator:.*?# ==========================================\n# File: retrieval/query_rewrite\.py', re.DOTALL)
            replacement = new_generator_code + "\n# ==========================================\n# File: retrieval/query_rewrite.py"
            
            if isinstance(source, list):
                source_str = "".join(source)
                source_str = pattern.sub(replacement, source_str)
                # Convert back to list of lines
                lines = source_str.splitlines(keepends=True)
                cell['source'] = lines
            else:
                cell['source'] = pattern.sub(replacement, source_str)
            print("Updated PipelineGenerator with vLLM.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    # The indent doesn't perfectly match jupyter's standard but it's valid JSON.
    # To match jupyter, we can use indent=1 or indent=2. Jupyter usually uses 1 or 2.
    
print("Notebook updated successfully.")
