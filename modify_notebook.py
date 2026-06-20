import json
import os

filepath = 'C:\\Users\\admin\\Downloads\\R2AI\\kaggle_inference.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'pipeline = LegalRAGPipeline(' in line:
                # Need to add reranker_model_name
                # Let's see if we can find the end of this call
                pass
            if 'llm_model_name=\"Qwen/Qwen2.5-1.5B-Instruct\"' in line:
                source[i] = line.replace('llm_model_name=\"Qwen/Qwen2.5-1.5B-Instruct\"', 'llm_model_name=\"Qwen/Qwen2.5-1.5B-Instruct\",\\n        reranker_model_name=\"/kaggle/input/bge-reranker-v2-m3\" # <-- Đổi đường dẫn này nếu bạn đặt tên dataset khác')

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print('Notebook updated')
