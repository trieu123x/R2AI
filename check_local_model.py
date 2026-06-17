import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
print("Cache dir path:", cache_dir)
if os.path.exists(cache_dir):
    print("Files inside:")
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            print(" -", os.path.join(root, f))
            
# Try loading directly
try:
    print("\nTrying to load model via transformers from local cache...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    print("Success loading model!")
except Exception as e:
    print("Error:", e)
