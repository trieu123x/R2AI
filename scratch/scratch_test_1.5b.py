import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print("Loading model:", model_name)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    print("SUCCESS: Model loaded on device:", model.device)
except Exception as e:
    print("FAILED:", str(e))
