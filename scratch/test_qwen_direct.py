import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa"
)
print("Model loaded.")

prompt = "Hello, how are you today?"
messages = [{"role": "user", "content": prompt}]
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt_str], return_tensors="pt").to(model.device)

print("Generating...")
t0 = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
print(f"Done in {time.time()-t0:.2f}s")
print(tokenizer.decode(outputs[0]))
