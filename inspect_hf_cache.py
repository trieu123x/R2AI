import os
import sys

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    print("HuggingFace Cache contents:")
    for item in os.listdir(cache_dir):
        print(" -", item)
else:
    print("HuggingFace cache directory not found at:", cache_dir)
