import re

with open("test_configs.log", "r", encoding="utf-16le") as f:
    text = f.read()

# Let's search for "Mode: vector" and print the blocks
blocks = re.split(r"===+", text)
for b in blocks:
    if "Mode: vector" in b or "Mode: hybrid" in b:
        print("=" * 40)
        print(b.strip())
