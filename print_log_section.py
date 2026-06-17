import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("test_configs.log", "r", encoding="utf-16le") as f:
    text = f.read()

# Let's print sections that match "Mode: vector"
pattern = r"==========================================\s*Mode: vector.*?(?===\n|$)"
matches = re.findall(pattern, text, re.DOTALL)
for match in matches:
    print(match)
    print("-" * 60)
