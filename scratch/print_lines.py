import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("test_configs.log", "r", encoding="utf-16le") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "Mode: vector" in line or "Mode: hybrid" in line:
        print(f"--- Found at line {idx} ---")
        for j in range(idx, min(idx + 30, len(lines))):
            print(lines[j].strip())
