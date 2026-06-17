import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("test_configs.log", "r", encoding="utf-16le") as f:
    lines = f.readlines()

print("--- Print around line 47 ---")
for j in range(47, min(90, len(lines))):
    print(f"{j:3d}: {lines[j].strip()}")
