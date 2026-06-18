import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fix_file_paths(file_path):
    # Compute relative depth from project root
    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
    parts = rel_path.split(os.sep)
    depth = len(parts) - 1
    
    # We want to replace PROJECT_ROOT definition
    # e.g., if depth is 2 (like src/retrieval/retriever.py), it should go up 3 levels
    # os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    levels = ["os.path.abspath(__file__)"]
    for _ in range(depth + 1):
        levels.append(f"os.path.dirname({levels[-1]})")
    
    target_expr = levels[-1]
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find any line that defines PROJECT_ROOT
    lines = content.splitlines()
    modified = False
    for i, line in enumerate(lines):
        if line.strip().startswith("PROJECT_ROOT ="):
            # Replace whatever is after = with the target_expr
            # e.g., PROJECT_ROOT = os.path.dirname(...)
            # But let's keep it simple: PROJECT_ROOT = target_expr
            lines[i] = f"PROJECT_ROOT = {target_expr}"
            modified = True
            
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[fix_paths] Updated PROJECT_ROOT in {rel_path} to go up {depth + 1} levels")

def main():
    # Walk through all python files in src/
    src_dir = os.path.join(PROJECT_ROOT, "src")
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                fix_file_paths(os.path.join(root, file))
                
    # Also walk through scratch/
    scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
    for root, dirs, files in os.walk(scratch_dir):
        for file in files:
            if file.endswith(".py"):
                fix_file_paths(os.path.join(root, file))

if __name__ == "__main__":
    main()
