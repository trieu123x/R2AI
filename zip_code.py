import os
import zipfile

SRC_DIR = "src"
ZIP_NAME = "r2ai_code.zip"

# Directories to skip entirely
EXCLUDE_DIRS = {
    "__pycache__",
    "bge-reranker-v2-m3",   # BAAI reranker model (large binary)
}

# File extensions to skip
EXCLUDE_EXTENSIONS = {
    ".db", ".index", ".pyc", ".pyo", ".json",
}

def zip_src(zip_filename=ZIP_NAME, src_dir=SRC_DIR):
    print(f"[zip] Creating '{zip_filename}' from '{src_dir}/' ...")
    count = 0
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            # Skip unwanted directories (in-place so os.walk won't recurse into them)
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in EXCLUDE_EXTENSIONS:
                    continue

                filepath = os.path.join(dirpath, filename)
                # Archive path relative to parent of src/
                # -> src/retrieval/batch_retrieve.py etc.
                arcname = os.path.relpath(filepath, start=".")
                zipf.write(filepath, arcname)
                print(f"  + {arcname}")
                count += 1

    size_kb = os.path.getsize(zip_filename) / 1024
    print(f"\n[zip] Done! {count} files -> '{zip_filename}' ({size_kb:.1f} KB)")

if __name__ == "__main__":
    zip_src()
