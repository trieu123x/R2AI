import time
import os
import zipfile

log_path = "batch_run.log"
target_json = "results.json"
zip_name = "results.zip"

print(f"Monitoring '{log_path}' to create '{zip_name}'...")

while True:
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "Đã ghi → results.json" in content or "✓ Xong!" in content:
                print("Batch retrieve completed successfully detected in log!")
                break
            # Also check if the python process died or finished
            # We can check if results.json exists and is non-empty
            if os.path.exists(target_json) and os.path.getsize(target_json) > 100000:
                print("results.json exists and looks complete!")
                break
    time.sleep(15)

# Wait 5 more seconds to ensure file is fully closed/flushed
time.sleep(5)

if os.path.exists(target_json):
    print(f"Creating flat zip file '{zip_name}' containing '{target_json}'...")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(target_json, os.path.basename(target_json))
    print("Zip file created successfully!")
else:
    print(f"Error: {target_json} does not exist!")
