"""Upload all project files to HuggingFace Space."""
import os
from huggingface_hub import HfApi, login

# --- Step 1: Login ---
token = input("Paste your HuggingFace token here (from https://huggingface.co/settings/tokens): ").strip()
login(token=token)

# --- Step 2: Upload ---
api = HfApi()
space_id = "SaishAmbar/token-economist-rl"

files_to_upload = [
    "environment.py",
    "agents.py",
    "train.py",
    "inference.py",
    "app.py",
    "client.py",
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "README.md",
    "results.png",
]

print(f"\nUploading {len(files_to_upload)} files to {space_id}...\n")

for fname in files_to_upload:
    fpath = os.path.join(os.path.dirname(__file__), fname)
    if os.path.exists(fpath):
        print(f"  Uploading {fname}...", end=" ")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=space_id,
            repo_type="space",
        )
        print("✓")
    else:
        print(f"  SKIP {fname} (not found)")

print(f"\n✅ Done! View your Space: https://huggingface.co/spaces/{space_id}")
print("Wait 2-3 minutes for Docker to rebuild, then re-submit!")
