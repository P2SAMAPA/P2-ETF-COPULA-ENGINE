"""
P2-ETF-COPULA-ENGINE  ·  upload.py
Push result files to the p2-etf-copula-results Hugging Face dataset.
Requires HF_TOKEN environment variable (set as GitHub Actions secret).
"""

from __future__ import annotations
import os
from huggingface_hub import HfApi

from config import HF_DATASET_OUT

_api = HfApi()


def upload_results(file_paths: list[str], repo_subdir: str = "results"):
    """
    Upload one or more local files to the HF dataset repository.

    Parameters
    ----------
    file_paths  : list of local file paths to upload
    repo_subdir : target subfolder inside the HF repo
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[upload] HF_TOKEN not set — skipping upload.")
        return

    for path in file_paths:
        if not os.path.exists(path):
            print(f"[upload] File not found, skipping: {path}")
            continue
        dest = f"{repo_subdir}/{os.path.basename(path)}"
        try:
            _api.upload_file(
                path_or_fileobj=path,
                path_in_repo=dest,
                repo_id=HF_DATASET_OUT,
                repo_type="dataset",
                token=token,
                commit_message=f"auto: update {os.path.basename(path)}",
            )
            print(f"[upload] ✓ {path}  →  {HF_DATASET_OUT}/{dest}")
        except Exception as e:
            print(f"[upload] ✗ Failed to upload {path}: {e}")
