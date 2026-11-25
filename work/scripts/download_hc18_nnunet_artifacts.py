# FIELD: HC18
# MODEL: nnU-Net
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

import os
import sys
import re
import time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

def download_file_with_resume(url: str, dest_path: Path, max_retries: int = 5):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    for attempt in range(max_retries):
        existing_size = temp_path.stat().st_size if temp_path.is_file() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
        else:
            mode = "wb"
            
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            if r.status_code == 416: # Range not satisfiable (already fully downloaded or invalid)
                if temp_path.is_file() and temp_path.stat().st_size > 0:
                    temp_path.replace(dest_path)
                    return
                # else restart
                mode = "wb"
                existing_size = 0
                r = requests.get(url, stream=True, timeout=30)
                
            r.raise_for_status()
            
            with open(temp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        
            temp_path.replace(dest_path)
            return
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in 2s...")
            time.sleep(2)
            
    raise RuntimeError(f"Failed to download {dest_path.name} after {max_retries} attempts.")

def main():
    target_base = Path(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\hc18_nnunet_f0")
    target_base.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    ApiReq = api.kernels_output.__globals__['ApiListKernelSessionOutputRequest']

    with api.build_kaggle_client() as kaggle:
        req = ApiReq()
        req.user_name = "prantakumerpandit390"
        req.kernel_slug = "hc18-nnunet-f0-250ep"
        req.page_size = 100
        token = None

        patterns = [
            r"^out/",
            r"plans\.json$",
            r"dataset\.json$",
            r"progress\.png$",
            r"training_log_.*\.txt$",
            r"summary\.json$",
            r"checkpoint_final\.pth$",
            r"checkpoint_best\.pth$",
        ]
        compiled = [re.compile(p) for p in patterns]

        to_download = []
        while True:
            req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in (resp.files or []):
                fname = item.file_name
                if any(cp.search(fname) for cp in compiled):
                    to_download.append((fname, item.url, getattr(item, 'total_bytes', None) or getattr(item, 'size', None)))
            token = resp.next_page_token
            if not token:
                break

        print(f"Scanned output pages. Found {len(to_download)} target files to download.")

    # Sort small files first so masks and plans download in seconds, then the two large checkpoints
    to_download.sort(key=lambda x: (x[0].endswith(".pth"), x[0]))

    downloaded = 0
    for fname, url, sz in to_download:
        dest_path = target_base / fname
        if dest_path.is_file() and dest_path.stat().st_size > 0:
            if not fname.endswith(".pth") or dest_path.stat().st_size > 300_000_000:
                continue

        print(f"Downloading {fname} (expected {sz} bytes)...")
        download_file_with_resume(url, dest_path)
        downloaded += 1
        print(f"  Saved {dest_path} ({dest_path.stat().st_size} bytes)")

    print(f"Done. Downloaded {downloaded} new files.")

if __name__ == "__main__":
    main()
