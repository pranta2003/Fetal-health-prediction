#!/usr/bin/env python3

# FIELD: AC
# MODEL: EfficientNet-B4 Landmark U-Net
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

"""scripts/67_download_f1f2_and_ac_fassd.py

Targeted, resumable downloader for:
1. hc18-nnunet-f1f2-100ep (masks, logs, plans, and checkpoints in out/)
2. ac-fassd-s42 (pooled CSV, summary JSON, fold results)
"""

import os
import re
import sys
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
            if r.status_code == 416:  # Range not satisfiable (already done or invalid)
                if temp_path.is_file() and temp_path.stat().st_size > 0:
                    temp_path.replace(dest_path)
                    return
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
            print(f"  Attempt {attempt+1} failed for {dest_path.name}: {e}. Retrying in 2s...")
            time.sleep(2)

    raise RuntimeError(f"Failed to download {dest_path.name} after {max_retries} attempts.")


def download_kernel_outputs(
    api: KaggleApi,
    owner_slug: str,
    kernel_slug: str,
    target_base: Path,
    patterns: list[str] | None = None,
):
    target_base.mkdir(parents=True, exist_ok=True)
    ApiReq = api.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]

    compiled = [re.compile(p) for p in patterns] if patterns else None

    print(f"\nScanning outputs for {owner_slug}/{kernel_slug}...")
    to_download = []
    with api.build_kaggle_client() as kaggle:
        req = ApiReq()
        req.user_name = owner_slug
        req.kernel_slug = kernel_slug
        req.page_size = 100
        token = None

        while True:
            req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in resp.files or []:
                fname = item.file_name
                if compiled is None or any(cp.search(fname) for cp in compiled):
                    sz = getattr(item, "total_bytes", None) or getattr(item, "size", None)
                    to_download.append((fname, item.url, sz))
            token = resp.next_page_token
            if not token:
                break

    print(f"Found {len(to_download)} matching files to download for {kernel_slug}.")
    # Sort: smaller files first (non-checkpoints), checkpoints last
    to_download.sort(key=lambda x: (x[0].endswith(".pth") or x[0].endswith(".pt"), x[0]))

    downloaded = 0
    for fname, url, sz in to_download:
        dest_path = target_base / fname
        if dest_path.is_file() and dest_path.stat().st_size > 0:
            if sz is not None and dest_path.stat().st_size == sz:
                continue
            elif sz is None:
                continue

        print(f"Downloading {fname} (size: {sz})...")
        download_file_with_resume(url, dest_path)
        downloaded += 1
        print(f"  Saved {dest_path.name} ({dest_path.stat().st_size} bytes)")

    print(f"Completed {kernel_slug}. Downloaded {downloaded} new files.")


def main():
    api = KaggleApi()
    api.authenticate()

    # 2. Download HC18 nnU-Net f1f2 output (targeted)
    hc18_target = Path(r"E:\BUBT-CSE-4th yr\Capstone\kaggle_outputs\hc18_nnunet_f1f2")
    download_kernel_outputs(
        api=api,
        owner_slug="prantakumerpandit390",
        kernel_slug="hc18-nnunet-f1f2-100ep",
        target_base=hc18_target,
        patterns=[
            r"^out/",
            r"train_fold_.*\.log$",
            r"plans\.json$",
            r"dataset\.json$",
            r"splits_final\.json$",
            r"checkpoint_.*\.pth$",
            r"validation_fold",
        ],
    )


if __name__ == "__main__":
    main()
