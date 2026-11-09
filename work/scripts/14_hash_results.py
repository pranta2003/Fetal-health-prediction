"""14_hash_results.py: Generate or verify SHA256SUMS for results directory.

Usage:
    python scripts/14_hash_results.py
    python scripts/14_hash_results.py --dir docs/results/hc18
    python scripts/14_hash_results.py --verify --dir docs/results/hc18
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

SUMS_FILENAME = "SHA256SUMS"
TEXT_EXTENSIONS = {".json", ".md", ".csv", ".txt", ".log", ".yaml", ".yml"}


def compute_sha256(filepath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file, normalizing CRLF to LF for text files.
    
    This ensures that SHA256SUMS digests remain strictly deterministic across
    Windows (CRLF) and Linux/Git (LF) checkouts.
    """
    filepath = Path(filepath)
    h = hashlib.sha256()
    if filepath.suffix.lower() in TEXT_EXTENSIONS:
        with open(filepath, "rb") as f:
            content = f.read()
        content = content.replace(b"\r\n", b"\n")
        h.update(content)
    else:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
    return h.hexdigest()


def generate_sums(target_dir: str | Path) -> Path:
    """Compute sha256 for all files in target_dir (recursively) and write SHA256SUMS."""
    target_dir = Path(target_dir).resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    sums_path = target_dir / SUMS_FILENAME
    entries: list[tuple[str, str]] = []

    for item in sorted(target_dir.rglob("*")):
        if item.is_file() and item.name != SUMS_FILENAME:
            sha = compute_sha256(item)
            rel_name = item.relative_to(target_dir).as_posix()
            entries.append((sha, rel_name))

    with open(sums_path, "w", encoding="utf-8", newline="\n") as f:
        for sha, fname in entries:
            f.write(f"{sha}  {fname}\n")

    print(f"Generated {sums_path} with {len(entries)} file entries:")
    for sha, fname in entries:
        print(f"  {sha}  {fname}")

    return sums_path


def verify_sums(target_dir: str | Path) -> bool:
    """Verify existing SHA256SUMS in target_dir."""
    target_dir = Path(target_dir).resolve()
    sums_path = target_dir / SUMS_FILENAME
    if not sums_path.exists():
        raise FileNotFoundError(f"SHA256SUMS not found at {sums_path}")

    all_ok = True
    with open(sums_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Verifying {sums_path} ({len(lines)} entries)...")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        exp_sha, fname = parts
        # Strip potential leading asterisk from binary mode sha256sum
        fname = fname.lstrip("*")
        file_path = target_dir / fname
        if not file_path.exists():
            print(f"  MISSING: {fname}")
            all_ok = False
            continue
        act_sha = compute_sha256(file_path)
        if act_sha.lower() == exp_sha.lower():
            print(f"  OK: {fname}")
        else:
            print(f"  FAILED: {fname} (expected {exp_sha}, got {act_sha})")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Generate or verify SHA256SUMS for results.")
    parser.add_argument(
        "--dir",
        type=str,
        default="docs/results/hc18",
        help="Path to results directory (default: docs/results/hc18)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing SHA256SUMS instead of writing",
    )
    args = parser.parse_args()

    if args.verify:
        success = verify_sums(args.dir)
        if not success:
            sys.exit(1)
    else:
        generate_sums(args.dir)


if __name__ == "__main__":
    main()
