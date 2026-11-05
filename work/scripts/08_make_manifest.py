"""08_make_manifest.py: Build or verify SHA-256 manifest of model checkpoints directory.

Usage:
    # Generate MANIFEST.sha256 in model directory:
    python scripts/08_make_manifest.py

    # Verify existing MANIFEST.sha256:
    python scripts/08_make_manifest.py --verify

    # Specify custom model directory:
    python scripts/08_make_manifest.py --dir C:\\fetal-models
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fetal import config


MANIFEST_FILENAME = "MANIFEST.sha256"


def compute_sha256(filepath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(model_dir: str | Path) -> Path:
    """Walk model_dir, compute size and sha256 for all files, and write MANIFEST.sha256."""
    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / MANIFEST_FILENAME

    entries = []
    for root, _, files in os.walk(model_dir):
        for fname in sorted(files):
            if fname == MANIFEST_FILENAME:
                continue
            full_path = Path(root) / fname
            rel_path = full_path.relative_to(model_dir).as_posix()
            size = full_path.stat().st_size
            sha = compute_sha256(full_path)
            entries.append((sha, size, rel_path))

    entries.sort(key=lambda x: x[2])

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"# SHA-256 Manifest for model directory: {model_dir}\n")
        f.write("# Format: SHA256  SIZE_BYTES  RELATIVE_PATH\n")
        for sha, size, rel_path in entries:
            f.write(f"{sha}  {size}  {rel_path}\n")

    print(f"[Manifest] Generated {manifest_path} with {len(entries)} file entries.")
    return manifest_path


def verify_manifest(model_dir: str | Path) -> tuple[bool, list[dict]]:
    """Verify existing MANIFEST.sha256 against actual files in model_dir.

    Returns:
        (all_ok, list_of_result_dicts)
    """
    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / MANIFEST_FILENAME

    if not manifest_path.is_file():
        print(f"[Manifest] Manifest file not found: {manifest_path}")
        return False, []

    results = []
    all_ok = True

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                continue
            exp_sha, exp_size_str, rel_path = parts
            exp_size = int(exp_size_str)

            file_path = model_dir / Path(rel_path)
            if not file_path.exists():
                print(f"[MISSING]  {rel_path}")
                results.append(dict(rel_path=rel_path, status="MISSING", expected_sha=exp_sha, actual_sha=None))
                all_ok = False
                continue

            act_size = file_path.stat().st_size
            act_sha = compute_sha256(file_path)

            if act_size == exp_size and act_sha == exp_sha:
                print(f"[OK]       {rel_path} ({act_size} bytes)")
                results.append(dict(rel_path=rel_path, status="OK", expected_sha=exp_sha, actual_sha=act_sha))
            else:
                print(f"[MISMATCH] {rel_path} (size act={act_size} exp={exp_size}, sha act={act_sha[:10]}... exp={exp_sha[:10]}...)")
                results.append(dict(rel_path=rel_path, status="MISMATCH", expected_sha=exp_sha, actual_sha=act_sha))
                all_ok = False

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"[Manifest] Verification summary: {ok_count}/{len(results)} files OK.")
    return all_ok, results


def main():
    parser = argparse.ArgumentParser(description="08_make_manifest: Generate or verify SHA-256 manifest of model weights.")
    parser.add_argument(
        "--dir",
        type=str,
        default=config.MODEL_DIR,
        help=f"Target model directory (default: {config.MODEL_DIR})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify files against existing MANIFEST.sha256 instead of generating it.",
    )
    args = parser.parse_args()

    if args.verify:
        all_ok, _ = verify_manifest(args.dir)
        sys.exit(0 if all_ok else 1)
    else:
        generate_manifest(args.dir)


if __name__ == "__main__":
    main()
