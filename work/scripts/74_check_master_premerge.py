# FIELD: General
# MODEL: None
# STEP: utility
# USED BY: work/docs/CODE_MAP.md

import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None, check=True):
    print(f"--> {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res

def main():
    root = Path("C:/fetal-github-clone")
    work = root / "work"
    
    # 1. Checkout part3-experiments and pull latest
    print("Checking out part3-experiments...")
    run("git checkout part3-experiments", cwd=root)
    run("git pull origin part3-experiments", cwd=root)
    
    # 2. Check deleted files compared to master
    # Query git diff --name-status master part3-experiments
    res = run("git diff --name-status master part3-experiments", cwd=root)
    deleted_files = []
    for line in res.stdout.strip().splitlines():
        if line.startswith("D\t"):
            deleted_files.append(line.split("\t", 1)[1].strip())
            
    print(f"Total deleted files compared to master: {len(deleted_files)}")
    for f in deleted_files:
        print(f"  D: {f}")
        
    monitored_prefixes = (
        "docs/results/", "notebooks/", "scripts/", "src/", "tests/", "configs/",
        "work/docs/results/", "work/notebooks/", "work/scripts/", "work/src/", "work/tests/", "work/configs/"
    )
    
    allowed_deleted_exact = {
        "work/README.md",
        "src/stream3_clinical/train_clinical_model.py",
        "src/stream3_clinical/support.txt"
    }
    
    unexpected_deletions = []
    for f in deleted_files:
        f_norm = f.replace("\\", "/")
        if any(f_norm.startswith(p) for p in monitored_prefixes):
            if f_norm not in allowed_deleted_exact and not f_norm.startswith("src/stream3_clinical/"):
                unexpected_deletions.append(f_norm)
                
    if unexpected_deletions:
        raise AssertionError(f"Unexpected deletions compared to master under monitored prefixes: {unexpected_deletions}")
    print("[PASS] No unexpected deleted files compared to master under monitored prefixes.")

    # 3. Verify all paths in PROOF_LOCATIONS.md exist on disk
    proof_path = work / "docs" / "report" / "PROOF_LOCATIONS.md"
    assert proof_path.exists(), f"Missing {proof_path}"
    
    missing_paths = []
    total_checked = 0
    with open(proof_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("|") and not line.startswith("| ID") and not line.startswith("|:--"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 6:
                    for idx in (3, 4):
                        rel_path = parts[idx].strip("`").replace("\\", "/")
                        if rel_path and rel_path not in ("N/A", "---", "-", "none", "None"):
                            total_checked += 1
                            file_on_disk = work / rel_path
                            if not file_on_disk.exists():
                                file_on_disk2 = root / rel_path
                                if not file_on_disk2.exists():
                                    missing_paths.append(rel_path)
                                
    print(f"Checked {total_checked} proof paths from PROOF_LOCATIONS.md. Missing: {len(missing_paths)}")
    if missing_paths:
        raise AssertionError(f"Missing proof files on disk: {missing_paths}")
    print(f"[PASS] All {total_checked} proof paths exist on disk.")

    # 4. Run pytest tests/ and scripts 56, 57, 65
    print("Running pytest tests/...")
    res_pytest = run("pytest tests/", cwd=work)
    print(res_pytest.stdout.splitlines()[-1])
    
    print("Running script 56...")
    res_56 = run("python scripts/56_build_report_numbers.py", cwd=work)
    print(res_56.stdout.strip())
    
    print("Running script 57...")
    res_57 = run("python scripts/57_verify_report_numbers.py", cwd=work)
    print(res_57.stdout.strip())
    
    print("Running script 65...")
    res_65 = run("python scripts/65_build_proof_map.py", cwd=work)
    print(res_65.stdout.strip())
    
    print("\nALL PRE-MERGE CHECKS FOR MASTER PASSED!")

if __name__ == "__main__":
    main()
