import os
import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add tests/reference to sys.path
ref_dir = Path(__file__).resolve().parent / "reference"
if str(ref_dir) not in sys.path:
    sys.path.insert(0, str(ref_dir))

# Take DATA_BASE and OUT_DIR dynamically from fetal.config (no hardcoded path)
from fetal import config
os.environ["DATA_BASE"] = config.DATA_DIR
os.environ["OUT_DIR"] = config.AC_V3_DIR
