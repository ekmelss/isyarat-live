"""Freeze the feature table. Owner only, committed through a referee-change PR.

    uv run python -m eval.freeze [data/features/static.csv]
"""

import sys
from pathlib import Path

from eval.data import FREEZE_FILE, digest_of

if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/features/static.csv")
    digest = digest_of(path)
    FREEZE_FILE.write_text(f"{digest}  {path.as_posix()}\n", encoding="utf-8", newline="\n")
    print(f"froze {path} at sha256 {digest}")
