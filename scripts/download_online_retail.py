"""Download the UCI "Online Retail II" dataset into data/track_b/raw/ (Track B).

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii — licensed CC BY 4.0, which
requires attribution. Cite it as UCI specifies:

    Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5CG6D.

The raw file is never committed (`data/track_b/*` is gitignored except its README). Only this
UCI repository page is used: mirrors on other sites carry other licence labels.

No credentials are needed. Standard library only, so it runs without the project environment.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = REPO_ROOT / "data" / "track_b" / "raw"
XLSX_NAME = "online_retail_II.xlsx"
CITATION = (
    "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5CG6D."
)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / XLSX_NAME
    if target.exists():
        print(f"{target} already present ({target.stat().st_size:,} bytes); not downloading again")
    else:
        zip_path = RAW_DIR / "online_retail_ii.zip"
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(XLSX_NAME, RAW_DIR)
        zip_path.unlink()

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target.name}: {target.stat().st_size:,} bytes, sha256 {digest}")
    print("Licence: CC BY 4.0 — attribution required. Cite as:")
    print(f"  {CITATION}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
