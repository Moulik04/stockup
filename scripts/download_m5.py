"""Download the M5 Forecasting Accuracy dataset via the Kaggle CLI into data/track_a/raw/.

Requires a free Kaggle account, having accepted the competition rules on kaggle.com, and API
credentials (~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY in .env).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPETITION = "m5-forecasting-accuracy"
RAW_DIR = REPO_ROOT / "data" / "track_a" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(RAW_DIR),
    ]
    subprocess.run(cmd, check=True)

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(RAW_DIR)], check=True)
        zip_path.unlink()

    print(f"M5 raw files ready in {RAW_DIR}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
