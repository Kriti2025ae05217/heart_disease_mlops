"""Download the UCI Heart Disease (Cleveland) dataset and write a cleaned CSV.

Usage:
    python scripts/download_data.py            # download + clean
    python scripts/download_data.py --force    # re-download even if present

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""
import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heart.data import RAW_PATH, CLEAN_PATH, load_raw, clean  # noqa: E402

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download raw file")
    args = parser.parse_args()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not RAW_PATH.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, RAW_PATH)
    else:
        print(f"Raw file already present: {RAW_PATH}")

    df = clean(load_raw(RAW_PATH))
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Wrote {CLEAN_PATH} shape={df.shape}")
    print(df["target"].value_counts().to_string())


if __name__ == "__main__":
    main()
