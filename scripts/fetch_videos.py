"""Download walkthrough videos listed in video_manifest.csv, then hand them
to build_dataset.py — so you can source training frames from YouTube tours of
Indian/UAE malls, hospitals and colleges instead of recording everything
yourself.

Run this ON YOUR LAPTOP (needs internet + yt-dlp):

  pip install yt-dlp
  python scripts/fetch_videos.py                       # downloads all rows
  python scripts/fetch_videos.py --place mall --country uae
  python scripts/fetch_videos.py --cc-only             # Creative-Commons only

Then extract + pre-label frames exactly as before:

  python scripts/build_dataset.py --videos videos/*.mp4 --out datasets/av_raw --fps 1

LICENSING NOTE (put this in the paper's Data section):
  * Frames extracted from YouTube videos are fine for ACADEMIC model training
    in most jurisdictions, but you may NOT redistribute the raw videos/frames.
    Publish only the trained weights + the manifest of URLs (standard practice:
    Kinetics, AudioSet, YouTube-8M all ship URL lists, not pixels).
  * Prefer Creative-Commons videos where possible: --cc-only keeps only videos
    whose YouTube metadata says "Creative Commons Attribution" — those frames
    CAN be redistributed with attribution.
  * Your own recorded videos remain the gold standard: same camera height,
    same blur/lighting as deployment. Use YouTube data to ADD variety, not to
    replace the ~1 hour of self-recorded footage.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

MANIFEST = Path(__file__).parent / "video_manifest.csv"


def load_rows(place: str | None, country: str | None):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if place:
        rows = [r for r in rows if r["place_type"] == place]
    if country:
        rows = [r for r in rows if r["country"] == country]
    return rows


def fetch(rows, out: Path, cc_only: bool, max_height: int):
    out.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable, "-m", "yt_dlp",
        # 720p is plenty: build_dataset.py trains at imgsz<=832 anyway,
        # and 4K downloads are 10x slower for zero mAP gain.
        "-f", f"bestvideo[height<={max_height}][ext=mp4]/best[height<={max_height}]",
        "--no-playlist",
        "--write-info-json",          # keeps title/license/channel for the paper
        "-o", str(out / "%(id)s_%(title).60s.%(ext)s"),
    ]
    if cc_only:
        base += ["--match-filters", "license*=Creative Commons"]

    ok = fail = 0
    for r in rows:
        print(f"\n=== {r['country']}/{r['place_type']}: {r['note']}")
        res = subprocess.run(base + [r["url"]])
        ok += res.returncode == 0
        fail += res.returncode != 0
    print(f"\nDone: {ok} downloaded, {fail} failed/skipped -> {out}")
    print("Next: python scripts/build_dataset.py --videos "
          f"{out}/*.mp4 --out datasets/av_raw --fps 1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("videos"))
    ap.add_argument("--place", choices=["mall", "hospital", "college", "street"])
    ap.add_argument("--country", choices=["india", "uae"])
    ap.add_argument("--cc-only", action="store_true",
                    help="keep only Creative-Commons-licensed videos")
    ap.add_argument("--max-height", type=int, default=720)
    a = ap.parse_args()
    rows = load_rows(a.place, a.country)
    if not rows:
        sys.exit("No manifest rows match the filter.")
    fetch(rows, a.out, a.cc_only, a.max_height)
