#!/usr/bin/env python3
# count_pt_per_video.py
# ────────────────────────────────────────────────
# matched.csv → video_pt_counts.csv
#   video_id , pt_count
import csv, argparse, collections, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--matched", default="data/matched.csv",
                help="output of match_pt_to_segments.py")
ap.add_argument("--out",     default="data/video_pt_counts.csv")
args = ap.parse_args()

pt_counts = collections.Counter()          # video_id → N

with open(args.matched, newline="") as f:
    for file, video_id, *_ in csv.reader(f):
        pt_counts[video_id] += 1

out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video_id", "pt_count"])
    for vid, n in sorted(pt_counts.items(), key=lambda x: (-x[1], x[0])):
        w.writerow([vid, n])

print(f"✅ counts for {len(pt_counts):,} videos written to {out}")
