"""Build public/clips/NNNN.mp4 (+ .jpg poster) for every sign from KArSL test-split frames.

Usage: python make_clips.py KARSL_ALL_DIR [--signer 02] [--ffmpeg PATH]

One signer is used for every clip so the dictionary looks consistent; if a sign is missing
for that signer, the next signer is used.
"""

import argparse
import os
import subprocess
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--signers", default="02,01,03")
ap.add_argument("--ffmpeg", default="ffmpeg")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "clips"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

total = 0
for n in range(1, 503):
    sign = f"{n:04}"
    video_dir = None
    for signer in args.signers.split(","):
        d = os.path.join(args.root, signer, sign)
        if os.path.isdir(d) and os.listdir(d):
            # The longest take is usually the most complete one.
            takes = sorted(os.listdir(d), key=lambda v: -len(os.listdir(os.path.join(d, v))))
            video_dir = os.path.join(d, takes[0])
            break
    if not video_dir:
        print("missing", sign)
        continue
    frames = sorted(os.listdir(video_dir))
    mp4 = os.path.join(args.out, f"{sign}.mp4")
    jpg = os.path.join(args.out, f"{sign}.jpg")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for f in frames:
            lst.write(f"file '{os.path.join(video_dir, f)}'\nduration {1 / 30:.6f}\n")
        lst.write(f"file '{os.path.join(video_dir, frames[-1])}'\n")
    subprocess.run(
        [args.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst.name,
         "-vf", "scale=256:256,fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "27",
         "-profile:v", "main", "-movflags", "+faststart", "-an", mp4],
        check=True,
    )
    os.unlink(lst.name)
    subprocess.run(
        [args.ffmpeg, "-y", "-loglevel", "error", "-i", os.path.join(video_dir, frames[len(frames) // 2]),
         "-vf", "scale=256:256", "-q:v", "5", jpg],
        check=True,
    )
    total += os.path.getsize(mp4) + os.path.getsize(jpg)
    if n % 50 == 0:
        print(f"{n} clips, {total / 1e6:.1f} MB")
print(f"done, {total / 1e6:.1f} MB")
