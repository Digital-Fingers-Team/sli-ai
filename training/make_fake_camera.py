"""Make a .y4m "webcam" from KArSL test videos for end-to-end tests in Chromium
(--use-file-for-fake-video-capture). Each sign is framed by empty background so the hands
leave the picture between signs, like a person lowering their hands.

Usage: python make_fake_camera.py OUT.y4m KARSL_ALL_DIR SIGNER SIGN [SIGN ...]
       [--slow N]  repeat each frame N times (for slow software-rendered test browsers)
       [--crop W,H] keep only this share of each frame (top-anchored), like a signer too close
                    to the camera; for the framing-coach test
"""

import argparse
import os
import subprocess
import tempfile

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("root")
ap.add_argument("signer")
ap.add_argument("signs", nargs="+", type=int)
ap.add_argument("--slow", type=int, default=1)
ap.add_argument("--gap", type=float, default=1.5)
ap.add_argument("--crop", help="W,H share of the frame to keep, e.g. 0.5,0.67")
ap.add_argument("--ffmpeg", default="ffmpeg")
args = ap.parse_args()

fps = 30
tmp = tempfile.mkdtemp()
lines = []


def add(path, frames):
    lines.append(f"file '{path}'\nduration {frames / fps:.6f}\n")


first = None
for n in args.signs:
    d = os.path.join(args.root, args.signer, f"{n:04}")
    take = sorted(os.listdir(d))[0]
    frames = sorted(os.listdir(os.path.join(d, take)))
    if first is None:
        first = os.path.join(d, take, frames[0])
        bg = Image.open(first).convert("RGB").getpixel((4, 4))
        blank = os.path.join(tmp, "blank.jpg")  # concat needs one codec: KArSL frames are JPEG
        Image.new("RGB", (256, 256), bg).save(blank, quality=95)
    add(blank, int(args.gap * fps * args.slow))
    for f in frames:
        add(os.path.join(d, take, f), args.slow)
add(blank, int(args.gap * fps * args.slow))
lines.append(f"file '{blank}'\n")

if args.crop:
    w, h = map(float, args.crop.split(","))
    # No padding: a phone camera fills the whole picture, so a cut-off hand meets the real edge.
    view = f"crop=iw*{w}:ih*{h}:iw*{(1 - w) / 2}:ih*{(1 - h) * 0.15},scale=-2:480,"
else:
    view = "scale=480:480,pad=640:480:80:0:color=0x3f9a8f,"

lst = os.path.join(tmp, "list.txt")
open(lst, "w").write("".join(lines))
subprocess.run(
    [args.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
     "-vf", view + "fps=30,format=yuv420p", args.out],
    check=True,
)
print("wrote", args.out)
