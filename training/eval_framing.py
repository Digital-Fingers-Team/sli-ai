"""What happens when the signer is framed tighter than in KArSL (a phone held upright, or too
close), so a hand leaves the picture while the elbow is still in view?

Crops KArSL test frames to simulate other framings, runs the same landmarkers (two hands) and
the word model, and reports accuracy plus how the framing looked (shoulder width share of the
image, frames with a wrist outside the picture) so the app can warn at the right point.

Usage: python eval_framing.py FRAMES_ROOT [--every K] [--signs 288,...]
"""

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks.python import BaseOptions, vision

import sli_kps

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--every", type=int, default=6)
ap.add_argument("--signers", default="01,02,03")
ap.add_argument("--signs", default="", help="extra sign folders (1-based) always included, e.g. 289 for أهلا وسهلا")
args = ap.parse_args()

session = ort.InferenceSession(os.path.join(sli_kps.HERE, "..", "public", "models", "karsl502.onnx"))
extractor = sli_kps.Extractor(num_hands=2)

# (width, height) of the kept region as a share of the KArSL frame, top-anchored like a person
# whose head stays in view. KArSL frames are square; a phone held upright is 3:4.
FRAMINGS = {
    "as recorded": (1.0, 1.0),
    "upright phone": (0.75, 1.0),
    "upright, closer": (0.6, 0.8),
    "upright, very close": (0.5, 0.67),
}


def crop(img, w, h):
    H, W = img.shape[:2]
    cw, ch = int(W * w), int(H * h)
    x0 = (W - cw) // 2
    y0 = int((H - ch) * 0.15)
    return img[y0: y0 + ch, x0: x0 + cw]


picks = []
for signer in args.signers.split(","):
    signs = sorted(os.listdir(os.path.join(args.root, signer)))
    chosen = set(signs[:: args.every]) | {f"{int(s):04d}" for s in args.signs.split(",") if s}
    for sign in sorted(chosen):
        videos = sorted(os.listdir(os.path.join(args.root, signer, sign)))
        picks.append((signer, sign, videos[0]))

stats = {k: {"ok": 0, "n": 0, "shoulders": [], "wrist_out": 0, "frames": 0} for k in FRAMINGS}
focus = {k: [] for k in FRAMINGS}
for signer, sign, video in picks:
    label = int(sign) - 1
    vdir = os.path.join(args.root, signer, sign, video)
    frames = [cv2.cvtColor(cv2.imread(os.path.join(vdir, f)), cv2.COLOR_BGR2RGB) for f in sorted(os.listdir(vdir))]
    for name, (w, h) in FRAMINGS.items():
        raws = [extractor.raw(np.ascontiguousarray(crop(fr, w, h))) for fr in frames]
        pred = int(np.argmax(session.run(None, {"input": sli_kps.model_input([sli_kps.features(r) for r in raws])})[0][0]))
        s = stats[name]
        s["n"] += 1
        s["ok"] += pred == label
        for r in raws:
            if r["pose"]:
                p = r["pose"]
                s["shoulders"].append(abs(p[11][0] - p[12][0]))
                s["frames"] += 1
                s["wrist_out"] += any(not (0 <= p[k][0] <= 1 and 0 <= p[k][1] <= 1) for k in (15, 16))
        if args.signs and str(int(sign)) in args.signs.split(","):
            focus[name].append(pred == label)
    print(signer, sign, {k: f"{v['ok']}/{v['n']}" for k, v in stats.items()}, flush=True)

print("FINAL")
for name, s in stats.items():
    sh = np.median(s["shoulders"]) if s["shoulders"] else float("nan")
    extra = f", focus signs {sum(focus[name])}/{len(focus[name])}" if focus[name] else ""
    print(f"{name}: top-1 {s['ok'] / s['n']:.1%} ({s['n']} videos), shoulders {sh:.2f} of width, "
          f"frames with a wrist outside {s['wrist_out'] / max(s['frames'], 1):.0%}{extra}")
