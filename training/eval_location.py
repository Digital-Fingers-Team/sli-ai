"""How much does the word model rely on where the hands are relative to the body?

The model sees location through the pose points (shoulders, elbows, wrists, relative to the
left shoulder and scaled by shoulder width); hand points are relative to their own wrist and
face points to the face. This ablates that location and measures the accuracy lost.

Usage: python eval_location.py [--every K]   (uses eval_pretrained.py's cache)
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort

import sli_kps

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=3)
args = ap.parse_args()

session = ort.InferenceSession(os.path.join(sli_kps.HERE, "..", "public", "models", "karsl502.onnx"))
ARMS = slice(2, 6)  # elbows and wrists inside the 6 pose rows


def classify(feats):
    x = sli_kps.tsn_sample(np.stack(feats)).reshape(1, sli_kps.SEQ_LEN, -1).astype(np.float32)
    return int(np.argmax(session.run(None, {"input": x})[0][0]))


cache = os.path.join(sli_kps.HERE, "cache")
files = sorted(f for f in os.listdir(cache) if f.startswith("h1_"))
files = [f for f in files if (int(f.split("_")[3]) - 1) % args.every == 0]
rng = np.random.default_rng(0)
means = []  # average arm position of each video, to borrow from another video
videos = []
for f in files:
    feats = [sli_kps.features(r) for r in json.load(open(os.path.join(cache, f)))]
    videos.append((int(f.split("_")[3]) - 1, feats))
    means.append(np.mean([ft[ARMS] for ft in feats], axis=0))

variants = {
    "as is": lambda feats, k: feats,
    # the arms stay where they are on average, but do not move
    "arms frozen at their mean": lambda feats, k: [np.concatenate([ft[:2], means[k], ft[6:]]) for ft in feats],
    # the arms move exactly as before, but around another video's average location
    "arms moved to another sign's place": lambda feats, k: [
        np.concatenate([ft[:2], ft[ARMS] - means[k] + means[(k + 1 + rng.integers(len(means) - 1)) % len(means)], ft[6:]]) for ft in feats
    ],
    # hand shape and face only
    "no arms at all": lambda feats, k: [np.concatenate([ft[:2], np.zeros_like(ft[ARMS]), ft[6:]]) for ft in feats],
}
ok = dict.fromkeys(variants, 0)
for k, (label, feats) in enumerate(videos):
    for name, fn in variants.items():
        ok[name] += classify(fn(feats, k)) == label
for name, n in ok.items():
    print(f"{name}: {n / len(videos):.1%}")
print("videos", len(videos))
