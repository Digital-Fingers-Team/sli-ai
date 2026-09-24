"""Should the app track both hands? The word model was trained with one hand per frame
(num_hands=1), but its input has a right- and a left-hand slot. Detects two hands on held-out
videos (pose and face reused from eval_pretrained.py's cache) and compares ways of feeding them.

Usage: python eval_two_hands.py FRAMES_ROOT [--every K] [--signers 01,02,03]
"""

import argparse
import json
import os

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks.python import BaseOptions, vision

import sli_kps

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--signers", default="01,02,03")
args = ap.parse_args()

session = ort.InferenceSession(os.path.join(sli_kps.HERE, "..", "public", "models", "karsl502.onnx"))
hands2 = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(sli_kps.LANDMARKERS, "hand_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_hands=2))


def classify(raws):
    return int(np.argmax(session.run(None, {"input": sli_kps.model_input([sli_kps.features(r) for r in raws])})[0][0]))


def pick(hands, how, prev):
    """One hand out of up to two, the way the app could choose it."""
    if len(hands) < 2:
        return hands
    if how == "first":
        return hands[:1]
    if how == "higher":  # the raised hand: smallest wrist y
        return [min(hands, key=lambda h: h["lms"][0][1])]
    if how == "moving":  # the hand whose wrist moved most since the previous frame
        def moved(h):
            p = next((q for q in prev if q["label"] == h["label"]), None)
            return np.hypot(h["lms"][0][0] - p["lms"][0][0], h["lms"][0][1] - p["lms"][0][1]) if p else 0
        return [max(hands, key=moved)]
    return hands


variants = ["one_hand", "both", "first", "higher", "moving"]
ok = dict.fromkeys(variants, 0)
two_seen = 0
n = 0
for signer in args.signers.split(","):
    for sign in sorted(os.listdir(os.path.join(args.root, signer)))[:: args.every]:
        video = sorted(os.listdir(os.path.join(args.root, signer, sign)))[0]
        cache = os.path.join(sli_kps.HERE, "cache", f"h1_{video}.json")
        if not os.path.exists(cache):
            continue
        one = json.load(open(cache))
        vdir = os.path.join(args.root, signer, sign, video)
        files = sorted(os.listdir(vdir))
        seqs = {v: [] for v in variants}
        prev = []
        any_two = False
        for raw, f in zip(one, files):
            rgb = cv2.cvtColor(cv2.imread(os.path.join(vdir, f)), cv2.COLOR_BGR2RGB)
            r = hands2.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            hands = [{"label": c[0].category_name, "lms": [[l.x, l.y, l.z] for l in lms]} for c, lms in zip(r.handedness, r.hand_landmarks)]
            any_two |= len(hands) == 2
            seqs["one_hand"].append(raw)
            for v in variants[1:]:
                seqs[v].append({**raw, "hands": pick(hands, v, prev)})
            prev = hands
        label = int(sign) - 1
        n += 1
        two_seen += any_two
        for v in variants:
            ok[v] += classify(seqs[v]) == label
        print(signer, sign, n, {k: round(c / n, 3) for k, c in ok.items()}, "videos with 2 hands", two_seen, flush=True)
print("FINAL", n, {k: round(c / n, 3) for k, c in ok.items()}, "videos with 2 hands", two_seen)
