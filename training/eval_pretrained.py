"""Measure the pretrained word model on held-out KArSL test videos.

Usage: python eval_pretrained.py FRAMES_DIR [--per-sign N] [--hands 1|2] [--out report.json]

FRAMES_DIR holds <sign>/<video>/<frame>.jpg as unpacked from the KArSL test archives.
Raw landmarks are cached next to the report so the browser port can be checked against them.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

import sli_kps

ap = argparse.ArgumentParser()
ap.add_argument("frames")
ap.add_argument("--per-sign", type=int, default=2)
ap.add_argument("--hands", type=int, default=1)
ap.add_argument("--model", default=os.path.join(sli_kps.HERE, "..", "public", "models", "karsl502.onnx"))
ap.add_argument("--cache", default=os.path.join(sli_kps.HERE, "cache"))
ap.add_argument("--out", default="report.json")
args = ap.parse_args()

session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
extractor = None
os.makedirs(args.cache, exist_ok=True)

results = []
t0 = time.time()
for sign in sorted(os.listdir(args.frames)):
    label = int(sign) - 1
    videos = sorted(os.listdir(os.path.join(args.frames, sign)))[: args.per_sign]
    for video in videos:
        cache = os.path.join(args.cache, f"h{args.hands}_{video}.json")
        if os.path.exists(cache):
            raws = json.load(open(cache))
        else:
            extractor = extractor or sli_kps.Extractor(num_hands=args.hands)
            vdir = os.path.join(args.frames, sign, video)
            raws = []
            for f in sorted(os.listdir(vdir)):
                bgr = cv2.imread(os.path.join(vdir, f))
                if bgr is not None:
                    raws.append(extractor.raw(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            json.dump(raws, open(cache, "w"))
        x = sli_kps.model_input([sli_kps.features(r) for r in raws])
        logits = session.run(None, {"input": x})[0][0]
        top5 = np.argsort(-logits)[:5].tolist()
        results.append({"sign": label, "video": video, "top5": top5})
    done = len(results)
    top1 = np.mean([r["top5"][0] == r["sign"] for r in results])
    print(f"sign {sign}: {done} videos, top1 so far {top1:.3f}, {time.time() - t0:.0f}s", file=sys.stderr)

top1 = float(np.mean([r["top5"][0] == r["sign"] for r in results]))
top5 = float(np.mean([r["sign"] in r["top5"] for r in results]))
json.dump({"videos": len(results), "top1": top1, "top5": top5, "hands": args.hands, "results": results},
          open(args.out, "w"), indent=1)
print(f"videos={len(results)} top1={top1:.3f} top5={top5:.3f}")
