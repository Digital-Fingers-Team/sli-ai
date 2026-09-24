"""Which MediaPipe speed-ups keep the word model's accuracy? Tracking (VIDEO) mode per part,
and refreshing pose/face only every other frame, against the IMAGE-mode reference.

Usage: python eval_video_mode.py FRAMES_DIR [--every K] [--fps 15]
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
ap.add_argument("frames")
ap.add_argument("--every", type=int, default=4, help="use every K-th sign")
ap.add_argument("--fps", type=int, default=15, help="simulated camera rate (source is 30)")
args = ap.parse_args()

session = ort.InferenceSession(os.path.join(sli_kps.HERE, "..", "public", "models", "karsl502.onnx"))


def classify(raws):
    return int(np.argmax(session.run(None, {"input": sli_kps.model_input([sli_kps.features(r) for r in raws])})[0][0]))


def opts(name):
    return BaseOptions(model_asset_path=os.path.join(sli_kps.LANDMARKERS, f"{name}_landmarker.task"))


def landmarkers(mode):
    return (
        vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=opts("pose"), running_mode=mode)),
        vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=opts("face"), running_mode=mode, num_faces=1)),
        vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=opts("hand"), running_mode=mode, num_hands=1)),
    )


def run(models, image, ts, video):
    pose, face, hands = models
    call = (lambda m: m.detect_for_video(image, ts)) if video else (lambda m: m.detect(image))
    p, f, h = call(pose), call(face), call(hands)
    return {
        "pose": [[l.x, l.y, l.z, l.visibility] for l in p.pose_landmarks[0]] if p.pose_landmarks else None,
        "face": [[l.x, l.y, l.z] for l in f.face_landmarks[0]] if f.face_landmarks else None,
        "hands": [{"label": c[0].category_name, "lms": [[l.x, l.y, l.z] for l in lms]} for c, lms in zip(h.handedness, h.hand_landmarks)],
    }


step = 30 // args.fps
image_models = landmarkers(vision.RunningMode.IMAGE)
variants = ["image", "image_body_half", "video_hands", "video_body", "video_all"]
ok = dict.fromkeys(variants, 0)
n = 0
for sign in sorted(os.listdir(args.frames))[:: args.every]:
    label = int(sign) - 1
    video = sorted(os.listdir(os.path.join(args.frames, sign)))[0]
    video_models = landmarkers(vision.RunningMode.VIDEO)
    vdir = os.path.join(args.frames, sign, video)
    seqs = {v: [] for v in variants}
    body_half = None
    for i, f in enumerate(sorted(os.listdir(vdir))[::step]):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(vdir, f)), cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        im = run(image_models, image, 0, False)
        vi = run(video_models, image, i * 1000 // args.fps, True)
        if i % 2 == 0:
            body_half = {"pose": im["pose"], "face": im["face"]}
        seqs["image"].append(im)
        seqs["image_body_half"].append({**body_half, "hands": im["hands"]})
        seqs["video_hands"].append({**im, "hands": vi["hands"]})
        seqs["video_body"].append({**vi, "hands": im["hands"]})
        seqs["video_all"].append(vi)
    for m in video_models:
        m.close()
    n += 1
    for v in variants:
        ok[v] += classify(seqs[v]) == label
    print(sign, n, {k: round(c / n, 3) for k, c in ok.items()}, flush=True)
print("FINAL", n, {k: round(c / n, 3) for k, c in ok.items()})
