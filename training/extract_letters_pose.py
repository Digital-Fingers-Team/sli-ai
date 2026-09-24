"""Body reference points for every KArSL letter video, frame-aligned with extract_letters.py.

Usage: python extract_letters_pose.py FRAMES_ROOT LETTERS_JSON OUT_JSON
Adds to each video "pose": per frame [nose, left shoulder, right shoulder] as [x, y] or null,
so train_letters.py --location can place the hand relative to the head and shoulders.
"""

import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import sli_kps

root, letters_path, out = sys.argv[1:4]
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(sli_kps.LANDMARKERS, "pose_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE))
POINTS = (0, 11, 12)

videos = json.load(open(letters_path))
for k, v in enumerate(videos):
    vdir = os.path.join(root, v["signer"], f"{v['sign'] + 1:04d}", v["video"])
    frames = []
    for f in sorted(os.listdir(vdir)):
        bgr = cv2.imread(os.path.join(vdir, f))
        if bgr is None:
            continue
        r = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
        frames.append([[round(r.pose_landmarks[0][i].x, 5), round(r.pose_landmarks[0][i].y, 5)] for i in POINTS] if r.pose_landmarks else None)
    assert len(frames) == len(v["frames"]), v["video"]
    v["pose"] = frames
    if k % 50 == 0:
        print(k, len(videos), flush=True)
json.dump(videos, open(out, "w"))
print("DONE", len(videos))
