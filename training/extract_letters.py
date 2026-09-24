"""Hand landmarks for every KArSL letter video (signs 32-70), for the live letter model.

Usage: python extract_letters.py FRAMES_ROOT OUT_JSON
FRAMES_ROOT holds <signer>/<sign>/<video>/<frame>.jpg for all three signers.
Output: [{"signer": "01", "sign": 31, "video": ..., "frames": [[[x,y,z]*21] | null, ...]}]
Only hands are extracted: letters are hand shapes, and hands alone are fast to run.
"""

import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import sli_kps

root, out = sys.argv[1], sys.argv[2]
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(sli_kps.LANDMARKERS, "hand_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_hands=1))

videos = []
for signer in sorted(os.listdir(root)):
    for sign in range(32, 71):
        sdir = os.path.join(root, signer, f"{sign:04d}")
        for video in sorted(os.listdir(sdir)):
            frames = []
            vdir = os.path.join(sdir, video)
            for f in sorted(os.listdir(vdir)):
                bgr = cv2.imread(os.path.join(vdir, f))
                if bgr is None:
                    continue
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                r = hands.detect(image)
                frames.append({"label": r.handedness[0][0].category_name,
                               "lms": [[round(l.x, 5), round(l.y, 5), round(l.z, 5)] for l in r.hand_landmarks[0]]}
                              if r.hand_landmarks else None)
            videos.append({"signer": signer, "sign": sign - 1, "video": video, "frames": frames})
        print(signer, sign, len(videos), flush=True)
json.dump(videos, open(out, "w"))
print("DONE", len(videos))
