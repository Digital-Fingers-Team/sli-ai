"""Keypoint extraction matching the pretrained KArSL ST-Transformer.

Produces a (184, 4) array per frame: 6 pose + 136 face + 21 right hand + 21 left hand,
each (x, y, z, v). The normalisation copies the reference implementation exactly,
including its index quirks, because the model was trained on that layout. The browser
port lives in src/recognition/features.ts and must stay in sync with this file.
"""

import json
import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARKERS = os.environ.get("SLI_LANDMARKERS", os.path.join(HERE, "..", "public", "mediapipe"))

POSE_IDX = (11, 12, 13, 14, 15, 16)  # shoulders, elbows, wrists
FACE_IDX = tuple(json.load(open(os.path.join(HERE, "face_idx.json")))["face"])  # 136 contour+iris points
POSE_NUM, FACE_NUM, HAND_NUM = 6, 136, 21
FEAT_NUM = POSE_NUM + FACE_NUM + 2 * HAND_NUM  # 184
SEQ_LEN = 50

# Reference quirks: the anchor index is applied to the *subset* arrays.
POSE_ANCHOR = 0  # mp NOSE (0) used on the subset -> left shoulder
POSE_SCALE = (11, 12)  # shoulder distance, on full landmarks
FACE_ANCHOR = 1  # mp face nose (1) used on the subset
FACE_SCALE = (474, 469)  # iris points, on full landmarks
HAND_SCALE = (2, 17)  # thumb MCP to pinky MCP

SL_POSE = slice(0, POSE_NUM)
SL_FACE = slice(POSE_NUM, POSE_NUM + FACE_NUM)
SL_RH = slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM)
SL_LH = slice(POSE_NUM + FACE_NUM + HAND_NUM, FEAT_NUM)


def _dist(lms, pair):
    a, b = lms[pair[0]], lms[pair[1]]
    return max(float(np.hypot(a.x - b.x, a.y - b.y)), 1e-6)


class Extractor:
    def __init__(self, num_hands=1):
        def opts(name):
            return BaseOptions(model_asset_path=os.path.join(LANDMARKERS, f"{name}_landmarker.task"))

        mode = vision.RunningMode.IMAGE
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=opts("pose"), running_mode=mode))
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=opts("face"), running_mode=mode, num_faces=1))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=opts("hand"), running_mode=mode, num_hands=num_hands))

    def raw(self, frame_rgb):
        """Return raw landmarks as plain lists so they can be saved and replayed."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        pose = self.pose.detect(image)
        face = self.face.detect(image)
        hands = self.hands.detect(image)
        return {
            "pose": [[l.x, l.y, l.z, l.visibility] for l in pose.pose_landmarks[0]] if pose.pose_landmarks else None,
            "face": [[l.x, l.y, l.z] for l in face.face_landmarks[0]] if face.face_landmarks else None,
            "hands": [
                {"label": h[0].category_name, "lms": [[l.x, l.y, l.z] for l in lms]}
                for h, lms in zip(hands.handedness, hands.hand_landmarks)
            ],
        }

    def close(self):
        self.pose.close()
        self.face.close()
        self.hands.close()


class _P:
    __slots__ = ("x", "y")

    def __init__(self, p):
        self.x, self.y = p[0], p[1]


def features(raw):
    """Raw landmarks for one frame -> normalised (184, 4) features."""
    out = np.zeros((FEAT_NUM, 4), dtype=np.float64)
    if raw["pose"]:
        lms = raw["pose"]
        pose = np.array([lms[i] for i in POSE_IDX], dtype=np.float64)
        pose[:, :3] -= pose[POSE_ANCHOR, :3]
        pose[:, :3] /= _dist([_P(p) for p in lms], POSE_SCALE)
        out[SL_POSE] = pose
    if raw["face"]:
        lms = raw["face"]
        face = np.array([[*lms[i], 1.0] for i in FACE_IDX], dtype=np.float64)
        face[:, :3] -= face[FACE_ANCHOR, :3]
        face[:, :3] /= _dist([_P(p) for p in lms], FACE_SCALE)
        out[SL_FACE] = face
    for hand in raw["hands"]:
        lms = hand["lms"]
        h = np.array([[*p, 1.0] for p in lms], dtype=np.float64)
        h[:, :3] -= h[0, :3]
        h[:, :3] /= _dist([_P(p) for p in lms], HAND_SCALE)
        out[SL_LH if hand["label"] == "Left" else SL_RH] = h
    return out


def tsn_sample(seq, target_len=SEQ_LEN):
    """Evaluation-mode TSN sampling: bin centres with linear interpolation."""
    n = seq.shape[0]
    ticks = np.linspace(0, n, target_len + 1)
    idx = ticks[:-1] + (ticks[1:] - ticks[:-1]) / 2.0
    idx = np.clip(idx, 0, n - 1 - 1e-6)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    a = (idx - lo)[:, None, None]
    return (1 - a) * seq[lo] + a * seq[hi]


def model_input(frame_features):
    seq = tsn_sample(np.stack(frame_features))
    return seq.reshape(1, SEQ_LEN, FEAT_NUM * 4).astype(np.float32)
