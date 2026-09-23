// Browser port of training/sli_kps.py. The word model was trained on exactly this layout,
// so any change here must be mirrored there (tests/unit/features.test.ts checks parity).

import faceIdx from '../../training/face_idx.json';

export const POSE_IDX = [11, 12, 13, 14, 15, 16] as const;
export const FACE_IDX: readonly number[] = faceIdx.face;
export const POSE_NUM = 6;
export const FACE_NUM = 136;
export const HAND_NUM = 21;
export const FEAT_NUM = POSE_NUM + FACE_NUM + 2 * HAND_NUM; // 184
export const FEAT_DIM = 4;
export const FRAME_SIZE = FEAT_NUM * FEAT_DIM; // 736
export const SEQ_LEN = 50;

// Reference quirks, kept on purpose: anchors index the subset arrays, scales the full landmark lists.
const POSE_ANCHOR = 0;
const POSE_SCALE: [number, number] = [11, 12];
const FACE_ANCHOR = 1;
const FACE_SCALE: [number, number] = [474, 469];
const HAND_SCALE: [number, number] = [2, 17];

const RH_OFFSET = (POSE_NUM + FACE_NUM) * FEAT_DIM;
const LH_OFFSET = (POSE_NUM + FACE_NUM + HAND_NUM) * FEAT_DIM;

export type Point = [number, number, number];
export type PosePoint = [number, number, number, number];

export interface RawFrame {
  pose: PosePoint[] | null;
  face: Point[] | null;
  hands: { label: string; lms: Point[] }[];
}

function dist(lms: ArrayLike<number>[], [a, b]: [number, number]): number {
  const p = lms[a];
  const q = lms[b];
  return Math.max(Math.hypot(p[0] - q[0], p[1] - q[1]), 1e-6);
}

function writePart(out: Float32Array, offset: number, pts: number[][], anchor: number, scale: number) {
  const ax = pts[anchor][0];
  const ay = pts[anchor][1];
  const az = pts[anchor][2];
  for (let i = 0; i < pts.length; i++) {
    const o = offset + i * FEAT_DIM;
    out[o] = (pts[i][0] - ax) / scale;
    out[o + 1] = (pts[i][1] - ay) / scale;
    out[o + 2] = (pts[i][2] - az) / scale;
    out[o + 3] = pts[i][3];
  }
}

/** One frame of raw landmarks -> 736 normalised features. Missing parts stay zero. */
export function frameFeatures(raw: RawFrame): Float32Array {
  const out = new Float32Array(FRAME_SIZE);
  if (raw.pose) {
    const pts = POSE_IDX.map((i) => raw.pose![i] as number[]);
    writePart(out, 0, pts, POSE_ANCHOR, dist(raw.pose, POSE_SCALE));
  }
  if (raw.face) {
    const pts = FACE_IDX.map((i) => [...raw.face![i], 1]);
    writePart(out, POSE_NUM * FEAT_DIM, pts, FACE_ANCHOR, dist(raw.face, FACE_SCALE));
  }
  for (const hand of raw.hands) {
    const pts = hand.lms.map((p) => [...p, 1]);
    writePart(out, hand.label === 'Left' ? LH_OFFSET : RH_OFFSET, pts, 0, dist(hand.lms, HAND_SCALE));
  }
  return out;
}

/** Evaluation-mode TSN sampling to SEQ_LEN frames (bin centres, linear interpolation). */
export function tsnSample(frames: Float32Array[], targetLen = SEQ_LEN): Float32Array {
  const n = frames.length;
  const out = new Float32Array(targetLen * FRAME_SIZE);
  if (n === 0) return out;
  for (let t = 0; t < targetLen; t++) {
    const start = (n * t) / targetLen;
    const end = (n * (t + 1)) / targetLen;
    let idx = start + (end - start) / 2;
    idx = Math.min(Math.max(idx, 0), n - 1 - 1e-6);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const a = idx - lo;
    const fl = frames[lo];
    const fh = frames[hi];
    const o = t * FRAME_SIZE;
    for (let k = 0; k < FRAME_SIZE; k++) out[o + k] = (1 - a) * fl[k] + a * fh[k];
  }
  return out;
}
