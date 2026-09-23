// Draws the tracked hand skeleton and arms over the (mirrored) camera view.

import type { RawFrame } from '../recognition/features';

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const ARM_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

export function drawOverlay(canvas: HTMLCanvasElement, raw: RawFrame, signing: boolean) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const hgt = canvas.height;
  ctx.clearRect(0, 0, w, hgt);
  // The video is shown mirrored, landmarks are not.
  const X = (x: number) => (1 - x) * w;
  const Y = (y: number) => y * hgt;
  const styles = getComputedStyle(canvas);

  if (raw.pose) {
    ctx.strokeStyle = styles.getPropertyValue('--overlay-arm').trim() || '#fff8';
    ctx.lineWidth = 3 * dpr;
    for (const [a, b] of ARM_EDGES) {
      const p = raw.pose[a];
      const q = raw.pose[b];
      if (p[3] < 0.5 || q[3] < 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(X(p[0]), Y(p[1]));
      ctx.lineTo(X(q[0]), Y(q[1]));
      ctx.stroke();
    }
  }
  const color = styles.getPropertyValue(signing ? '--live' : '--overlay-hand').trim() || '#F2B632';
  for (const hand of raw.hands) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3 * dpr;
    for (const [a, b] of HAND_EDGES) {
      ctx.beginPath();
      ctx.moveTo(X(hand.lms[a][0]), Y(hand.lms[a][1]));
      ctx.lineTo(X(hand.lms[b][0]), Y(hand.lms[b][1]));
      ctx.stroke();
    }
    for (const p of hand.lms) {
      ctx.beginPath();
      ctx.arc(X(p[0]), Y(p[1]), 3.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
