"""Train the live letter model: one hand, one frame -> which of KArSL's 39 letter signs.

Usage: python train_letters.py LETTERS_JSON [--out ../public/models/letters.json]
LETTERS_JSON comes from extract_letters.py. Accuracy is reported leave-one-signer-out (train
on two signers, test on the third), which is the honest estimate for a new person; the shipped
model is then trained on all three. letter_features() must match handFeatures() in
src/recognition/letters.ts.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("letters")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "models", "letters.json"))
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--span", default="0.3,0.8", help="part of each video's hand frames that shows the letter")
ap.add_argument("--mirror", choices=["none", "label"], default="none",
                help="label: mirror one handedness label onto the other (MediaPipe's label flips with hand orientation)")
ap.add_argument("--extra", action="store_true", help="add fingertip-to-fingertip and fingertip-to-wrist distances")
ap.add_argument("--location", action="store_true",
                help="add where the wrist is relative to the shoulders and the nose (needs extract_letters_pose.py output)")
ap.add_argument("--quick", action="store_true", help="evaluation only, do not write the model")
ap.add_argument("--fixture-only", action="store_true", help="only rewrite the parity fixture from the model in --out")
ap.add_argument("--exclude", help="train without this signer and write only the model (for tests/replay with SLI_SIGNER)")
ap.add_argument("--fixture", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "letters-parity.json"))
args = ap.parse_args()

videos = json.load(open(args.letters))
classes = sorted({v["sign"] for v in videos})
cls_index = {c: i for i, c in enumerate(classes)}
labels = Counter(f["label"] for v in videos for f in v["frames"] if f)
MIRROR = min(labels, key=labels.get) if args.mirror == "label" else ""
print("handedness", dict(labels), "mirror", MIRROR or "none")
TIPS = [0, 4, 8, 12, 16, 20]
PAIRS = [(a, b) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]
lo, hi = map(float, args.span.split(","))


def location(lms, label, pose):
    """Wrist relative to the shoulder midpoint and to the nose, in shoulder widths."""
    if not pose:
        return [0.0, 0.0, 0.0, 0.0]
    (nx, ny), (lx, ly), (rx, ry) = pose
    width = max(float(np.hypot(lx - rx, ly - ry)), 1e-6)
    wx, wy = lms[0][0], lms[0][1]
    flip = -1.0 if label == MIRROR else 1.0
    return [flip * (wx - (lx + rx) / 2) / width, (wy - (ly + ry) / 2) / width, flip * (wx - nx) / width, (wy - ny) / width]


def letter_features(lms, label, pose=None):
    p = np.asarray(lms, dtype=np.float64)
    p = p - p[0]
    scale = max(float(np.hypot(p[9, 0], p[9, 1])), 1e-6)
    p = p / scale
    if label == MIRROR:
        p[:, 0] = -p[:, 0]
    out = p.reshape(-1)
    if args.extra:
        out = np.concatenate([out, [np.linalg.norm(p[a] - p[b]) for a, b in PAIRS]])
    if args.location:
        out = np.concatenate([out, location(lms, label, pose)])
    return out


def samples(vids):
    X, y, vid = [], [], []
    for k, v in enumerate(vids):
        hand = [(i, f) for i, f in enumerate(v["frames"]) if f]
        n = len(hand)
        for i, f in hand[int(n * lo): max(int(n * hi), int(n * lo) + 1)]:
            X.append(letter_features(f["lms"], f["label"], v["pose"][i] if args.location else None))
            y.append(cls_index[v["sign"]])
            vid.append(k)
    return np.array(X, dtype=np.float32), np.array(y), np.array(vid)


def augment(X, rng):
    """Small in-plane rotations, per-axis stretch and jitter: other people hold their hands
    differently, and hold them a little higher or lower, nearer or further from the body."""
    parts = [augment_shape(X[:, :63], rng)]
    if args.extra:  # distances are recomputed from the augmented points
        P = parts[0].reshape(len(X), 21, 3)
        parts.append(np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in PAIRS], 1))
    if args.location:
        loc = X[:, -4:].copy()
        shift = rng.normal(0, 0.15, (len(X), 2))
        loc[:, :2] += shift
        loc[:, 2:] += shift
        parts.append(loc + rng.normal(0, 0.05, loc.shape))
    return np.concatenate(parts, 1).astype(np.float32)


def augment_shape(X, rng):
    P = X.reshape(len(X), 21, 3).copy()
    th = rng.uniform(-0.3, 0.3, len(X))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x, yy = P[:, :, 0].copy(), P[:, :, 1].copy()
    P[:, :, 0] = c * x - s * yy
    P[:, :, 1] = s * x + c * yy
    P *= rng.uniform(0.85, 1.15, (len(X), 1, 3))
    P += rng.normal(0, 0.03, P.shape)
    return P.reshape(len(X), -1).astype(np.float32)


def train(X, y, rng, epochs=args.epochs, hidden=args.hidden):
    sizes = [X.shape[1], hidden, hidden, len(classes)]
    W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
    B = [np.zeros(b, np.float32) for b in sizes[1:]]
    params = W + B
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step, lr, wd, batch = 0, 2e-3, 1e-4, 256
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        Xa = augment(X, rng)
        for i in range(0, len(X), batch):
            idx = order[i: i + batch]
            acts = [Xa[idx]]
            for k in range(len(W)):
                z = acts[-1] @ W[k] + B[k]
                acts.append(np.maximum(z, 0) if k < len(W) - 1 else z)
            logits = acts[-1]
            e = np.exp(logits - logits.max(1, keepdims=True))
            prob = e / e.sum(1, keepdims=True)
            g = prob
            g[np.arange(len(idx)), y[idx]] -= 1
            g /= len(idx)
            gW, gB = [None] * len(W), [None] * len(W)
            for k in reversed(range(len(W))):
                gW[k] = acts[k].T @ g + wd * W[k]
                gB[k] = g.sum(0)
                if k:
                    g = (g @ W[k].T) * (acts[k] > 0)
            step += 1
            rate = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            for j, (p, gp) in enumerate(zip(params, gW + gB)):
                m[j] = 0.9 * m[j] + 0.1 * gp
                v[j] = 0.999 * v[j] + 0.001 * gp * gp
                p -= rate * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return W, B


def predict(W, B, X):
    a = X
    for k in range(len(W)):
        a = a @ W[k] + B[k]
        if k < len(W) - 1:
            a = np.maximum(a, 0)
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def write_fixture(model):
    """Parity fixture for tests/unit/letters.test.ts: a few frames of each handedness label, with
    the features and probabilities computed here from the weights as saved."""
    Wr = [np.array(l["w"], np.float32).reshape(l["rows"], l["cols"]) for l in model["layers"]]
    Br = [np.array(l["b"], np.float32) for l in model["layers"]]
    picks = []
    for lab in sorted(labels):
        picks += [{**f, "pose": v["pose"][i] if args.location else None}
                  for v in videos[::97] for i, f in enumerate(v["frames"]) if f and f["label"] == lab][:3]
    feats = np.array([letter_features(f["lms"], f["label"], f["pose"]) for f in picks], np.float32)
    json.dump({"frames": picks, "features": feats.round(5).tolist(), "probs": predict(Wr, Br, feats).round(5).tolist()},
              open(args.fixture, "w"))
    print("wrote", args.fixture)


METRICS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "data", "metrics.json")


def write_metrics(report):
    """Held-out-signer accuracy for the About page, next to the word model's numbers."""
    metrics = json.load(open(METRICS)) if os.path.exists(METRICS) else {}
    metrics["letters"] = {
        "signers": len(report),
        "videos": round(float(np.mean([r["videos"] for r in report.values()])), 4),
        "frames": round(float(np.mean([r["frames"] for r in report.values()])), 4),
    }
    json.dump(metrics, open(METRICS, "w"), indent=1)


if args.fixture_only:
    model = json.load(open(args.out))
    write_fixture(model)
    write_metrics(model["heldOut"])
    raise SystemExit

rng = np.random.default_rng(0)
signers = sorted({v["signer"] for v in videos})
report = {}
if args.exclude:
    videos = [v for v in videos if v["signer"] != args.exclude]
    signers = []
for held in signers:
    Xtr, ytr, _ = samples([v for v in videos if v["signer"] != held])
    test_videos = [v for v in videos if v["signer"] == held]
    Xte, yte, vte = samples(test_videos)
    W, B = train(Xtr, ytr, rng)
    P = predict(W, B, Xte)
    frame_acc = float((P.argmax(1) == yte).mean())
    # Per video: the letter the averaged probabilities pick, like the app's smoothing does.
    vid_ok = [P[vte == k].mean(0).argmax() == cls_index[v["sign"]] for k, v in enumerate(test_videos) if (vte == k).any()]
    confusions = Counter((classes[a], classes[b]) for a, b in zip(yte, P.argmax(1)) if a != b).most_common(5)
    report[held] = {"frames": round(frame_acc, 3), "videos": round(float(np.mean(vid_ok)), 3), "confusions": confusions}
    print("held-out signer", held, report[held], flush=True)

if report:
    print("mean frame acc", np.mean([r["frames"] for r in report.values()]), "video acc", np.mean([r["videos"] for r in report.values()]))
if args.quick:
    raise SystemExit

X, y, _ = samples(videos)
W, B = train(X, y, rng)
model = {
    "classes": classes,
    "mirror": MIRROR,
    "extra": args.extra,
    "location": args.location,
    "layers": [
        {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "w": [round(float(x), 5) for x in w.reshape(-1)], "b": [round(float(x), 5) for x in b]}
        for w, b in zip(W, B)
    ],
    "heldOut": report,
}
json.dump(model, open(args.out, "w"), separators=(",", ":"))
print("wrote", args.out, os.path.getsize(args.out) // 1024, "KB")
if not args.exclude:
    write_fixture(model)
    write_metrics(report)
