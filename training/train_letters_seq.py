"""Letter model over a short window of frames instead of one frame.

Several letter pairs share a hand shape and differ by a small movement (ي/ى/ئ, ج/ح, ت/ة); a
one-frame model cannot see it. This one reads the last WINDOW frames (at ~15 fps): the same
per-frame hand features as train_letters.py plus the wrist's path over the window.

Usage: python train_letters_seq.py LETTERS_JSON [--window 8] [--quick] [--out ...]
Accuracy is leave-one-signer-out per video (probabilities averaged over the windows that end
in the letter part of the video). --window 1 is the one-frame model under the same protocol.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("letters")
ap.add_argument("--window", type=int, default=8)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--span", default="0.25,1.0", help="part of the hand frames where a window may end")
ap.add_argument("--shape-last", action="store_true",
                help="hand shape of the last frame only, plus the wrist path over the window")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--calibrate", type=int, default=0,
                help="simulate a user recording each letter N times: use N of the held-out signer's videos per letter to adapt, test on the rest")
ap.add_argument("--user-only", action="store_true",
                help="calibration fine-tunes on the user's examples alone (what a phone can do without the training set)")
ap.add_argument("--ft-epochs", type=int, default=8)
ap.add_argument("--ft-lr", type=float, default=5e-4)
ap.add_argument("--exclude", help="train the shipped-style model without this signer (for tests/replay with SLI_SIGNER); skips evaluation")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--fixtures", action="store_true",
                help="also write tests/fixtures/letters-seq-parity.json and letters-calib.json for the browser port")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "models", "letters-seq.json"))
args = ap.parse_args()

K = args.window
videos = json.load(open(args.letters))
classes = sorted({v["sign"] for v in videos})
cls_index = {c: i for i, c in enumerate(classes)}
labels = Counter(f["label"] for v in videos for f in v["frames"] if f)
MIRROR = min(labels, key=labels.get)
TIPS = [0, 4, 8, 12, 16, 20]
PAIRS = [(a, b) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]
lo, hi = map(float, args.span.split(","))


def normalise(frame):
    """Hand points relative to the wrist, in palm lengths, one handedness mirrored; and the raw
    wrist and palm length for the motion features."""
    p = np.asarray(frame["lms"], dtype=np.float64)
    wrist = p[0, :2].copy()
    palm = max(float(np.hypot(p[9, 0] - p[0, 0], p[9, 1] - p[0, 1])), 1e-6)
    q = (p - p[0]) / palm
    flip = frame["label"] == MIRROR
    if flip:
        q[:, 0] = -q[:, 0]
    return q, wrist, palm, flip


def windows_of(video, stride, rng=None):
    """(points K×21×3, motion K×2) for each window ending in the letter span, at one frame rate."""
    frames = video["frames"][(rng.integers(stride) if rng is not None else 0)::stride]
    hand_idx = [i for i, f in enumerate(frames) if f]
    if not hand_idx:
        return []
    n = len(hand_idx)
    ends = hand_idx[int(n * lo): max(int(n * hi), int(n * lo) + 1)]
    out = []
    for end in ends:
        idx = list(range(end - K + 1, end + 1))
        if idx[0] < 0 or sum(1 for i in idx if frames[i]) < max(1, K - 2):
            continue
        pts, wrists, palm_end = [], [], None
        last = None
        for i in idx:
            f = frames[i] or last  # a missed detection repeats the previous frame
            if f is None:
                f = next(frames[j] for j in idx if frames[j])
            last = f
            q, w, palm, flip = normalise(f)
            pts.append(q)
            wrists.append(w)
        q_end, w_end, palm_end, flip_end = normalise(frames[end])
        motion = (np.array(wrists) - w_end) / palm_end  # path of the wrist, ending at 0
        if flip_end:
            motion[:, 0] = -motion[:, 0]
        out.append((np.array(pts), motion))
    return out


def featurise(P, M):
    """P: N×K×21×3 points, M: N×K×2 motion -> N×(K·(63+15+2)) features."""
    d = np.stack([np.linalg.norm(P[:, :, a] - P[:, :, b], axis=-1) for a, b in PAIRS], -1)
    if args.shape_last:
        last = np.concatenate([P[:, -1].reshape(len(P), -1), d[:, -1]], -1)
        return np.concatenate([last, M.reshape(len(M), -1)], -1).astype(np.float32)
    return np.concatenate([P.reshape(*P.shape[:2], -1), d, M], -1).reshape(len(P), -1).astype(np.float32)


def augment(P, M, rng):
    """One rotation/stretch per window (a person's hand posture), jitter per frame, and a small
    change of motion size and direction."""
    N = len(P)
    th = rng.uniform(-0.3, 0.3, N)[:, None, None]
    c, s = np.cos(th), np.sin(th)
    P = P.copy()
    x, y = P[..., 0].copy(), P[..., 1].copy()
    P[..., 0], P[..., 1] = c * x - s * y, s * x + c * y
    P *= rng.uniform(0.85, 1.15, (N, 1, 1, 3))
    P += rng.normal(0, 0.03, P.shape)
    mx, my = M[..., 0].copy(), M[..., 1].copy()
    th2 = th[..., 0]
    M = np.stack([np.cos(th2) * mx - np.sin(th2) * my, np.sin(th2) * mx + np.cos(th2) * my], -1)
    M = M * rng.uniform(0.7, 1.3, (N, 1, 1)) + rng.normal(0, 0.05, M.shape)
    return P, M


def dataset(vids, rng=None, strides=(2,)):
    P, M, y, vid = [], [], [], []
    for k, v in enumerate(vids):
        for stride in strides:
            for p, m in windows_of(v, stride, rng):
                P.append(p)
                M.append(m)
                y.append(cls_index[v["sign"]])
                vid.append(k)
    return np.array(P), np.array(M), np.array(y), np.array(vid)


def train(P, M, y, rng, init=None, epochs=None, lr=2e-3):
    epochs = epochs or args.epochs
    X0 = featurise(P, M)
    sizes = [X0.shape[1], args.hidden, 128, len(classes)]
    if init:
        W, B = [w.copy() for w in init[0]], [b.copy() for b in init[1]]
    else:
        W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        B = [np.zeros(b, np.float32) for b in sizes[1:]]
    params = W + B
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    step, wd, batch = 0, 1e-4, 256
    for epoch in range(epochs):
        Xa = featurise(*augment(P, M, rng))
        order = rng.permutation(len(Xa))
        for i in range(0, len(Xa), batch):
            idx = order[i: i + batch]
            acts = [Xa[idx]]
            for k in range(len(W)):
                z = acts[-1] @ W[k] + B[k]
                acts.append(np.maximum(z, 0) if k < len(W) - 1 else z)
            e = np.exp(acts[-1] - acts[-1].max(1, keepdims=True))
            g = e / e.sum(1, keepdims=True)
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
                m1[j] = 0.9 * m1[j] + 0.1 * gp
                m2[j] = 0.999 * m2[j] + 0.001 * gp * gp
                p -= rate * (m1[j] / (1 - 0.9 ** step)) / (np.sqrt(m2[j] / (1 - 0.999 ** step)) + 1e-8)
    return W, B


def embed(W, B, X):
    """Activations of the last hidden layer."""
    a = X
    for k in range(len(W) - 1):
        a = np.maximum(a @ W[k] + B[k], 0)
    return a


def video_acc(prob, vt, vids):
    return float(np.mean([prob[vt == k].mean(0).argmax() == cls_index[v["sign"]] for k, v in enumerate(vids) if (vt == k).any()]))


def predict(W, B, X):
    a = X
    for k in range(len(W)):
        a = a @ W[k] + B[k]
        if k < len(W) - 1:
            a = np.maximum(a, 0)
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


rng = np.random.default_rng(args.seed)
report = {}
if args.exclude:
    videos = [v for v in videos if v["signer"] != args.exclude]
for held in ([] if args.exclude else sorted({v["signer"] for v in videos})):
    # Train at 10-30 fps (strides 1-3 of the 30 fps source); test at 15 fps.
    P, M, y, _ = dataset([v for v in videos if v["signer"] != held], rng, strides=(1, 2, 3))
    W, B = train(P, M, y, rng)
    test = [v for v in videos if v["signer"] == held]
    if args.calibrate:
        calib, test = [], []
        for c in classes:
            vs = sorted((v for v in videos if v["signer"] == held and v["sign"] == c), key=lambda v: v["video"])
            calib += vs[: args.calibrate]
            test += vs[args.calibrate:]
        Pt, Mt, yt, vt = dataset(test)
        Xt = featurise(Pt, Mt)
        base = predict(W, B, Xt)
        Pc, Mc, yc, _ = dataset(calib, rng, strides=(1, 2, 3))
        # Prototypes: mean hidden activation of the user's own examples of each letter.
        Ec, Et = embed(W, B, featurise(Pc, Mc)), embed(W, B, Xt)
        proto = np.stack([Ec[yc == i].mean(0) if (yc == i).any() else np.zeros(Ec.shape[1]) for i in range(len(classes))])
        nrm = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
        sim = nrm(Et) @ nrm(proto).T
        e = np.exp((sim - sim.max(1, keepdims=True)) / 0.05)
        pproto = e / e.sum(1, keepdims=True)
        # Fine-tuning: a few epochs on the training signers plus the user's examples, repeated.
        if args.user_only:
            Wf, Bf = train(Pc, Mc, yc, rng, init=(W, B), epochs=args.ft_epochs, lr=args.ft_lr)
        else:
            rep_n = max(1, len(P) // (5 * max(len(Pc), 1)))
            Wf, Bf = train(np.concatenate([P] + [Pc] * rep_n), np.concatenate([M] + [Mc] * rep_n),
                           np.concatenate([y] + [yc] * rep_n), rng, init=(W, B), epochs=args.ft_epochs, lr=args.ft_lr)
        tuned = predict(Wf, Bf, Xt)
        report[held] = {
            "base": round(video_acc(base, vt, test), 3),
            "prototypes": round(video_acc(0.5 * base + 0.5 * pproto, vt, test), 3),
            "finetuned": round(video_acc(tuned, vt, test), 3),
            "test videos": len(test),
        }
        print("held-out signer", held, report[held], flush=True)
        continue
    Pt, Mt, yt, vt = dataset(test)
    prob = predict(W, B, featurise(Pt, Mt))
    vid_ok = [prob[vt == k].mean(0).argmax() == cls_index[v["sign"]] for k, v in enumerate(test) if (vt == k).any()]
    conf = Counter((classes[a], classes[b]) for a, b in zip(yt, prob.argmax(1)) if a != b).most_common(5)
    report[held] = {"windows": round(float((prob.argmax(1) == yt).mean()), 3), "videos": round(float(np.mean(vid_ok)), 3), "confusions": conf}
    print("held-out signer", held, report[held], flush=True)
if args.calibrate and report:
    for key in ("base", "prototypes", "finetuned"):
        print(f"calibrate {args.calibrate}, {key}: {np.mean([r[key] for r in report.values()]):.3f}")
    raise SystemExit
if report:
    print(f"window {K}: mean window acc {np.mean([r['windows'] for r in report.values()]):.3f}, video acc {np.mean([r['videos'] for r in report.values()]):.3f}")
if args.quick:
    raise SystemExit

def model_json(W, B, extra=None):
    return {
        "classes": classes, "mirror": MIRROR, "window": K, "shapeLast": args.shape_last, "layers": [
            {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "w": [round(float(x), 5) for x in w.reshape(-1)], "b": [round(float(x), 5) for x in b]}
            for w, b in zip(W, B)
        ], **(extra or {}),
    }


def rounded(model):
    return ([np.array(l["w"], np.float32).reshape(l["rows"], l["cols"]) for l in model["layers"]],
            [np.array(l["b"], np.float32) for l in model["layers"]])


P, M, y, _ = dataset(videos, rng, strides=(1, 2, 3))
W, B = train(P, M, y, rng)
model = model_json(W, B, {"heldOut": report})
json.dump(model, open(args.out, "w"), separators=(",", ":"))
print("wrote", args.out, os.path.getsize(args.out) // 1024, "KB")

if args.fixtures:
    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures")
    strip = lambda frames: [{"label": f["label"], "lms": [[round(c, 4) for c in pt] for pt in f["lms"]]} if f else None for f in frames]
    # Parity: windows over two videos (both handedness labels), as the browser builds them from a stream.
    Wr, Br = rounded(model)
    lo, hi = 0.0, 1.0
    picks = [next(v for v in videos if any(f and f["label"] == lab for f in v["frames"])) for lab in sorted(labels)]
    cases = []
    for v in picks:
        frames = strip(v["frames"][::2])
        wins = windows_of({"frames": frames}, 1)
        ends = [i for i in range(len(frames)) if frames[i] and i >= K - 1 and sum(1 for j in range(i - K + 1, i + 1) if frames[j]) >= max(1, K - 2)]
        Pw = np.array([w[0] for w in wins])
        Mw = np.array([w[1] for w in wins])
        X = featurise(Pw, Mw)
        cases.append({"frames": frames, "ends": ends, "features": X.round(5).tolist(), "probs": predict(Wr, Br, X).round(5).tolist()})
    json.dump({"cases": cases}, open(os.path.join(fixtures, "letters-seq-parity.json"), "w"))
    # Calibration: a base model that never saw signer 03, one recording per letter from them, and
    # three more per letter to test on. The browser's fine-tuning must reproduce the gain.
    lo, hi = map(float, args.span.split(","))
    base_videos = [v for v in videos if v["signer"] != "03"]
    Pb, Mb, yb, _ = dataset(base_videos, rng, strides=(1, 2, 3))
    Wb, Bb = train(Pb, Mb, yb, rng)
    base_model = model_json(Wb, Bb)
    calib, test = [], []
    for c in classes:
        vs = sorted((v for v in videos if v["signer"] == "03" and v["sign"] == c), key=lambda v: v["video"])
        calib.append({"sign": c, "frames": strip(vs[0]["frames"])})
        test += [{"sign": c, "frames": strip(v["frames"])} for v in vs[1:4]]
    Wbr, Bbr = rounded(base_model)
    Pt, Mt, yt, vt = dataset(test)
    base_acc = video_acc(predict(Wbr, Bbr, featurise(Pt, Mt)), vt, test)
    Pc, Mc, yc, _ = dataset(calib, rng, strides=(1, 2, 3))
    Wf, Bf = train(Pc, Mc, yc, rng, init=(Wbr, Bbr), epochs=args.ft_epochs, lr=args.ft_lr)
    tuned_acc = video_acc(predict(Wf, Bf, featurise(Pt, Mt)), vt, test)
    json.dump({"model": base_model, "span": [lo, hi], "calibration": calib, "test": test,
               "python": {"base": round(base_acc, 4), "finetuned": round(tuned_acc, 4), "epochs": args.ft_epochs, "lr": args.ft_lr}},
              open(os.path.join(fixtures, "letters-calib.json"), "w"), separators=(",", ":"))
    print("fixtures written; calibration base", round(base_acc, 3), "finetuned", round(tuned_acc, 3))
