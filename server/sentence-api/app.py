"""Large sentence model behind a tiny HTTP API (POST /api/sentence).

Body: SF1 features of one sentence as little-endian float32, T x 356, 8 <= T <= 450 (15 fps).
Stateless: requests are neither logged nor stored.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

DIM, MIN_T, MAX_T = 356, 8, 450
QUEUE_WAIT_S = 4
DRAIN_MAX = 2 << 20
MODEL_ID, MODEL_NAME = "large-us", "SLI Sentences Large US"  # community model (HF bibo18, 2026-10-09): 11.3% US-dev WER vs 13.5% for Large v2


def greedy(lp):
    """CTC greedy decoding (blank = 0), as training/sentences/ctc.py."""
    best, probs = lp.argmax(-1), np.exp(lp.max(-1))
    ids, conf, prev, run = [], [], 0, []
    for k, p in zip(best, probs):
        if k != prev and prev != 0:
            conf.append(float(np.mean(run)))
        if k != 0 and k != prev:
            ids.append(int(k))
            run = []
        if k != 0:
            run.append(p)
        prev = k
    if prev != 0:
        conf.append(float(np.mean(run)))
    return ids, conf


def make_server(model, vocab_path, port=8000, workers=1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(model, opts)
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    slots = threading.BoundedSemaphore(workers)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # no request logs
            pass

        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/sentence/health":
                return self.reply(200, {"ok": True, "model": MODEL_ID, "name": MODEL_NAME})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/sentence":
                return self.reply(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_T * DIM * 4:
                self.close_connection = True
                if n <= DRAIN_MAX:  # read it, so a client still uploading gets the 413 and not a broken pipe
                    self.rfile.read(n)
                return self.reply(413, {"error": f"at most {MAX_T} frames"})
            body = self.rfile.read(n)
            if n % (DIM * 4) or n < MIN_T * DIM * 4:
                return self.reply(400, {"error": f"body must be T x {DIM} float32 with T >= {MIN_T}"})
            x = np.frombuffer(body, "<f4").reshape(1, -1, DIM)
            if not np.isfinite(x).all():
                return self.reply(400, {"error": "non-finite values"})
            if not slots.acquire(timeout=QUEUE_WAIT_S):
                return self.reply(503, {"error": "busy"})
            try:
                t = time.perf_counter()
                lp = sess.run(["logprobs"], {"feats": x})[0][0]
                ms = int((time.perf_counter() - t) * 1000)
            finally:
                slots.release()
            ids, conf = greedy(lp)
            glosses = [vocab["glosses"][i - 1] for i in ids]
            key = " ".join(glosses)
            self.reply(200, {"ids": ids, "glosses": glosses, "text": vocab["lookup"].get(key, key),
                             "conf": [round(c, 3) for c in conf], "ms": ms, "model": MODEL_ID})

        def do_PUT(self):
            self.reply(405, {"error": "method not allowed"})

        do_DELETE = do_PATCH = do_PUT

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


if __name__ == "__main__":
    make_server(os.environ["SLI_MODEL"], os.environ["SLI_VOCAB"], int(os.environ.get("SLI_PORT", 8000)),
                int(os.environ.get("SLI_WORKERS", 1))).serve_forever()
