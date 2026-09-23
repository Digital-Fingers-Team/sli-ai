"""Write tests/fixtures/parity.json: raw landmarks of real KArSL test videos plus the model
input and top-5 the Python pipeline produces, so the TypeScript port can be checked exactly.

Usage: python export_parity.py CACHE_JSON [CACHE_JSON ...]
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort

import sli_kps

session = ort.InferenceSession(os.path.join(sli_kps.HERE, "..", "public", "models", "karsl502.onnx"))
cases = []
for path in sys.argv[1:]:
    raws = json.load(open(path))
    x = sli_kps.model_input([sli_kps.features(r) for r in raws])
    logits = session.run(None, {"input": x})[0][0]
    cases.append({
        "name": os.path.basename(path),
        "raws": raws,
        "input_sum": float(np.abs(x).sum()),
        "input_head": x.reshape(-1)[:2000].tolist(),
        "top5": np.argsort(-logits)[:5].tolist(),
    })

out = os.path.join(sli_kps.HERE, "..", "tests", "fixtures", "parity.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(cases, open(out, "w"))
print(f"wrote {len(cases)} cases -> {out}")
