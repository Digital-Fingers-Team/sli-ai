"""WER on SI-dev sentences that are NOT in the US train split (leak-free for US-trained models).

    python -m training.sentences.eval_clean A.onnx B.onnx
"""
import json
import os
import sys

from training.sentences.evaluate import evaluate
from training.sentences.train import load_ish

if __name__ == "__main__":
    feats, index, vocab = load_ish(None)
    clean = [r for r in index if r["splits"].get("SI") == "dev" and r["splits"].get("US") != "train"]
    for r in clean:
        r["splits"]["CLEAN"] = "dev"
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            from training.sentences.kaggle_run import find_input
            path = find_input(os.path.basename(path))
        res = evaluate(path, feats, clean, vocab, "CLEAN", "dev")
        print(res, flush=True)
        json.dump(res, open(os.path.join(os.environ.get("SLI_OUT", "."), f"clean-{res['model']}.json"), "w"))
