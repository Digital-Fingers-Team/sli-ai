"""Combine eval_pretrained.py reports into src/data/metrics.json and print a markdown summary.

Usage: python summarize_eval.py report_s01.json report_s02.json report_s03.json
"""

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
signs = json.load(open(os.path.join(HERE, "..", "src", "data", "signs.json")))

results = []
for path in sys.argv[1:]:
    results += json.load(open(path))["results"]

n = len(results)
top1 = sum(r["top5"][0] == r["sign"] for r in results) / n
top5 = sum(r["sign"] in r["top5"] for r in results) / n

by_cat = defaultdict(lambda: [0, 0])
misses = defaultdict(list)
for r in results:
    cat = signs[r["sign"]]["cat"]
    by_cat[cat][1] += 1
    if r["top5"][0] == r["sign"]:
        by_cat[cat][0] += 1
    else:
        misses[r["sign"]].append(r["top5"][0])

metrics = {
    "videos": n,
    "signers": len(sys.argv) - 1,
    "top1": round(top1, 4),
    "top5": round(top5, 4),
    "byCategory": {c: round(ok / tot, 4) for c, (ok, tot) in sorted(by_cat.items())},
}
json.dump(metrics, open(os.path.join(HERE, "..", "src", "data", "metrics.json"), "w"), indent=1)

print(f"**{n} test videos, {metrics['signers']} signers: top-1 {top1:.1%}, top-5 {top5:.1%}**\n")
print("| Category | Top-1 |\n| --- | --- |")
for c, (ok, tot) in sorted(by_cat.items(), key=lambda kv: kv[1][0] / kv[1][1]):
    print(f"| {c} | {ok / tot:.1%} ({ok}/{tot}) |")
print("\nSigns missed by more than one signer:\n")
for sign, got in sorted(misses.items(), key=lambda kv: -len(kv[1])):
    if len(got) > 1:
        print(f"- {signs[sign]['ar']} → " + ", ".join(signs[g]["ar"] for g in got))
