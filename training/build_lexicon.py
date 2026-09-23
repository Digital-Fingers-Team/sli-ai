"""Generate src/data/signs.json from the KArSL-502 labels.

Each entry: {id, ar, en, cat, aliases}. `id` is the model's class index (0-based);
aliases are the alternative spellings packed into the label ("كأس ( كوب)", "مخدر/ بنج").
Clip availability is filled in later by make_clips.py.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
labels = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "karsl_labels.json")))
AR, EN = labels["AR_WORDS"], labels["EN_WORDS"]

# 1-based KArSL sign numbers -> category
CATS = [
    (1, 31, "numbers"), (32, 70, "letters"), (71, 159, "health"), (160, 191, "verbs"),
    (192, 223, "family"), (224, 272, "traits"), (273, 288, "places"), (289, 298, "social"),
    (299, 355, "home"), (356, 458, "religion"), (459, 502, "jobs"),
]

# Extra everyday spellings that mean exactly the same sign (not guesses at other signs).
EXTRA = {
    "أهلا وسهلاً": ["أهلا", "اهلا", "أهلاً", "مرحبا", "أهلا وسهلا", "اهلا وسهلا"],
    "شكراً": ["شكرا", "متشكر", "شكرًا"],
    "السلام عليكم": ["سلام عليكم", "السلام"],
    "سعيد (مسرور)": ["فرحان", "مبسوط"],
    "طبيب": ["دكتور"],
    "معلم / مدرس": ["أستاذ", "استاذ"],
    "هاتف ( تلفون)": ["تليفون", "موبايل", "جوال"],
    "دورة مياه (حمام)": ["تواليت"],
    "يصلي / الصلاة": ["صلاة", "يصلى"],
    "يتوضأ / وضوء": ["الوضوء"],
    "بيت": ["منزل", "البيت"],
    "مستشفى": ["مستشفي"],
    "ماء زمزم": ["زمزم"],
    "جمجة": ["جمجمة"],
    "محافظ / وال": ["والي", "محافظ"],
}


def category(n):
    for lo, hi, name in CATS:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def aliases(label):
    label = str(label)
    parts = re.split(r"[/()]", label)
    out = []
    for p in parts:
        p = re.sub(r"[A-Za-z]+", "", p).strip(" -")
        if p:
            out.append(p)
    whole = re.sub(r"\s*[()]\s*", " ", label).replace("/", " ").strip()
    if "-" in label:  # "شهيق - زفير"
        out.append(label.replace(" - ", " "))
    seen = []
    for a in out + EXTRA.get(label, []):
        a = re.sub(r"\s+", " ", a).strip()
        if a and a not in seen and a != whole:
            seen.append(a)
    return seen


signs = []
for i, (ar, en) in enumerate(zip(AR, EN)):
    ar_s = str(ar)
    primary = re.split(r"[/(]", ar_s)[0].strip(" -") or ar_s
    signs.append({
        "id": i,
        "ar": primary,
        "label": ar_s,
        "en": str(en).strip(),
        "cat": category(i + 1),
        "aliases": aliases(ar_s),
    })

out = os.path.join(HERE, "..", "src", "data", "signs.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(signs, open(out, "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(signs)} signs -> {out}")
