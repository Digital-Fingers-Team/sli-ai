# Sentence model rounds

Word error rate (WER, lower is better) on Isharah-1000, signer-independent split (SI): 10 train
signers, 1 dev signer (model selection), 4 test signers (evaluated once, at the end).

## Targets

- **Stopping target (spec):** the Isharah paper reports no pose-based baseline, so its best SI
  baseline of any kind: Swin-MSTP (RGB video), **dev 17.9 / test 26.6**. Source: Alyami et al.,
  arXiv 2506.03615, Table 4 (US: best SMKD dev 56.6 / test 48.0).
- **For reference, pose-based results on the same splits (MSLR 2025 challenge,** arXiv 2508.09372**):**
  CNN-BiLSTM baseline dev 14.54 / test 22.62; best entry (Signer-Invariant Conformer) dev 7.31 /
  test 13.07. US: best dev 55.08 / test 47.78.

## Rounds

| Round | Date | Model | Change (one thing) | Epochs | Dev WER | Kaggle GPU h | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2026-09-28 | large | KArSL isolated-sign pre-training (502 signs) | 30 | 0.032 (KArSL test, seen signers) | 0.7 (baraasaad) | init for Isharah rounds |
| 1 | 2026-09-28 | large | Isharah SI, init from round 0 | 60 | **0.262** | 1.8 (baraasaad) | train loss 48 -> 0.016 while dev flat from epoch ~40: overfits the 10 training signers |
| 2 | 2026-09-28 | large | no pre-training (vs round 1) | 60 | 0.893 | 1.6 (mono768) | never leaves the all-blank solution (loss stuck ~5.2): pre-training is required |
| 3 | 2026-09-28 | large | strong augmentation (vs round 1) | 60 | **0.215** | 1.7 (baraasaad) | rot 25/15, scale 0.3/0.2, speed 0.6-1.4, more/longer hand drops. Best so far |
| 4 | 2026-09-28 | large | dropout 0.1 -> 0.3 (vs round 1) | 60 | 0.261 | 1.8 (mono768) | no gain: dropped |
| 5 | 2026-09-28 | large | xstrong augmentation (vs round 3) | 60 | **0.204** | 1.7 (baraasaad) | best so far; still improving at epoch 59; rot 35/20, scale 0.4/0.25, speed 0.5-1.5, 3 hand drops up to 1.5 s |
| 6 | 2026-09-28 | large | strong augmentation + dropout 0.2 (vs round 3) | 60 | 0.241 | 1.7 (mono768) | dropout hurts: dropped for good |
| s0 | 2026-09-28 | small | KArSL pre-training | 30 | 0.010 (KArSL test, seen signers) | 0.6 (selia097) | init for small rounds |
| s1 | 2026-09-28 | small | Isharah, init s0, normal augmentation (plain baseline) | 60 | 0.268 | 1.5 (selia097) | baseline for distillation |
| 7 | 2026-09-28 | large | xstrong, 100 epochs (vs round 5) | 100 | **0.1758** | 2.8 (baraasaad) | beats the 17.9 target; best large so far |
| 8 | 2026-09-28 | large | xxstrong, 100 epochs (vs round 7) | 100 | 0.181 | 2.8 (mono768) | augmentation axis saturated (slightly better hidden hand); rot 45/25, scale 0.5/0.3, speed 0.45-1.6, 4 hand drops |
| s2 | 2026-09-28 | small | distill from round 5 + xstrong augmentation | 80 | **0.1735** | 2.6 (selia097) | beats the 17.9 target and its own teacher (0.204) |
| 9 | 2026-09-28 | large | self-distillation from round 5 + xstrong, 100 epochs | 100 | **0.1745** | 2.1 (selia097) | best large: sentence accuracy 67.2% vs 64.5% (round 7); served as SLI Sentences Large v2 |

## Stress tests (dev WER: plain / one hand hidden 0.5-1.5 s, ~1/3 of frames / 1.5x faster)

| Round | Plain | Hidden hand | Fast |
| --- | --- | --- | --- |
| 1 | 0.262 | 0.309 | 0.250 |
| 3 | 0.215 | 0.228 | 0.214 |
| 4 | 0.261 | 0.298 | 0.239 |
| 5 | 0.204 | 0.214 | 0.206 |
| 6 | 0.241 | 0.263 | 0.228 |
| s1 | 0.268 | 0.297 | 0.248 |
| s2 | 0.173 | 0.180 | 0.167 |
| 7 | 0.176 | 0.191 | 0.169 |
| 8 | 0.181 | 0.185 | 0.178 |
| 9 | 0.1745 | 0.180 | 0.168 |

## No face (signer far from the camera; the app's FaceLandmarker misses small faces)

Found by the e2e test: on a wide ArabSign frame FaceLandmarker finds no face (0/15 frames) while
pose and hands are found; cropped to phone-like framing it finds 15/15. Training data (Holistic)
almost always has a face. Cost of losing the face entirely (dev WER, exported ONNX):

| Model | Plain | No face |
| --- | --- | --- |
| large r7 | 0.1758 | 0.1875 |
| small s2 | 0.1735 | 0.1940 |
| large r9 | 0.1745 | 0.1917 |

1-2 points: no special retraining. The e2e camera uses phone-like framing.

## Community model: Large US (HF bibo18, 2026-10-09) — served since 2026-10-15

`isharah-large-US.onnx` from DF-Team/sli-sentences, same I/O as Large v2. It was fine-tuned on the
Isharah **US** train split, which holds 900 of the 1000 SI-dev videos, so its SI-dev score is not
an honest unseen-signer number. `eval_clean.py` scores only the 100 SI-dev videos outside US train.

| Model | SI dev (leaky for US) | US dev | Clean SI dev (n=100) |
| --- | --- | --- | --- |
| Large v2 (r9) | 0.1745 | 0.1349 | 0.4168 |
| Large US | 0.1327 | 0.1131 | 0.4082 |

Better on unseen sentences, a tie on the clean set. Rollback: copy
`~/sli-work/models/sentences-large-r9.onnx` over `sentences-large.onnx`, restore MODEL_ID/NAME in
app.py, run `server/sentence-api/run.sh`.
