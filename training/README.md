# Model, data and evaluation

The recognition model is the pretrained ST-Transformer from
[yousefelkilany/Word-level-Arabic-Sign-language](https://github.com/yousefelkilany/Word-level-Arabic-Sign-language)
(MIT), trained on KArSL-502. This folder reproduces its preprocessing, measures it on held-out
data, and builds the assets the app ships.

| File | Purpose |
| --- | --- |
| `sli_kps.py` | Keypoint extraction + normalisation, identical to what the model was trained on. `src/recognition/features.ts` is its browser port. |
| `face_idx.json` | The 136 face-mesh points the model uses (contours + irises). |
| `karsl_labels.json` | KArSL-502 labels (Arabic / English), from the same repo. |
| `build_lexicon.py` | Builds `src/data/signs.json`: categories and spelling variants for text → sign. |
| `eval_pretrained.py` | Accuracy on KArSL test videos; caches raw landmarks in `cache/`. |
| `summarize_eval.py` | Turns eval reports into `src/data/metrics.json` and the tables below. |
| `export_parity.py` | Fixture proving the TypeScript features match Python (`tests/unit/features.test.ts`). |
| `make_clips.py` | The signer videos in `public/clips/` (signer 02, KArSL test split, H.264 256×256). |
| `make_fake_camera.py` | A `.y4m` camera for the Playwright tests. |

Setup: Python 3.9+, `pip install mediapipe==0.10.14 onnxruntime opencv-python-headless numpy py7zr`.
KArSL is on the dataset authors' Google Drive (linked from https://hamzah-luqman.github.io/KArSL/);
the test archives unpack to `<signer>/<sign>/<video>/<frame>.jpg`.

## Isolated signs

One test video per sign per signer, each classified whole:

**1506 test videos, 3 signers: top-1 97.0%, top-5 99.6%**

| Category | Top-1 |
| --- | --- |
| home | 93.6% (160/171) |
| traits | 93.9% (138/147) |
| family | 94.8% (91/96) |
| verbs | 96.9% (93/96) |
| jobs | 97.7% (129/132) |
| numbers | 97.8% (91/93) |
| places | 97.9% (47/48) |
| religion | 98.1% (303/309) |
| letters | 98.3% (115/117) |
| health | 98.9% (264/267) |
| social | 100.0% (30/30) |

Signs missed by more than one signer: قطارة, يصبغ (→ صباغ), مسجل, كاميرا فوتوغرافية, آية.

KArSL splits train/test by repetition, not by person, so all three signers were seen in training.
Expect lower accuracy for new people.

## Sentences through the app's pipeline

`tests/replay` feeds cached landmarks through the same TypeScript features, decoder and model
the browser uses: 40 sentences of 3 random signs with hands-down pauses, subsampled to a camera
frame rate. Run with `SLI_REPLAY=1 npx vitest run` (`SLI_FPS`, `SLI_OPTS` to experiment).

| Camera fps | Words right | Sentences exactly right | Extra words per sentence |
| --- | --- | --- | --- |
| 5 | 62% | 37% | 0 |
| 10 | 89% | 78% | 0 |
| 15 | 95% | 88% | 0.03 |
| 30 | 93% | 88% | 0.05 |

What moved these numbers while tuning the decoder (`src/recognition/decoder.ts`):

- Keeping frames where the hand is momentarily lost inside a sign: the training clips have them,
  and classifying only hand frames turned correct answers wrong.
- No lead-in before the hand appears (400 ms of lead-in cut word accuracy from ~95% to ~70%).
- Committing mid-sign only when the same answer holds at ≥ 0.9 for 4 checks (~1 s); looser early
  commits added wrong words.
- After an early commit, the rest of the segment counts as a new sign only if it lasts ≥ 700 ms
  at ≥ 0.8.

Frame rate matters most: below ~10 fps there are too few frames per sign.

## Browser parity

On the same KArSL frames, the browser's MediaPipe landmarks are identical to Python's
(mean difference 0.0) and give the same prediction; `tests/unit/features.test.ts` checks the
feature code and model output against Python on three real videos.
