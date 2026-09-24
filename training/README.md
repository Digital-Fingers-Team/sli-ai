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
| `eval_video_mode.py` | Which MediaPipe speed-ups (tracking per part, pose/face every other frame) keep the word model's accuracy. |
| `extract_letters.py` | Hand landmarks of every KArSL letter video, for the letter model. |
| `eval_location.py` | How much the word model relies on where the hands are relative to the body. |
| `extract_letters_pose.py` | Nose and shoulders for every letter video, for the letter model's location input. |
| `eval_two_hands.py` | Whether to track one hand or two, and how to feed them to the word model. |
| `train_letters_seq.py` | The shipped letter model (hand shape + 4-frame wrist path, `public/models/letters-seq.json`), calibration experiments (`--calibrate`), and the browser fixtures (`--fixtures`). |
| `train_letters.py` | Trains the per-frame letter model (`public/models/letters.json`), reports leave-one-signer-out accuracy, writes the parity fixture. |

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

## Landmark speed-ups

MediaPipe's tracking (VIDEO) mode skips detection on most frames, but the model was trained on
landmarks detected frame by frame. `eval_video_mode.py` on 126 held-out videos (signer 02,
every 4th sign, 15 fps):

| Landmarks | Top-1 |
| --- | --- |
| All detected per frame (as in training) | 98.4% |
| Pose and face refreshed every other frame | 98.4% |
| Pose and face tracked | 99.2% |
| Hands tracked | 87.3% |
| Everything tracked | 89.7% |

So the app detects hands on every frame and tracks pose and face. On devices below 18 fps
pose and face are refreshed only every other frame (in sentences that costs 2-3 points, see below).

## Hand location

Where a sign is made (at the head, chest, a shoulder) is part of the sign. The word model sees
it through the pose points: elbows and wrists relative to the left shoulder, scaled by
shoulder width (hand points are relative to their own wrist). `eval_location.py` removes that
information on 504 held-out videos:

| Arm points given to the word model | Top-1 |
| --- | --- |
| As recorded | 96.8% |
| Held at their average place (no arm movement) | 77.8% |
| Same movement, moved to another sign's place | 62.5% |
| None (hand shape and face only) | 27.8% |

So location is a large part of what the word model reads. For letters it is not: adding the
wrist's place relative to the shoulders and the nose to the letter model (`--location`, body
points from `extract_letters_pose.py`) gave 76.7% vs 75.9% at 40 epochs but 76.1% vs 76.5% at
60, i.e. within run-to-run noise. It would also need body tracking in Letters mode, which
skips it to stay fast, so the shipped letter model reads the hand shape only. Its remaining
confusions (ي/ى/ئ, ج/ح) differ by movement, which a one-frame model cannot see.

## Two hands

The word model was trained with one hand per frame (`num_hands=1` in its extractor), but its
input has a right- and a left-hand slot. `eval_two_hands.py` detects two hands on 504 held-out
videos (every 3rd sign, all three signers; 316 of them show two hands at some point):

| Hands given to the model | Top-1 |
| --- | --- |
| One (as in training) | 97.0% |
| Both | **99.0%** |
| One of two: the first detected | 97.0% |
| One of two: the one that moved most | 98.2% |
| One of two: the raised one | 93.3% |

So the app tracks two hands and gives the word model both. For letters (one hand) it uses the
raised hand, which is the spelling hand when the other rests.

## Sentences through the app's pipeline

`tests/replay` feeds cached landmarks through the same TypeScript features, decoder and model
the browser uses: 3 random signs per sentence, subsampled to a camera frame rate, either with
the hands lowered between signs or signed straight through (each clip's hand-less start and
end cut, `SLI_CONTINUOUS=1`). Words are counted in order (longest common subsequence), so one
missed word does not mark the rest wrong. Run with `SLI_REPLAY=1 npx vitest run` (`SLI_FPS`,
`SLI_OPTS`, `SLI_BODY_EVERY`, `SLI_REPLAY_TRIALS` to experiment).

80 sentences each, pose and face every frame:

| Camera fps | Words right, pause between signs | Words right, no pause | Word shown after the sign ends (median) |
| --- | --- | --- | --- |
| 10 | 91% | 60% | 0.35 s |
| 15 | 93% | 54% | 0.27 s |
| 30 | 94% | 54% | 0.30 s |

Pose and face every other frame, 15 fps: 90% / 53%.

The decoder (`src/recognition/decoder.ts`) got there by:

- Classifying the growing window since the current sign began: its confidence peaks where the
  sign ends and drops once the next sign starts, so the peak is committed after two checks
  confirm the drop and the next sign starts from the peak. Against the earlier rule (commit
  only when the hands drop or one answer stays at 0.9 for a second): 36% → 54% with no pause,
  94% → 93% with pauses, 15 fps.
- Rules that commit whenever the short-window answer changes were tried and dropped: partway
  through a sign the model is often confidently wrong, and they fell to 62-75% with pauses.
- Hands down for 300 ms ends a sign (450 ms before: same accuracy, words showed 130 ms later).
- Keeping frames where the hand is momentarily lost inside a sign, and no lead-in before the
  hand appears (400 ms of lead-in cut word accuracy from ~95% to ~70%).

Signing without pauses stays much harder: the model only ever saw single signs that start and
end with the hands down. A continuous-signing model would need sentence-level training data.

## Letters

KArSL has 39 letter signs, 8 videos per signer each. `train_letters.py` trains a small MLP on
one hand's 21 points per frame (relative to the wrist, scaled by palm size, one handedness label
mirrored, plus fingertip distances), from the middle of each video's hand frames, with rotation,
stretch and jitter augmentation. Accuracy is leave-one-signer-out, i.e. on a person the model
never saw:

| Held-out signer | Frames | Videos (averaged over the held letter) |
| --- | --- | --- |
| 01 | 78.6% | 81.1% |
| 02 | 69.4% | 71.2% |
| 03 | 74.0% | 77.3% |
| **Mean** | **74.0%** | **76.5%** |

Variants tried (videos): without mirroring 74.9%, without fingertip distances 73.4%, without either 72.1%.

## Letters over a short window, and calibration

`train_letters_seq.py` gives the letter model the last few frames (15 fps, trained at 10-30).
Leave-one-signer-out, per video:

| Letter model | Seed 0 | Seed 1 |
| --- | --- | --- |
| One frame (same protocol) | 75.2% | 75.9% |
| 8 frames, full hand each | 73.7% | |
| Newest hand shape + 8-frame wrist path | 71.0% | |
| **Newest hand shape + 4-frame wrist path (shipped)** | **77.0%** | **76.9%** |

A small, consistent gain; the letters that differ by movement (ي/ى/ئ, ج/ح) stay the main
errors, so movement is not what the model is missing. What is missing is people: the model
has only three signers to learn from. So the app can adapt to its user ("Teach the app your
hand", `src/pages/teach.ts`): the user signs each letter once, and the model is fine-tuned on
the device. Simulated with each held-out signer's first video per letter as the recording
(`--calibrate 1`), tested on their other videos:

| Adaptation | Letters right |
| --- | --- |
| None | 76.9% |
| Prototypes (nearest recorded letter, blended) | 78.7% |
| Fine-tune on the recordings + training data, 8 epochs | 82.9% |
| Fine-tune on the recordings only, 8 epochs | 83.3% |
| **Fine-tune on the recordings only, 30 epochs (shipped)** | **86.4%** |

Two recordings per letter with training data: 84.7%. Fine-tuning on the recordings alone is
what a phone can do (the training set is not shipped), and it is the best. The browser port
(`src/recognition/letternet.ts`) reproduces it: `tests/unit/letters.test.ts` fine-tunes a model
that never saw signer 03 on one recording per letter, 79.8% → 90.4% (Python: 89.5%).

## Words and letters together (Auto mode)

`src/recognition/interpreter.ts` runs the word decoder and the speller together. Hand speed
separates them well: holding a letter, the hand moves 0.27 palm lengths per second (median),
signing a word 4.9 (signer 01, 15 fps), so the speller only types while the hand is below 1.0.
Words have still moments too, so a letter is also refused while the hand is low (in the lap)
or the word model, which also knows the 39 letter signs, confidently sees a non-letter word.

Replay on signer 01 only, with a letter model trained without signer 01
(`train_letters.py --exclude 01`), 60 sentences each, 15 fps, words right (LCS):

| Sentences | Words mode | Letters mode | Auto, no guards | Auto |
| --- | --- | --- | --- | --- |
| 3 signs, pauses | 90% | | 83% | 89% |
| 3 signs, no pauses | 59% | | 57% | 61% |
| word, letter, letter, word, pauses | 95% | | 86% | 91% |
| word, letter, letter, word, no pauses | 40% | | 50% | 47% |
| 3 letters, pauses | | 80% | 94% | 94% |
| 3 letters, no pauses | | 78% | 59% | 43% |

The guards remove the stray letters Auto added inside word sentences, at the cost of spelling
without pauses, where Letters mode stays the better choice. A "spelling session" (after a
typed letter, only very confident words until the hands drop) was tried and made it worse. Most errors are
letters that share a hand shape and differ by movement or a hamza: ي/ى/ئ, ن/ئـ, أ/ئـ, ت/ة,
ا/آ, ج/ح, ز/ذ, ر/د. The shipped model is trained on all three signers.

## Browser parity

On the same KArSL frames, the browser's MediaPipe landmarks are identical to Python's
(mean difference 0.0) and give the same prediction; `tests/unit/features.test.ts` checks the
feature code and model output against Python on three real videos.
