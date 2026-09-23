# SLI — AI Arabic Sign Language Interpreter

SLI translates **Arabic Sign Language ⇄ Arabic text ⇄ speech**, entirely in the browser.

- **Sign → text → speech.** Sign in front of the camera; recognised words build a sentence that
  can be spoken aloud in Arabic. Each word can be swapped for the model's next-best guesses.
- **Text or voice → sign.** Type or speak Arabic; each word is shown as a real signer video.
  Unknown words and names are fingerspelled with letter signs, and numbers are composed from
  number signs.
- **Conversation.** A deaf and a hearing person talk on one screen, with a shared transcript.
- **Dictionary.** All 502 signs, searchable, with a video for each.

Recognition runs on the device (MediaPipe + ONNX Runtime Web). Camera images never leave it,
and after the first visit the app works offline.

## How it works

| Step | What happens | Code |
| --- | --- | --- |
| Landmarks | MediaPipe pose, face and hand landmarkers on each camera frame | `src/recognition/landmarks.ts` |
| Features | 184 keypoints × (x, y, z, visibility), normalised exactly as in training | `src/recognition/features.ts` |
| Segmenting | A sign runs from when a hand appears until no hand has been seen for 450 ms | `src/recognition/decoder.ts` |
| Classifying | Frames resampled to 50 and classified by an ST-Transformer over 502 KArSL signs | `src/recognition/classifier.ts` |
| Text → sign | Arabic normalisation, phrase matching, light stemming, numbers, fingerspelling | `src/translate/` |

The recognition model is the MIT-licensed
[Word-level Arabic Sign Language ST-Transformer](https://github.com/yousefelkilany/Word-level-Arabic-Sign-language),
trained on [KArSL-502](https://hamzah-luqman.github.io/KArSL/) (Unified Arabic Sign Language: 31 numbers,
39 letters and 432 words).

## Measured accuracy

On the KArSL **test** split (details and method in `training/README.md`):

- **Isolated signs:** 97.0% top-1, 99.6% top-5 over 1,506 videos (502 signs × 3 signers).
- **3-sign sentences through the app pipeline:** 89% of words right at 10 fps, 95% at 15 fps,
  93% at 30 fps; 62% at 5 fps, so low-end devices will struggle.

The three KArSL signers also appear in the training data, so accuracy with new people is
lower. Good light and the whole upper body in frame help most.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173
npm test             # unit tests (features parity with Python, decoder, text→sign planner)
npm run build        # typecheck + production build in dist/
```

End-to-end tests drive real Chromium with a KArSL video as the camera:

```bash
python training/make_fake_camera.py tests/fixtures/camera.y4m KARSL_DIR 01 290 497 --slow 4
npx playwright test
```

`?delegate=CPU` in the URL forces MediaPipe onto the CPU (for devices with broken WebGL),
and `?debug` keeps raw landmarks on `window.__sliFrames`.

## Credits

- **Digital Fingers team** — Anas Mohamed Mokhtar, Iyad Abdel Raouf Samir.
- **KArSL** — Sidig, Luqman, Mahmoud, Mohandes. *KArSL: Arabic Sign Language Database.*
  ACM TALLIP 20(1), 2021. Sign videos and labels.
- **Word-level ArSL ST-Transformer** — Yousef Elkilany, MIT License.
- **MediaPipe** (Apache 2.0) and **ONNX Runtime Web** (MIT).
