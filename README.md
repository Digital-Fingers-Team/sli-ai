# SLI — AI Arabic Sign Language Interpreter

SLI translates **Arabic Sign Language ⇄ Arabic text ⇄ speech**, entirely in the browser.

- **Sign → text → speech.** Sign in front of the camera; the word appears while you sign and
  builds a sentence that can be spoken aloud in Arabic. Each word can be swapped for the
  model's next-best guesses.
- **Live fingerspelling.** In Letters mode each hand shape is read frame by frame, so letters
  are typed one after another without lowering the hand.
- **Text or voice → sign.** Type or speak Arabic; each word is shown as a real signer video.
  Unknown words and names are fingerspelled with letter signs, and numbers are composed from
  number signs.
- **Conversation.** A deaf and a hearing person talk on one screen, with a shared transcript.
- **Dictionary.** All 502 signs, searchable, with a video for each.

Recognition runs on the device (MediaPipe + ONNX Runtime Web), in Web Workers so the camera
view stays smooth. Camera images never leave the device, and after the first visit the app
works offline.

## How it works

| Step | What happens | Code |
| --- | --- | --- |
| Landmarks | MediaPipe in a worker: hands detected on every frame, pose and face tracked | `src/recognition/landmarks*.ts` |
| Features | 184 keypoints × (x, y, z, visibility), normalised exactly as in training | `src/recognition/features.ts` |
| Segmenting | Commits a sign when the next one takes over or the hands drop for 300 ms | `src/recognition/decoder.ts` |
| Classifying | Frames resampled to 50 and classified by an ST-Transformer over 502 KArSL signs, in its own worker | `src/recognition/classifier*.ts` |
| Letters | A per-frame hand-shape model (trained here on KArSL's letter videos) and a hold-to-type speller | `src/recognition/letters.ts`, `speller.ts` |
| Text → sign | Arabic normalisation, phrase matching, light stemming, numbers, fingerspelling | `src/translate/` |

The recognition model is the MIT-licensed
[Word-level Arabic Sign Language ST-Transformer](https://github.com/yousefelkilany/Word-level-Arabic-Sign-language),
trained on [KArSL-502](https://hamzah-luqman.github.io/KArSL/) (Unified Arabic Sign Language: 31 numbers,
39 letters and 432 words).

## Measured accuracy

On the KArSL **test** split (details and method in `training/README.md`):

- **Isolated signs:** 97.0% top-1, 99.6% top-5 over 1,506 videos (502 signs × 3 signers).
- **3-sign sentences through the app pipeline, 15 fps:** 93% of words with a short pause
  between signs, 54% when signing straight through without lowering the hands.
- **Letters, on a signer the model never saw:** 76.5% (leave-one-signer-out over KArSL's three
  signers). Most errors are letters sharing a hand shape (ي/ى/ئ, ت/ة, ج/ح, ز/ذ).

The word model was trained on sign clips that start and end with the hands down, so it is best
with a brief pause between signs. Its three test signers also appear in its training data, so
accuracy with new people is lower. Good light and the whole upper body in frame help most.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173
npm test             # unit tests (features parity with Python, decoder, text→sign planner)
npm run build        # typecheck + production build in dist/
```

End-to-end tests drive real Chromium with KArSL videos as the camera:

```bash
python training/make_fake_camera.py tests/fixtures/camera.y4m KARSL_DIR 01 290 497 --slow 4
python training/make_fake_camera.py tests/fixtures/camera-letters.y4m KARSL_DIR 01 33 54 --slow 4
npx playwright test
```

`?delegate=CPU` in the URL forces MediaPipe onto the CPU (for devices with broken WebGL),
`?hands=video` / `?body=image` change the per-part tracking modes, `?bodyEvery=N` fixes how
often pose and face are refreshed, and `?debug` keeps raw landmarks on `window.__sliFrames`.

## Credits

- **Digital Fingers team** — Anas Mohamed Mokhtar, Iyad Abdel Raouf Samir.
- **KArSL** — Sidig, Luqman, Mahmoud, Mohandes. *KArSL: Arabic Sign Language Database.*
  ACM TALLIP 20(1), 2021. Sign videos and labels.
- **Word-level ArSL ST-Transformer** — Yousef Elkilany, MIT License.
- **MediaPipe** (Apache 2.0) and **ONNX Runtime Web** (MIT).
