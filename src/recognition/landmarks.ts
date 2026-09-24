import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import type { Point, PosePoint, RawFrame } from './features';

export type Delegate = 'GPU' | 'CPU';

type Mode = 'IMAGE' | 'VIDEO';

export interface LandmarkOptions {
  base: string;
  delegate?: Delegate; // forced delegate; otherwise GPU with CPU fallback
  // IMAGE detects every frame from scratch, like the training extractor; VIDEO tracks between
  // frames and is cheaper. Tracked hands cost the word model accuracy, tracked pose and face do
  // not (training/eval_video_mode.py), so the defaults are IMAGE hands and VIDEO body.
  handsMode: Mode;
  bodyMode: Mode;
}

/** Pose + face + hand landmarkers, configured like the training extractor (1 face, 1 hand). */
export class Landmarks {
  private lastBody: Pick<RawFrame, 'pose' | 'face'> = { pose: null, face: null };

  private constructor(
    private pose: PoseLandmarker,
    private face: FaceLandmarker,
    private hands: HandLandmarker,
    readonly delegate: Delegate,
    private handsMode: Mode,
    private bodyMode: Mode,
  ) {}

  static async load(o: LandmarkOptions, onProgress?: (step: string) => void): Promise<Landmarks> {
    const fileset = await FilesetResolver.forVisionTasks(`${o.base}runtime/mediapipe`);
    const make = async (delegate: Delegate) => {
      const opts = (name: string, runningMode: Mode) => ({
        baseOptions: { modelAssetPath: `${o.base}mediapipe/${name}_landmarker.task`, delegate },
        runningMode,
      });
      onProgress?.('pose');
      const pose = await PoseLandmarker.createFromOptions(fileset, opts('pose', o.bodyMode));
      onProgress?.('face');
      const face = await FaceLandmarker.createFromOptions(fileset, { ...opts('face', o.bodyMode), numFaces: 1 });
      onProgress?.('hand');
      const hands = await HandLandmarker.createFromOptions(fileset, { ...opts('hand', o.handsMode), numHands: 1 });
      return new Landmarks(pose, face, hands, delegate, o.handsMode, o.bodyMode);
    };
    if (o.delegate) return make(o.delegate);
    try {
      return await make('GPU');
    } catch (err) {
      console.warn('GPU delegate unavailable, using CPU', err);
      return make('CPU');
    }
  }

  /**
   * `body` false skips pose and face and reuses their last result: the body and face move far
   * less than the hands, and hands-only frames cost a fraction of a full one.
   */
  detect(source: TexImageSource | ImageBitmap, timestampMs: number, body = true, face = true): RawFrame {
    const image = this.bodyMode === 'IMAGE';
    if (body) {
      const pose = image ? this.pose.detect(source) : this.pose.detectForVideo(source, timestampMs);
      this.lastBody.pose = pose.landmarks[0]?.map((l): PosePoint => [l.x, l.y, l.z, l.visibility ?? 0]) ?? null;
      if (face) {
        const f = image ? this.face.detect(source) : this.face.detectForVideo(source, timestampMs);
        this.lastBody.face = f.faceLandmarks[0]?.map(p3) ?? null;
      }
    }
    const hands =
      this.handsMode === 'IMAGE' ? this.hands.detect(source) : this.hands.detectForVideo(source, timestampMs);
    return {
      ...this.lastBody,
      hands: hands.landmarks.map((lms, i) => ({
        label: hands.handedness[i]?.[0]?.categoryName ?? 'Right',
        lms: lms.map(p3),
      })),
    };
  }

  close() {
    this.pose.close();
    this.face.close();
    this.hands.close();
  }
}

const p3 = (l: NormalizedLandmark): Point => [l.x, l.y, l.z];
