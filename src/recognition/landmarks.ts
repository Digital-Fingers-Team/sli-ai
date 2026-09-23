import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import type { Point, PosePoint, RawFrame } from './features';

type Delegate = 'GPU' | 'CPU';

// IMAGE mode detects every frame independently, like the training extractor did; VIDEO mode
// tracks between frames. ?mode=video switches to tracking for comparison.
const MODE: 'IMAGE' | 'VIDEO' = new URLSearchParams(location.search).get('mode') === 'video' ? 'VIDEO' : 'IMAGE';

/** Pose + face + hand landmarkers, configured like the training extractor (1 face, 1 hand). */
export class Landmarks {
  private constructor(
    private pose: PoseLandmarker,
    private face: FaceLandmarker,
    private hands: HandLandmarker,
    readonly delegate: Delegate,
  ) {}

  static async load(base: string, onProgress?: (step: string) => void): Promise<Landmarks> {
    const fileset = await FilesetResolver.forVisionTasks(`${base}runtime/mediapipe`);
    const make = async (delegate: Delegate) => {
      const opts = (name: string) => ({
        baseOptions: { modelAssetPath: `${base}mediapipe/${name}_landmarker.task`, delegate },
        runningMode: MODE,
      });
      onProgress?.('pose');
      const pose = await PoseLandmarker.createFromOptions(fileset, opts('pose'));
      onProgress?.('face');
      const face = await FaceLandmarker.createFromOptions(fileset, { ...opts('face'), numFaces: 1 });
      onProgress?.('hand');
      const hands = await HandLandmarker.createFromOptions(fileset, { ...opts('hand'), numHands: 1 });
      return new Landmarks(pose, face, hands, delegate);
    };
    // ?delegate=CPU forces the CPU path (devices whose WebGL is emulated or broken).
    if (new URLSearchParams(location.search).get('delegate')?.toUpperCase() === 'CPU') return make('CPU');
    try {
      return await make('GPU');
    } catch (err) {
      console.warn('GPU delegate unavailable, using CPU', err);
      return make('CPU');
    }
  }

  detect(source: HTMLVideoElement | HTMLCanvasElement | ImageBitmap, timestampMs: number): RawFrame {
    const image = MODE === 'IMAGE';
    const pose = image ? this.pose.detect(source) : this.pose.detectForVideo(source, timestampMs);
    const face = image ? this.face.detect(source) : this.face.detectForVideo(source, timestampMs);
    const hands = image ? this.hands.detect(source) : this.hands.detectForVideo(source, timestampMs);
    const p3 = (l: NormalizedLandmark): Point => [l.x, l.y, l.z];
    return {
      pose: pose.landmarks[0]?.map((l): PosePoint => [l.x, l.y, l.z, l.visibility ?? 0]) ?? null,
      face: face.faceLandmarks[0]?.map(p3) ?? null,
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
