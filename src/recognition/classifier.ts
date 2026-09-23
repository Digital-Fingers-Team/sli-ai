import * as ort from 'onnxruntime-web/wasm';
import { FRAME_SIZE, SEQ_LEN } from './features';
import { softmaxTopK, type Guess } from './topk';

export type { Guess } from './topk';

export class SignClassifier {
  private constructor(private session: ort.InferenceSession) {}

  static async load(base: string): Promise<SignClassifier> {
    // The wasm binary is bundled by Vite (dist/assets), so no wasmPaths override is needed.
    // Threads need cross-origin isolation, which GitHub Pages cannot provide.
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(`${base}models/karsl502.onnx`, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new SignClassifier(session);
  }

  /** `sequence` is SEQ_LEN * FRAME_SIZE features as produced by tsnSample. */
  async classify(sequence: Float32Array, k = 5): Promise<Guess[]> {
    const input = new ort.Tensor('float32', sequence, [1, SEQ_LEN, FRAME_SIZE]);
    const out = await this.session.run({ input });
    return softmaxTopK(out.output.data as Float32Array, k);
  }
}
