export interface Guess {
  id: number;
  p: number;
}

export function softmaxTopK(logits: ArrayLike<number>, k: number): Guess[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += exps[i] = Math.exp(logits[i] - max);
  const order = Array.from(exps.keys()).sort((a, b) => exps[b] - exps[a]);
  return order.slice(0, k).map((id) => ({ id, p: exps[id] / sum }));
}
