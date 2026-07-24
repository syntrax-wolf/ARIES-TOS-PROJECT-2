import type { Point } from "./dataset";
import { FEATURE_SIZE, NUM_CLASSES } from "./mnist";

// Inference-only counterpart to scripts/train-cnn.mjs — same tiny
// architecture, same weight shapes, but this module never trains anything.
// It just loads the fitted weights baked at public/cnn/weights.json and runs
// the forward pass, keeping every intermediate grid around so the animation
// can show real activations rather than a decorative stand-in.
//
// Architecture: input 14x14x1 -> pad(1) 16x16x1 -> conv1 3x3x8 -> 14x14x8
// (relu) -> pool1 2x2 -> 7x7x8 -> pad(1) 9x9x8 -> conv2 3x3x16 -> 7x7x16
// (relu) -> pool2 2x2 (floor) -> 3x3x16 -> flatten 144 -> dense1 32 (relu)
// -> dense2 10 -> softmax.

export const PAD = 1;
export const CONV1_FILTERS = 8;
export const CONV1_K = 3;
export const CONV2_FILTERS = 16;
export const CONV2_K = 3;
export const POOL = 2;
export const DENSE1_SIZE = 32;

export interface CnnWeights {
  conv1: { filters: number[][][][]; biases: number[] };
  conv2: { filters: number[][][][]; biases: number[] };
  dense1: { W: number[][]; b: number[] };
  dense2: { W: number[][]; b: number[] };
}

export type Grid = number[][][]; // [H][W][C]

export interface CnnForward {
  input: number[][]; // 14x14, normalized to [0,1]
  padded1: Grid; // 16x16x1, zero-padded input
  conv1: Grid; // 14x14x8, post-ReLU
  pool1: Grid; // 7x7x8
  padded2: Grid; // 9x9x8, zero-padded pool1
  conv2: Grid; // 7x7x16, post-ReLU
  pool2: Grid; // 3x3x16
  dense1: number[]; // 32, post-ReLU
  logits: number[]; // 10
  probs: number[]; // 10
  predicted: number;
}

let cache: Promise<CnnWeights> | null = null;

export function loadCnnWeights(): Promise<CnnWeights> {
  if (!cache) {
    cache = fetch("/cnn/weights.json").then((r) => r.json());
  }
  return cache;
}

function relu(v: number): number {
  return v > 0 ? v : 0;
}

function softmax(z: number[]): number[] {
  const max = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function convForward(x: Grid, filters: number[][][][], biases: number[], K: number): Grid {
  const H = x.length;
  const W = x[0].length;
  const Cin = x[0][0].length;
  const Cout = filters.length;
  const outH = H - K + 1;
  const outW = W - K + 1;
  const out: Grid = Array.from({ length: outH }, () => Array.from({ length: outW }, () => new Array(Cout).fill(0)));
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let co = 0; co < Cout; co++) {
        let sum = biases[co];
        for (let ky = 0; ky < K; ky++) {
          for (let kx = 0; kx < K; kx++) {
            for (let ci = 0; ci < Cin; ci++) sum += x[oh + ky][ow + kx][ci] * filters[co][ky][kx][ci];
          }
        }
        out[oh][ow][co] = sum;
      }
    }
  }
  return out;
}

function reluGrid(x: Grid): Grid {
  return x.map((row) => row.map((cell) => cell.map(relu)));
}

/** Zero-pads an [H][W][C] grid by `p` on every side. */
function padGrid(x: Grid, p: number): Grid {
  const H = x.length;
  const W = x[0].length;
  const C = x[0][0].length;
  const out: Grid = Array.from({ length: H + 2 * p }, () => Array.from({ length: W + 2 * p }, () => new Array(C).fill(0)));
  for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) for (let c = 0; c < C; c++) out[h + p][w + p][c] = x[h][w][c];
  return out;
}

/** Floor (valid) max pool — an odd trailing row/column is left out of the output. */
function maxPool(pre: Grid): Grid {
  const H = pre.length;
  const W = pre[0].length;
  const C = pre[0][0].length;
  const outH = Math.floor(H / POOL);
  const outW = Math.floor(W / POOL);
  const out: Grid = Array.from({ length: outH }, () => Array.from({ length: outW }, () => new Array(C).fill(0)));
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let c = 0; c < C; c++) {
        let best = -Infinity;
        for (let dy = 0; dy < POOL; dy++) {
          for (let dx = 0; dx < POOL; dx++) {
            best = Math.max(best, pre[oh * POOL + dy][ow * POOL + dx][c]);
          }
        }
        out[oh][ow][c] = best;
      }
    }
  }
  return out;
}

function flatten(x: Grid): number[] {
  const H = x.length;
  const W = x[0].length;
  const C = x[0][0].length;
  const out: number[] = new Array(H * W * C);
  let idx = 0;
  for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) for (let c = 0; c < C; c++) out[idx++] = x[h][w][c];
  return out;
}

function denseForward(x: number[], W: number[][], b: number[]): number[] {
  const outSize = b.length;
  const z = new Array(outSize);
  for (let j = 0; j < outSize; j++) {
    let sum = b[j];
    for (let i = 0; i < x.length; i++) sum += x[i] * W[i][j];
    z[j] = sum;
  }
  return z;
}

/** Reshapes a flat 196-length feature vector (0-255 range) into a normalized 14x14 grid. */
export function featuresToGrid(features: Float32Array): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < FEATURE_SIZE; r++) {
    const row: number[] = [];
    for (let c = 0; c < FEATURE_SIZE; c++) row.push(features[r * FEATURE_SIZE + c] / 255);
    grid.push(row);
  }
  return grid;
}

export function forwardCnn(weights: CnnWeights, features: Float32Array): CnnForward {
  const input = featuresToGrid(features);
  const x: Grid = input.map((row) => row.map((v) => [v]));

  const padded1 = padGrid(x, PAD);
  const z1 = convForward(padded1, weights.conv1.filters, weights.conv1.biases, CONV1_K);
  const a1 = reluGrid(z1);
  const pool1 = maxPool(a1);

  const padded2 = padGrid(pool1, PAD);
  const z2 = convForward(padded2, weights.conv2.filters, weights.conv2.biases, CONV2_K);
  const a2 = reluGrid(z2);
  const pool2 = maxPool(a2);

  const flat = flatten(pool2);
  const zd1 = denseForward(flat, weights.dense1.W, weights.dense1.b);
  const ad1 = zd1.map(relu);
  const logits = denseForward(ad1, weights.dense2.W, weights.dense2.b);
  const probs = softmax(logits);

  let best = -Infinity;
  let predicted = 0;
  for (let k = 0; k < NUM_CLASSES; k++) {
    if (probs[k] > best) {
      best = probs[k];
      predicted = k;
    }
  }

  return { input, padded1, conv1: a1, pool1, padded2, conv2: a2, pool2, dense1: ad1, logits, probs, predicted };
}

export function predictCnn(weights: CnnWeights, point: Point): number {
  return forwardCnn(weights, point.features).predicted;
}
