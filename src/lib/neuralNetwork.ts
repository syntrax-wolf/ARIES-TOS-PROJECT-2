import type { Point } from "./dataset";
import { NUM_CLASSES } from "./mnist";
import { makeRng } from "./rng";

export interface NetworkHyperparams {
  hiddenLayers: number; // 1-3, kept small on purpose
  nodesPerLayer: number; // width shared by every hidden layer
}

export interface Layer {
  W: number[][]; // W[i][j] = weight from input i to output j
  b: number[];
}

export interface NetworkSnapshot {
  layers: Layer[];
}

export interface TrainedNetwork {
  snapshots: NetworkSnapshot[]; // snapshots[0] = freshly initialized weights, last = fully trained
  layerSizes: number[]; // [inputSize, ...hidden, outputSize]
}

const EPOCHS = 70;
const LEARNING_RATE = 0.04;
const GRAD_CLIP = 4;
const NUM_SNAPSHOTS = 7;

function relu(x: number) {
  return x > 0 ? x : 0;
}

function softmax(z: number[]): number[] {
  const max = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function makeLayer(inSize: number, outSize: number, rng: () => number): Layer {
  const scale = Math.sqrt(1 / inSize);
  const W = Array.from({ length: inSize }, () => Array.from({ length: outSize }, () => (rng() * 2 - 1) * scale));
  const b = new Array(outSize).fill(0);
  return { W, b };
}

function cloneLayers(layers: Layer[]): Layer[] {
  return layers.map((l) => ({ W: l.W.map((row) => row.slice()), b: l.b.slice() }));
}

export function normalizeFeatures(features: Float32Array | Uint8Array): number[] {
  return Array.from(features, (v) => v / 255);
}

interface ForwardResult {
  activations: number[][]; // activations[0] = input, activations[last] = softmax output
  zs: number[][]; // pre-activations, one per weight layer
}

function forward(layers: Layer[], x: number[]): ForwardResult {
  const activations: number[][] = [x];
  const zs: number[][] = [];
  for (let l = 0; l < layers.length; l++) {
    const { W, b } = layers[l];
    const outSize = b.length;
    const prev = activations[l];
    const z = new Array(outSize);
    for (let j = 0; j < outSize; j++) {
      let sum = b[j];
      for (let i = 0; i < prev.length; i++) sum += prev[i] * W[i][j];
      z[j] = sum;
    }
    zs.push(z);
    const isOutput = l === layers.length - 1;
    activations.push(isOutput ? softmax(z) : z.map(relu));
  }
  return { activations, zs };
}

function backward(layers: Layer[], activations: number[][], zs: number[][], label: number) {
  const numLayers = layers.length;
  let delta = activations[numLayers].slice();
  delta[label] -= 1; // softmax + cross-entropy gradient simplifies to (prediction - target)

  const grads: { gW: number[][]; gb: number[] }[] = new Array(numLayers);

  for (let l = numLayers - 1; l >= 0; l--) {
    const inputActs = activations[l];
    const inSize = inputActs.length;
    const outSize = layers[l].b.length;
    const gW: number[][] = Array.from({ length: inSize }, () => new Array(outSize).fill(0));
    const gb: number[] = new Array(outSize).fill(0);

    for (let j = 0; j < outSize; j++) {
      gb[j] = delta[j];
      for (let i = 0; i < inSize; i++) gW[i][j] = inputActs[i] * delta[j];
    }
    grads[l] = { gW, gb };

    if (l > 0) {
      const newDelta = new Array(inSize).fill(0);
      for (let i = 0; i < inSize; i++) {
        let s = 0;
        for (let j = 0; j < outSize; j++) s += layers[l].W[i][j] * delta[j];
        newDelta[i] = s * (zs[l - 1][i] > 0 ? 1 : 0); // ReLU derivative
      }
      delta = newDelta;
    }
  }
  return grads;
}

function clip(v: number): number {
  return v > GRAD_CLIP ? GRAD_CLIP : v < -GRAD_CLIP ? -GRAD_CLIP : v;
}

function applyGrads(layers: Layer[], grads: { gW: number[][]; gb: number[] }[], lr: number) {
  for (let l = 0; l < layers.length; l++) {
    const { W, b } = layers[l];
    const { gW, gb } = grads[l];
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) W[i][j] -= lr * clip(gW[i][j]);
    }
    for (let j = 0; j < b.length; j++) b[j] -= lr * clip(gb[j]);
  }
}

/**
 * A handful of small hidden layers trained with plain per-sample SGD backprop.
 * Weight snapshots are captured at evenly spaced epochs so the animation can
 * show real training progression rather than a purely decorative pulse.
 */
export function trainNeuralNetwork(points: Point[], params: NetworkHyperparams, seed: number): TrainedNetwork {
  const rng = makeRng(seed);
  const inputSize = points[0]?.features.length ?? 0;
  const sizes = [inputSize, ...Array(params.hiddenLayers).fill(params.nodesPerLayer), NUM_CLASSES];

  const layers: Layer[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    layers.push(makeLayer(sizes[l], sizes[l + 1], rng));
  }

  const trainX = points.map((p) => normalizeFeatures(p.features));
  const trainY = points.map((p) => p.label);

  const snapshots: NetworkSnapshot[] = [{ layers: cloneLayers(layers) }];
  const snapshotEpochs = new Set<number>();
  for (let i = 1; i < NUM_SNAPSHOTS; i++) {
    snapshotEpochs.add(Math.round((i / (NUM_SNAPSHOTS - 1)) * EPOCHS));
  }

  const order = points.map((_, i) => i);
  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const idx of order) {
      const { activations, zs } = forward(layers, trainX[idx]);
      const grads = backward(layers, activations, zs, trainY[idx]);
      applyGrads(layers, grads, LEARNING_RATE);
    }
    if (snapshotEpochs.has(epoch)) snapshots.push({ layers: cloneLayers(layers) });
  }
  while (snapshots.length < NUM_SNAPSHOTS) snapshots.push({ layers: cloneLayers(layers) });

  return { snapshots, layerSizes: sizes };
}

export function predict(snapshot: NetworkSnapshot, point: Point): number {
  const x = normalizeFeatures(point.features);
  const { activations } = forward(snapshot.layers, x);
  const output = activations[activations.length - 1];
  let best = -Infinity;
  let label = 0;
  for (let i = 0; i < output.length; i++) {
    if (output[i] > best) {
      best = output[i];
      label = i;
    }
  }
  return label;
}
