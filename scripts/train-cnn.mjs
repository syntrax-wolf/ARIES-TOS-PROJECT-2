// One-off offline trainer for Sylva's "pre-trained CNN" demo. Run with:
//   node scripts/train-cnn.mjs
// Reads the same real-MNIST binary assets the app ships (public/mnist), trains
// a small real convolutional network with plain hand-rolled backprop (no
// framework), and writes the fitted weights to public/cnn/weights.json. The
// app only ever does forward-pass inference with that file at runtime — this
// script is the one place the CNN is actually trained, which is what makes it
// legitimately "pre-trained" rather than a canned animation.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const RAW_SIZE = 28;
const RAW_PIXELS = RAW_SIZE * RAW_SIZE;
const FEATURE_SIZE = 14;
const NUM_CLASSES = 10;

const CONV1_FILTERS = 4;
const CONV1_K = 3;
const CONV2_FILTERS = 8;
const CONV2_K = 3;
const POOL = 2;
const DENSE1_SIZE = 16;

const EPOCHS = 22;
const LEARNING_RATE = 0.015;
const GRAD_CLIP = 2;

function makeRng(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- data loading -----------------------------------------------------

function readBinFile(path) {
  const buf = readFileSync(path);
  // Node pools small allocations from a shared backing ArrayBuffer — reading
  // .buffer directly (without byteOffset/length) would grab the whole pool.
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

const images = readBinFile(join(ROOT, "public/mnist/images28.bin"));
const labels = readBinFile(join(ROOT, "public/mnist/labels.bin"));
const count = labels.length;

function downsample(image28) {
  const out = [];
  for (let r = 0; r < FEATURE_SIZE; r++) {
    const row = [];
    for (let c = 0; c < FEATURE_SIZE; c++) {
      const r0 = r * 2;
      const c0 = c * 2;
      const sum =
        image28[r0 * RAW_SIZE + c0] +
        image28[r0 * RAW_SIZE + c0 + 1] +
        image28[(r0 + 1) * RAW_SIZE + c0] +
        image28[(r0 + 1) * RAW_SIZE + c0 + 1];
      row.push(sum / 4 / 255); // normalized to [0,1]
    }
    out.push(row);
  }
  return out; // [14][14]
}

const rng = makeRng(20260722);
const indices = Array.from({ length: count }, (_, i) => i);
for (let i = indices.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [indices[i], indices[j]] = [indices[j], indices[i]];
}

const samples = indices.map((idx) => ({
  grid: downsample(images.subarray(idx * RAW_PIXELS, (idx + 1) * RAW_PIXELS)),
  label: labels[idx],
}));

const testCount = Math.round(samples.length * 0.15);
const testSamples = samples.slice(0, testCount);
const trainSamples = samples.slice(testCount);
console.log(`train=${trainSamples.length} test=${testSamples.length}`);

// ---- parameter init ----------------------------------------------------

function make4d(a, b, c, d, scale, rng) {
  return Array.from({ length: a }, () =>
    Array.from({ length: b }, () => Array.from({ length: c }, () => Array.from({ length: d }, () => (rng() * 2 - 1) * scale)))
  );
}
function make2d(a, b, scale, rng) {
  return Array.from({ length: a }, () => Array.from({ length: b }, () => (rng() * 2 - 1) * scale));
}

const initRng = makeRng(7);
const params = {
  conv1: {
    filters: make4d(CONV1_FILTERS, CONV1_K, CONV1_K, 1, Math.sqrt(1 / (CONV1_K * CONV1_K * 1)), initRng),
    biases: new Array(CONV1_FILTERS).fill(0),
  },
  conv2: {
    filters: make4d(CONV2_FILTERS, CONV2_K, CONV2_K, CONV1_FILTERS, Math.sqrt(1 / (CONV2_K * CONV2_K * CONV1_FILTERS)), initRng),
    biases: new Array(CONV2_FILTERS).fill(0),
  },
  dense1: { W: make2d(2 * 2 * CONV2_FILTERS, DENSE1_SIZE, Math.sqrt(1 / (2 * 2 * CONV2_FILTERS)), initRng), b: new Array(DENSE1_SIZE).fill(0) },
  dense2: { W: make2d(DENSE1_SIZE, NUM_CLASSES, Math.sqrt(1 / DENSE1_SIZE), initRng), b: new Array(NUM_CLASSES).fill(0) },
};

// ---- ops ----------------------------------------------------------------

function relu(v) {
  return v > 0 ? v : 0;
}
function reluGrad(v) {
  return v > 0 ? 1 : 0;
}
function softmax(z) {
  const max = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
function zeros3d(h, w, c) {
  return Array.from({ length: h }, () => Array.from({ length: w }, () => new Array(c).fill(0)));
}
function zeros4d(a, b, c, d) {
  return Array.from({ length: a }, () => Array.from({ length: b }, () => Array.from({ length: c }, () => new Array(d).fill(0))));
}

// x: [H][W][Cin] -> z: [H-K+1][W-K+1][Cout]
function convForward(x, filters, biases, K) {
  const H = x.length,
    W = x[0].length,
    Cin = x[0][0].length,
    Cout = filters.length;
  const outH = H - K + 1,
    outW = W - K + 1;
  const z = zeros3d(outH, outW, Cout);
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let co = 0; co < Cout; co++) {
        let sum = biases[co];
        for (let ky = 0; ky < K; ky++) {
          for (let kx = 0; kx < K; kx++) {
            for (let ci = 0; ci < Cin; ci++) {
              sum += x[oh + ky][ow + kx][ci] * filters[co][ky][kx][ci];
            }
          }
        }
        z[oh][ow][co] = sum;
      }
    }
  }
  return z;
}

function map3d(x, f) {
  return x.map((row) => row.map((cell) => cell.map(f)));
}

// pre: [H][W][C] pre-pool activations -> {out:[H/2][W/2][C], argmax:[H/2][W/2][C] -> 0..3}
function maxPoolForward(pre) {
  const H = pre.length,
    W = pre[0].length,
    C = pre[0][0].length;
  const outH = H / POOL,
    outW = W / POOL;
  const out = zeros3d(outH, outW, C);
  const argmax = zeros3d(outH, outW, C);
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let c = 0; c < C; c++) {
        let best = -Infinity,
          bestIdx = 0;
        for (let p = 0; p < POOL * POOL; p++) {
          const dy = Math.floor(p / POOL),
            dx = p % POOL;
          const v = pre[oh * POOL + dy][ow * POOL + dx][c];
          if (v > best) {
            best = v;
            bestIdx = p;
          }
        }
        out[oh][ow][c] = best;
        argmax[oh][ow][c] = bestIdx;
      }
    }
  }
  return { out, argmax };
}

function flatten(x) {
  const H = x.length,
    W = x[0].length,
    C = x[0][0].length;
  const out = new Array(H * W * C);
  let idx = 0;
  for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) for (let c = 0; c < C; c++) out[idx++] = x[h][w][c];
  return out;
}
function unflatten(vec, H, W, C) {
  const out = zeros3d(H, W, C);
  let idx = 0;
  for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) for (let c = 0; c < C; c++) out[h][w][c] = vec[idx++];
  return out;
}

function denseForward(x, W, b) {
  const outSize = b.length;
  const z = new Array(outSize);
  for (let j = 0; j < outSize; j++) {
    let sum = b[j];
    for (let i = 0; i < x.length; i++) sum += x[i] * W[i][j];
    z[j] = sum;
  }
  return z;
}

function clip(v) {
  return v > GRAD_CLIP ? GRAD_CLIP : v < -GRAD_CLIP ? -GRAD_CLIP : v;
}

// ---- forward + backward for a single sample -----------------------------

function forward(grid) {
  const x = grid.map((row) => row.map((v) => [v])); // [14][14][1]
  const z1 = convForward(x, params.conv1.filters, params.conv1.biases, CONV1_K); // [12][12][4]
  const a1 = map3d(z1, relu);
  const { out: p1, argmax: argmax1 } = maxPoolForward(a1); // [6][6][4]
  const z2 = convForward(p1, params.conv2.filters, params.conv2.biases, CONV2_K); // [4][4][8]
  const a2 = map3d(z2, relu);
  const { out: p2, argmax: argmax2 } = maxPoolForward(a2); // [2][2][8]
  const flat = flatten(p2); // 32
  const zd1 = denseForward(flat, params.dense1.W, params.dense1.b);
  const ad1 = zd1.map(relu);
  const logits = denseForward(ad1, params.dense2.W, params.dense2.b);
  const probs = softmax(logits);
  return { x, z1, a1, p1, argmax1, z2, a2, p2, argmax2, flat, zd1, ad1, logits, probs };
}

function trainOne(grid, label) {
  const fwd = forward(grid);
  const grads = {
    conv1: { filters: zeros4d(CONV1_FILTERS, CONV1_K, CONV1_K, 1), biases: new Array(CONV1_FILTERS).fill(0) },
    conv2: { filters: zeros4d(CONV2_FILTERS, CONV2_K, CONV2_K, CONV1_FILTERS), biases: new Array(CONV2_FILTERS).fill(0) },
    dense1: { W: make2d(2 * 2 * CONV2_FILTERS, DENSE1_SIZE, 0, () => 0), b: new Array(DENSE1_SIZE).fill(0) },
    dense2: { W: make2d(DENSE1_SIZE, NUM_CLASSES, 0, () => 0), b: new Array(NUM_CLASSES).fill(0) },
  };

  const dLogits = fwd.probs.slice();
  dLogits[label] -= 1;

  // dense2
  for (let j = 0; j < NUM_CLASSES; j++) {
    grads.dense2.b[j] = dLogits[j];
    for (let i = 0; i < DENSE1_SIZE; i++) grads.dense2.W[i][j] = fwd.ad1[i] * dLogits[j];
  }
  const dAd1 = new Array(DENSE1_SIZE).fill(0);
  for (let i = 0; i < DENSE1_SIZE; i++) {
    let s = 0;
    for (let j = 0; j < NUM_CLASSES; j++) s += params.dense2.W[i][j] * dLogits[j];
    dAd1[i] = s;
  }
  const dZd1 = dAd1.map((v, i) => v * reluGrad(fwd.zd1[i]));

  // dense1
  const flatSize = fwd.flat.length;
  for (let j = 0; j < DENSE1_SIZE; j++) {
    grads.dense1.b[j] = dZd1[j];
    for (let i = 0; i < flatSize; i++) grads.dense1.W[i][j] = fwd.flat[i] * dZd1[j];
  }
  const dFlat = new Array(flatSize).fill(0);
  for (let i = 0; i < flatSize; i++) {
    let s = 0;
    for (let j = 0; j < DENSE1_SIZE; j++) s += params.dense1.W[i][j] * dZd1[j];
    dFlat[i] = s;
  }

  // unflatten into pool2 gradient, route through maxpool2 argmax into a2 gradient
  const dP2 = unflatten(dFlat, 2, 2, CONV2_FILTERS);
  const dA2 = zeros3d(4, 4, CONV2_FILTERS);
  for (let oh = 0; oh < 2; oh++) {
    for (let ow = 0; ow < 2; ow++) {
      for (let c = 0; c < CONV2_FILTERS; c++) {
        const p = fwd.argmax2[oh][ow][c];
        const dy = Math.floor(p / POOL),
          dx = p % POOL;
        dA2[oh * POOL + dy][ow * POOL + dx][c] += dP2[oh][ow][c];
      }
    }
  }
  const dZ2 = map3d(dA2, () => 0);
  for (let h = 0; h < 4; h++) for (let w = 0; w < 4; w++) for (let c = 0; c < CONV2_FILTERS; c++) dZ2[h][w][c] = dA2[h][w][c] * reluGrad(fwd.z2[h][w][c]);

  // conv2 grads + gradient wrt its input (pool1 output, 6x6x4)
  const dP1 = zeros3d(6, 6, CONV1_FILTERS);
  for (let oh = 0; oh < 4; oh++) {
    for (let ow = 0; ow < 4; ow++) {
      for (let co = 0; co < CONV2_FILTERS; co++) {
        const g = dZ2[oh][ow][co];
        grads.conv2.biases[co] += g;
        for (let ky = 0; ky < CONV2_K; ky++) {
          for (let kx = 0; kx < CONV2_K; kx++) {
            for (let ci = 0; ci < CONV1_FILTERS; ci++) {
              grads.conv2.filters[co][ky][kx][ci] += fwd.p1[oh + ky][ow + kx][ci] * g;
              dP1[oh + ky][ow + kx][ci] += params.conv2.filters[co][ky][kx][ci] * g;
            }
          }
        }
      }
    }
  }

  // route through maxpool1 argmax into a1 gradient
  const dA1 = zeros3d(12, 12, CONV1_FILTERS);
  for (let oh = 0; oh < 6; oh++) {
    for (let ow = 0; ow < 6; ow++) {
      for (let c = 0; c < CONV1_FILTERS; c++) {
        const p = fwd.argmax1[oh][ow][c];
        const dy = Math.floor(p / POOL),
          dx = p % POOL;
        dA1[oh * POOL + dy][ow * POOL + dx][c] += dP1[oh][ow][c];
      }
    }
  }
  const dZ1 = zeros3d(12, 12, CONV1_FILTERS);
  for (let h = 0; h < 12; h++) for (let w = 0; w < 12; w++) for (let c = 0; c < CONV1_FILTERS; c++) dZ1[h][w][c] = dA1[h][w][c] * reluGrad(fwd.z1[h][w][c]);

  // conv1 grads (input x is not trainable, so no need to propagate further)
  for (let oh = 0; oh < 12; oh++) {
    for (let ow = 0; ow < 12; ow++) {
      for (let co = 0; co < CONV1_FILTERS; co++) {
        const g = dZ1[oh][ow][co];
        grads.conv1.biases[co] += g;
        for (let ky = 0; ky < CONV1_K; ky++) {
          for (let kx = 0; kx < CONV1_K; kx++) {
            grads.conv1.filters[co][ky][kx][0] += fwd.x[oh + ky][ow + kx][0] * g;
          }
        }
      }
    }
  }

  return { grads, predicted: fwd.probs.indexOf(Math.max(...fwd.probs)) };
}

function applyGrads(grads, lr) {
  for (let co = 0; co < CONV1_FILTERS; co++) {
    params.conv1.biases[co] -= lr * clip(grads.conv1.biases[co]);
    for (let ky = 0; ky < CONV1_K; ky++) for (let kx = 0; kx < CONV1_K; kx++) params.conv1.filters[co][ky][kx][0] -= lr * clip(grads.conv1.filters[co][ky][kx][0]);
  }
  for (let co = 0; co < CONV2_FILTERS; co++) {
    params.conv2.biases[co] -= lr * clip(grads.conv2.biases[co]);
    for (let ky = 0; ky < CONV2_K; ky++)
      for (let kx = 0; kx < CONV2_K; kx++) for (let ci = 0; ci < CONV1_FILTERS; ci++) params.conv2.filters[co][ky][kx][ci] -= lr * clip(grads.conv2.filters[co][ky][kx][ci]);
  }
  for (let i = 0; i < params.dense1.W.length; i++) for (let j = 0; j < DENSE1_SIZE; j++) params.dense1.W[i][j] -= lr * clip(grads.dense1.W[i][j]);
  for (let j = 0; j < DENSE1_SIZE; j++) params.dense1.b[j] -= lr * clip(grads.dense1.b[j]);
  for (let i = 0; i < DENSE1_SIZE; i++) for (let j = 0; j < NUM_CLASSES; j++) params.dense2.W[i][j] -= lr * clip(grads.dense2.W[i][j]);
  for (let j = 0; j < NUM_CLASSES; j++) params.dense2.b[j] -= lr * clip(grads.dense2.b[j]);
}

function evaluate(set) {
  let correct = 0;
  for (const s of set) {
    const { probs } = forward(s.grid);
    const predicted = probs.indexOf(Math.max(...probs));
    if (predicted === s.label) correct += 1;
  }
  return correct / set.length;
}

// ---- training loop --------------------------------------------------------

const order = trainSamples.map((_, i) => i);
const shuffleRng = makeRng(99);
const start = Date.now();
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(shuffleRng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  for (const idx of order) {
    const s = trainSamples[idx];
    const { grads } = trainOne(s.grid, s.label);
    applyGrads(grads, LEARNING_RATE);
  }
  if (epoch % 4 === 0 || epoch === EPOCHS) {
    const acc = evaluate(testSamples);
    console.log(`epoch ${epoch}: test accuracy ${(acc * 100).toFixed(1)}% (${((Date.now() - start) / 1000).toFixed(1)}s)`);
  }
}

const outDir = join(ROOT, "public/cnn");
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, "weights.json"), JSON.stringify(params));
console.log("wrote public/cnn/weights.json");
