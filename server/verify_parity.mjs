/**
 * Prints the browser pipeline's ground truth for a seed, so dataset.py can be
 * checked against it. Run:  node server/verify_parity.mjs 42
 *
 * Deliberately re-implements nothing: it imports the same source the app uses.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

// --- copied call-sites, not logic: these mirror what generateDataset() does ---
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

const RAW_SIZE = 28;
const RAW_PIXELS = RAW_SIZE * RAW_SIZE;
const FEATURE_SIZE = 14;
const FEATURE_PIXELS = FEATURE_SIZE * FEATURE_SIZE;
const STANDARD_SAMPLES = 1200;
const STANDARD_TEST_SIZE = 0.2;

function downsampleImage(image28) {
  const out = new Float32Array(FEATURE_PIXELS);
  for (let r = 0; r < FEATURE_SIZE; r++) {
    for (let c = 0; c < FEATURE_SIZE; c++) {
      const r0 = r * 2;
      const c0 = c * 2;
      const sum =
        image28[r0 * RAW_SIZE + c0] +
        image28[r0 * RAW_SIZE + c0 + 1] +
        image28[(r0 + 1) * RAW_SIZE + c0] +
        image28[(r0 + 1) * RAW_SIZE + c0 + 1];
      out[r * FEATURE_SIZE + c] = sum / 4;
    }
  }
  return out;
}

const seed = Number(process.argv[2] ?? 42);

const images = new Uint8Array(readFileSync(join(here, "../public/mnist/images28.bin")));
const labels = new Uint8Array(readFileSync(join(here, "../public/mnist/labels.bin")));
const count = labels.length;

const rng = makeRng(seed);
const indices = Array.from({ length: count }, (_, i) => i);
for (let i = indices.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [indices[i], indices[j]] = [indices[j], indices[i]];
}

const n = Math.min(STANDARD_SAMPLES, count);
const picked = indices.slice(0, n);
const testCount = Math.round(n * STANDARD_TEST_SIZE);

const firstTrainIdx = picked[testCount];
const firstTrainFeatures = downsampleImage(images.subarray(firstTrainIdx * RAW_PIXELS, (firstTrainIdx + 1) * RAW_PIXELS));

let featureSum = 0;
for (const idx of picked) {
  const f = downsampleImage(images.subarray(idx * RAW_PIXELS, (idx + 1) * RAW_PIXELS));
  for (const v of f) featureSum += v;
}

console.log(JSON.stringify({
  seed,
  count,
  rng_first_5: Array.from({ length: 5 }, () => makeRng(seed)()).slice(0, 1),
  first_10_indices: picked.slice(0, 10),
  test_count: testCount,
  train_count: n - testCount,
  first_test_labels: Array.from(picked.slice(0, 10)).map((i) => labels[i]),
  first_train_label: labels[firstTrainIdx],
  first_train_feature_head: Array.from(firstTrainFeatures.slice(0, 6)),
  feature_sum: Math.round(featureSum * 1000) / 1000,
}, null, 2));
