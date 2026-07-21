import { downsampleImage, getRawImage, loadMnistRaw, NUM_CLASSES } from "./mnist";

export interface DatasetConfig {
  samples: number; // total subset size (train + test), drawn from the bundled MNIST pool
  testSize: number; // 0.1 - 0.5
  seed: number;
}

export interface Point {
  features: Float32Array; // 14x14 downsampled pixel intensities, the tree's actual training input
  raw: Uint8Array; // original 28x28 pixels, for display/thumbnails only
  label: number; // digit 0-9
}

export interface Dataset {
  train: Point[];
  test: Point[];
  classes: number[];
}

function makeRng(seed: number) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export async function generateDataset(config: DatasetConfig): Promise<Dataset> {
  const raw = await loadMnistRaw();
  const rng = makeRng(config.seed);
  const n = Math.min(config.samples, raw.count);

  const indices = Array.from({ length: raw.count }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const points: Point[] = indices.slice(0, n).map((idx) => {
    const rawImage = getRawImage(raw, idx);
    return { features: downsampleImage(rawImage), raw: rawImage, label: raw.labels[idx] };
  });

  const testCount = Math.round(n * config.testSize);
  const test = points.slice(0, testCount);
  const train = points.slice(testCount);
  const classes = Array.from({ length: NUM_CLASSES }, (_, i) => i);

  return { train, test, classes };
}
