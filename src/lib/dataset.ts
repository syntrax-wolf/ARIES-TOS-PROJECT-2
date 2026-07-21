import { downsampleImage, getRawImage, loadMnistRaw, NUM_CLASSES } from "./mnist";
import { makeRng } from "./rng";

// Fixed dataset size — keeps the experience predictable and focused on the
// hyperparameters, rather than adding a second axis of variation.
export const STANDARD_SAMPLES = 1200;
export const STANDARD_TEST_SIZE = 0.2;

export interface DatasetConfig {
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

export async function generateDataset(config: DatasetConfig): Promise<Dataset> {
  const raw = await loadMnistRaw();
  const rng = makeRng(config.seed);
  const n = Math.min(STANDARD_SAMPLES, raw.count);

  const indices = Array.from({ length: raw.count }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const points: Point[] = indices.slice(0, n).map((idx) => {
    const rawImage = getRawImage(raw, idx);
    return { features: downsampleImage(rawImage), raw: rawImage, label: raw.labels[idx] };
  });

  const testCount = Math.round(n * STANDARD_TEST_SIZE);
  const test = points.slice(0, testCount);
  const train = points.slice(testCount);
  const classes = Array.from({ length: NUM_CLASSES }, (_, i) => i);

  return { train, test, classes };
}
