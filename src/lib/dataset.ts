export type DatasetShape = "moons" | "circles" | "blobs" | "xor" | "linear";

export interface DatasetConfig {
  shape: DatasetShape;
  noise: number; // 0 - 0.5
  samples: number; // 100 - 2000
  testSize: number; // 0.1 - 0.5
  seed: number;
}

export interface Point {
  x: number;
  y: number;
  label: number;
}

export interface Dataset {
  train: Point[];
  test: Point[];
  classes: number[];
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
}

// Deterministic PRNG (mulberry32) so a given seed always reproduces the same dataset.
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

function gaussian(rng: () => number) {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeMoons(n: number, noise: number, rng: () => number): Point[] {
  const pts: Point[] = [];
  const nOut = Math.floor(n / 2);
  const nIn = n - nOut;
  for (let i = 0; i < nOut; i++) {
    const t = (i / nOut) * Math.PI;
    const x = Math.cos(t);
    const y = Math.sin(t);
    pts.push({ x: x + gaussian(rng) * noise, y: y + gaussian(rng) * noise, label: 0 });
  }
  for (let i = 0; i < nIn; i++) {
    const t = (i / nIn) * Math.PI;
    const x = 1 - Math.cos(t);
    const y = 1 - Math.sin(t) - 0.5;
    pts.push({ x: x + gaussian(rng) * noise, y: y + gaussian(rng) * noise, label: 1 });
  }
  return pts;
}

function makeCircles(n: number, noise: number, rng: () => number): Point[] {
  const pts: Point[] = [];
  const nOut = Math.floor(n / 2);
  const nIn = n - nOut;
  for (let i = 0; i < nOut; i++) {
    const t = (i / nOut) * Math.PI * 2;
    pts.push({
      x: Math.cos(t) + gaussian(rng) * noise * 0.5,
      y: Math.sin(t) + gaussian(rng) * noise * 0.5,
      label: 0,
    });
  }
  for (let i = 0; i < nIn; i++) {
    const t = (i / nIn) * Math.PI * 2;
    pts.push({
      x: 0.5 * Math.cos(t) + gaussian(rng) * noise * 0.5,
      y: 0.5 * Math.sin(t) + gaussian(rng) * noise * 0.5,
      label: 1,
    });
  }
  return pts;
}

function makeBlobs(n: number, noise: number, rng: () => number): Point[] {
  const centers = [
    { x: -0.9, y: -0.6, label: 0 },
    { x: 0.9, y: -0.5, label: 1 },
    { x: 0.0, y: 0.9, label: 2 },
  ];
  const pts: Point[] = [];
  const spread = 0.35 + noise * 0.9;
  for (let i = 0; i < n; i++) {
    const c = centers[i % centers.length];
    pts.push({
      x: c.x + gaussian(rng) * spread,
      y: c.y + gaussian(rng) * spread,
      label: c.label,
    });
  }
  return pts;
}

function makeXor(n: number, noise: number, rng: () => number): Point[] {
  const pts: Point[] = [];
  const spread = 0.28 + noise * 0.7;
  for (let i = 0; i < n; i++) {
    const qx = rng() < 0.5 ? -1 : 1;
    const qy = rng() < 0.5 ? -1 : 1;
    const label = qx * qy > 0 ? 0 : 1;
    pts.push({
      x: qx * 0.6 + gaussian(rng) * spread,
      y: qy * 0.6 + gaussian(rng) * spread,
      label,
    });
  }
  return pts;
}

function makeLinear(n: number, noise: number, rng: () => number): Point[] {
  const pts: Point[] = [];
  const spread = 0.4 + noise * 1.4;
  for (let i = 0; i < n; i++) {
    const x = (rng() - 0.5) * 2.4;
    const y = (rng() - 0.5) * 2.4;
    const margin = x - y; // separating line: x = y
    const label = margin + gaussian(rng) * spread * 0.3 > 0 ? 0 : 1;
    pts.push({ x, y, label });
  }
  return pts;
}

export function generateDataset(config: DatasetConfig): Dataset {
  const rng = makeRng(config.seed);
  let raw: Point[];
  switch (config.shape) {
    case "moons":
      raw = makeMoons(config.samples, config.noise, rng);
      break;
    case "circles":
      raw = makeCircles(config.samples, config.noise, rng);
      break;
    case "blobs":
      raw = makeBlobs(config.samples, config.noise, rng);
      break;
    case "xor":
      raw = makeXor(config.samples, config.noise, rng);
      break;
    case "linear":
      raw = makeLinear(config.samples, config.noise, rng);
      break;
  }

  // shuffle (Fisher-Yates) using the same seeded rng for reproducibility
  for (let i = raw.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [raw[i], raw[j]] = [raw[j], raw[i]];
  }

  const testCount = Math.round(raw.length * config.testSize);
  const test = raw.slice(0, testCount);
  const train = raw.slice(testCount);

  const xs = raw.map((p) => p.x);
  const ys = raw.map((p) => p.y);
  const pad = 0.4;
  const bounds = {
    xMin: Math.min(...xs) - pad,
    xMax: Math.max(...xs) + pad,
    yMin: Math.min(...ys) - pad,
    yMax: Math.max(...ys) + pad,
  };

  const classes = Array.from(new Set(raw.map((p) => p.label))).sort();

  return { train, test, classes, bounds };
}
