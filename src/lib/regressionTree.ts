// A small regression CART — the "weak learner" gradient boosting repeatedly
// fits to the current residual. Splits minimize sum-of-squared-error rather
// than gini, but the sweep technique is the same single-sorted-pass trick
// used in decisionTree.ts.

export interface RegressionSample {
  features: Float32Array;
  target: number;
}

const MIN_SAMPLES_LEAF = 5;
const MIN_SAMPLES_SPLIT = 11;

export interface RegressionNode {
  depth: number;
  mean: number;
  isLeaf: boolean;
  featureIndex?: number;
  threshold?: number;
  left?: RegressionNode;
  right?: RegressionNode;
  sampleCount: number;
  /** Raw SSE decrease achieved by this node's split — feeds feature-importance weighting. */
  splitGain?: number;
}

export interface RegressionTree {
  root: RegressionNode;
  maxDepth: number;
}

function meanOf(samples: RegressionSample[]): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const s of samples) sum += s.target;
  return sum / samples.length;
}

function sse(sum: number, sumSq: number, n: number): number {
  if (n === 0) return 0;
  return sumSq - (sum * sum) / n;
}

interface SplitCandidate {
  featureIndex: number;
  threshold: number;
  sseDecrease: number;
}

function bestSplitForFeature(samples: RegressionSample[], featureIndex: number, parentSse: number): SplitCandidate | null {
  const n = samples.length;
  const entries = new Array<{ v: number; t: number }>(n);
  for (let i = 0; i < n; i++) entries[i] = { v: samples[i].features[featureIndex], t: samples[i].target };
  entries.sort((a, b) => a.v - b.v);

  let totalSum = 0;
  let totalSumSq = 0;
  for (const e of entries) {
    totalSum += e.t;
    totalSumSq += e.t * e.t;
  }

  let leftSum = 0;
  let leftSumSq = 0;
  let best: SplitCandidate | null = null;

  for (let i = 0; i < n - 1; i++) {
    const e = entries[i];
    leftSum += e.t;
    leftSumSq += e.t * e.t;
    const leftCount = i + 1;
    const nextV = entries[i + 1].v;
    if (nextV === e.v) continue;

    const rightCount = n - leftCount;
    if (leftCount < MIN_SAMPLES_LEAF || rightCount < MIN_SAMPLES_LEAF) continue;

    const rightSum = totalSum - leftSum;
    const rightSumSq = totalSumSq - leftSumSq;
    const weightedSse = sse(leftSum, leftSumSq, leftCount) + sse(rightSum, rightSumSq, rightCount);
    const sseDecrease = parentSse - weightedSse;

    if (!best || sseDecrease > best.sseDecrease) {
      best = { featureIndex, threshold: (e.v + nextV) / 2, sseDecrease };
    }
  }
  return best;
}

function findBestSplit(samples: RegressionSample[], parentSse: number, numFeatures: number): SplitCandidate | null {
  let best: SplitCandidate | null = null;
  for (let f = 0; f < numFeatures; f++) {
    const candidate = bestSplitForFeature(samples, f, parentSse);
    if (candidate && (best === null || candidate.sseDecrease > best.sseDecrease)) best = candidate;
  }
  return best;
}

function partition(samples: RegressionSample[], featureIndex: number, threshold: number): [RegressionSample[], RegressionSample[]] {
  const left: RegressionSample[] = [];
  const right: RegressionSample[] = [];
  for (const s of samples) (s.features[featureIndex] <= threshold ? left : right).push(s);
  return [left, right];
}

export function trainRegressionTree(samples: RegressionSample[], maxDepth: number): RegressionTree {
  const numFeatures = samples[0]?.features.length ?? 0;
  let observedDepth = 0;

  function build(samples: RegressionSample[], depth: number): RegressionNode {
    observedDepth = Math.max(observedDepth, depth);
    const mean = meanOf(samples);
    const node: RegressionNode = { depth, mean, isLeaf: true, sampleCount: samples.length };

    if (depth >= maxDepth || samples.length < MIN_SAMPLES_SPLIT) return node;

    let sum = 0;
    let sumSq = 0;
    for (const s of samples) {
      sum += s.target;
      sumSq += s.target * s.target;
    }
    const parentSse = sse(sum, sumSq, samples.length);
    const split = findBestSplit(samples, parentSse, numFeatures);
    if (!split || split.sseDecrease <= 1e-9) return node;

    const [leftSamples, rightSamples] = partition(samples, split.featureIndex, split.threshold);
    node.isLeaf = false;
    node.featureIndex = split.featureIndex;
    node.threshold = split.threshold;
    node.splitGain = split.sseDecrease;
    node.left = build(leftSamples, depth + 1);
    node.right = build(rightSamples, depth + 1);
    return node;
  }

  const root = build(samples, 0);
  return { root, maxDepth: observedDepth };
}

export function predictRegression(root: RegressionNode, features: Float32Array): number {
  let node = root;
  while (!node.isLeaf && node.left && node.right) {
    const v = features[node.featureIndex as number];
    node = v <= (node.threshold as number) ? node.left : node.right;
  }
  return node.mean;
}

/** Level-order traversal — used to sequence the weak-learner "sapling" growth animation. */
export function regressionBfsOrder(root: RegressionNode): RegressionNode[] {
  const order: RegressionNode[] = [];
  let queue: RegressionNode[] = [root];
  while (queue.length) {
    const next: RegressionNode[] = [];
    for (const n of queue) {
      order.push(n);
      if (n.left) next.push(n.left);
      if (n.right) next.push(n.right);
    }
    queue = next;
  }
  return order;
}
