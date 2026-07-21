import type { Point } from "./dataset";
import { featureImportance, predict, trainDecisionTree, type TrainedTree } from "./decisionTree";
import { makeRng } from "./rng";

export interface ForestHyperparams {
  maxDepth: number;
  numTrees: number;
}

export interface TrainedForest {
  trees: TrainedTree[];
  numFeatures: number;
}

function sampleWithReplacement(points: Point[], n: number, rng: () => number): Point[] {
  const out = new Array<Point>(n);
  for (let i = 0; i < n; i++) {
    out[i] = points[Math.floor(rng() * points.length)];
  }
  return out;
}

/**
 * Each tree trains on its own bootstrap resample of the data and, at every
 * split, only gets to consider a random subset of features (~sqrt of the
 * total) — the two sources of randomness that make a random forest more than
 * just N copies of the same decision tree.
 */
export function trainRandomForest(points: Point[], params: ForestHyperparams, seed: number): TrainedForest {
  const numFeatures = points[0]?.features.length ?? 0;
  const featureSubsetSize = Math.max(1, Math.round(Math.sqrt(numFeatures)));

  const trees: TrainedTree[] = [];
  for (let i = 0; i < params.numTrees; i++) {
    const bootstrapRng = makeRng(seed * 1000003 + i * 7919 + 1);
    const bootstrap = sampleWithReplacement(points, points.length, bootstrapRng);
    const splitRng = makeRng(seed * 104729 + i * 15485863 + 7);
    const tree = trainDecisionTree(bootstrap, { maxDepth: params.maxDepth }, { featureSubsetSize, rng: splitRng });
    trees.push(tree);
  }

  return { trees, numFeatures };
}

/** Majority vote across every tree in the forest. */
export function predictForest(forest: TrainedForest, point: Point): number {
  const votes: Record<number, number> = {};
  for (const tree of forest.trees) {
    const p = predict(tree.root, point);
    votes[p] = (votes[p] ?? 0) + 1;
  }
  let best = -Infinity;
  let label = 0;
  for (const [k, v] of Object.entries(votes)) {
    if (v > best) {
      best = v;
      label = Number(k);
    }
  }
  return label;
}

/** Average of each tree's normalized feature importance, renormalized to max=1. */
export function forestFeatureImportance(forest: TrainedForest, numFeatures: number): Float64Array {
  const sum = new Float64Array(numFeatures);
  for (const tree of forest.trees) {
    const imp = featureImportance(tree, numFeatures);
    for (let i = 0; i < numFeatures; i++) sum[i] += imp[i];
  }
  const max = sum.reduce((m, v) => Math.max(m, v), 1e-9);
  for (let i = 0; i < numFeatures; i++) sum[i] /= max;
  return sum;
}
