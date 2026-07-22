import type { Point } from "./dataset";
import { NUM_CLASSES } from "./mnist";
import { regressionBfsOrder, trainRegressionTree, predictRegression, type RegressionTree } from "./regressionTree";

// Kept to a single knob, like the tree-based algorithms: how many rounds of
// weak learners get added. Learning rate and each learner's depth are fixed,
// sane defaults so a beginner isn't asked to tune three things at once.
export interface BoostingHyperparams {
  numLearners: number;
}

const LEARNING_RATE = 0.35;
export const LEARNER_MAX_DEPTH = 2;

export interface BoostingRound {
  trees: RegressionTree[]; // one small regression tree per class, fit to that class's residual
}

export interface TrainedBoosting {
  rounds: BoostingRound[];
  numFeatures: number;
}

function softmax(scores: number[]): number[] {
  const max = Math.max(...scores);
  const exps = scores.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Multi-class additive boosting: at every round, each class gets its own
 * shallow regression tree fit to that class's current softmax residual
 * (label indicator minus predicted probability). Adding the scaled
 * prediction of every round's tree, class by class, is literally what
 * "combining weak learners into a strong model" means here — nothing about
 * the accumulation is decorative.
 */
export function trainGradientBoosting(points: Point[], params: BoostingHyperparams): TrainedBoosting {
  const n = points.length;
  const numFeatures = points[0]?.features.length ?? 0;
  const scores: number[][] = Array.from({ length: NUM_CLASSES }, () => new Array(n).fill(0));
  const rounds: BoostingRound[] = [];

  for (let round = 0; round < params.numLearners; round++) {
    const probs: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
      probs[i] = softmax(NUM_CLASSES > 0 ? Array.from({ length: NUM_CLASSES }, (_, k) => scores[k][i]) : []);
    }

    const trees: RegressionTree[] = [];
    for (let k = 0; k < NUM_CLASSES; k++) {
      const samples = points.map((p, i) => ({
        features: p.features,
        target: (p.label === k ? 1 : 0) - probs[i][k],
      }));
      const tree = trainRegressionTree(samples, LEARNER_MAX_DEPTH);
      trees.push(tree);
      for (let i = 0; i < n; i++) {
        scores[k][i] += LEARNING_RATE * predictRegression(tree.root, points[i].features);
      }
    }
    rounds.push({ trees });
  }

  return { rounds, numFeatures };
}

export function boostingScores(model: TrainedBoosting, point: Point): number[] {
  const scores = new Array(NUM_CLASSES).fill(0);
  for (const round of model.rounds) {
    for (let k = 0; k < NUM_CLASSES; k++) {
      scores[k] += LEARNING_RATE * predictRegression(round.trees[k].root, point.features);
    }
  }
  return scores;
}

export function predictBoosting(model: TrainedBoosting, point: Point): number {
  const scores = boostingScores(model, point);
  let best = -Infinity;
  let label = 0;
  for (let k = 0; k < scores.length; k++) {
    if (scores[k] > best) {
      best = scores[k];
      label = k;
    }
  }
  return label;
}

/** Sum of (sample-share weighted) split gains across every round and class, normalized to max=1. */
export function boostingFeatureImportance(model: TrainedBoosting, numFeatures: number): Float64Array {
  const importance = new Float64Array(numFeatures);
  for (const round of model.rounds) {
    for (const tree of round.trees) {
      const total = tree.root.sampleCount;
      if (total === 0) continue;
      for (const node of regressionBfsOrder(tree.root)) {
        if (node.isLeaf || node.featureIndex === undefined || node.splitGain === undefined) continue;
        importance[node.featureIndex] += (node.sampleCount / total) * node.splitGain;
      }
    }
  }
  const max = importance.reduce((m, v) => Math.max(m, v), 1e-9);
  for (let i = 0; i < numFeatures; i++) importance[i] /= max;
  return importance;
}
