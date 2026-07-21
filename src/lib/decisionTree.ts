import type { Point } from "./dataset";

// Kept deliberately minimal: max depth is the only knob exposed to the learner.
// Everything else below is a fixed, sane default (gini impurity, best-split search,
// small leaf/split floors so a max-depth-only tree still stays a readable size).
export interface TreeHyperparams {
  maxDepth: number;
}

const MIN_SAMPLES_LEAF = 4;
const MIN_SAMPLES_SPLIT = 9;

export interface TreeNode {
  id: number;
  depth: number;
  points: Point[];
  classCounts: Record<number, number>;
  prediction: number;
  impurity: number;
  isLeaf: boolean;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  parentId: number | null;
}

export interface TrainedTree {
  root: TreeNode;
  nodeCount: number;
  leafCount: number;
  depth: number;
}

function classCounts(points: Point[]): Record<number, number> {
  const counts: Record<number, number> = {};
  for (const p of points) counts[p.label] = (counts[p.label] ?? 0) + 1;
  return counts;
}

function giniFromCounts(counts: Record<number, number>, total: number): number {
  if (total === 0) return 0;
  let sumSq = 0;
  for (const k in counts) {
    const p = counts[k] / total;
    sumSq += p * p;
  }
  return 1 - sumSq;
}

function giniRightFromDiff(totalCounts: Record<number, number>, leftCounts: Record<number, number>, rightTotal: number): number {
  if (rightTotal === 0) return 0;
  let sumSq = 0;
  for (const k in totalCounts) {
    const rightCount = totalCounts[k] - (leftCounts[k] ?? 0);
    const p = rightCount / rightTotal;
    sumSq += p * p;
  }
  return 1 - sumSq;
}

function majorityLabel(counts: Record<number, number>): number {
  let best = -Infinity;
  let label = 0;
  for (const [k, v] of Object.entries(counts)) {
    if (v > best) {
      best = v;
      label = Number(k);
    }
  }
  return label;
}

interface SplitCandidate {
  featureIndex: number;
  threshold: number;
  impurityDecrease: number;
}

/**
 * Single sorted sweep per feature: O(n log n) instead of re-scanning all points
 * for every candidate threshold. Necessary once features number in the hundreds
 * (14x14 pixel grid) rather than just two.
 */
function bestSplitForFeature(points: Point[], featureIndex: number, parentImpurity: number): SplitCandidate | null {
  const n = points.length;
  const entries = new Array<{ v: number; label: number }>(n);
  for (let i = 0; i < n; i++) {
    entries[i] = { v: points[i].features[featureIndex], label: points[i].label };
  }
  entries.sort((a, b) => a.v - b.v);

  const totalCounts: Record<number, number> = {};
  for (const e of entries) totalCounts[e.label] = (totalCounts[e.label] ?? 0) + 1;

  const leftCounts: Record<number, number> = {};
  let leftTotal = 0;
  let best: SplitCandidate | null = null;

  for (let i = 0; i < n - 1; i++) {
    const e = entries[i];
    leftCounts[e.label] = (leftCounts[e.label] ?? 0) + 1;
    leftTotal += 1;
    const nextV = entries[i + 1].v;
    if (nextV === e.v) continue;

    const rightTotal = n - leftTotal;
    if (leftTotal < MIN_SAMPLES_LEAF || rightTotal < MIN_SAMPLES_LEAF) continue;

    const leftImpurity = giniFromCounts(leftCounts, leftTotal);
    const rightImpurity = giniRightFromDiff(totalCounts, leftCounts, rightTotal);
    const weighted = (leftTotal / n) * leftImpurity + (rightTotal / n) * rightImpurity;
    const impurityDecrease = parentImpurity - weighted;

    if (!best || impurityDecrease > best.impurityDecrease) {
      best = { featureIndex, threshold: (e.v + nextV) / 2, impurityDecrease };
    }
  }
  return best;
}

function findBestSplit(points: Point[], parentImpurity: number): SplitCandidate | null {
  const numFeatures = points[0]?.features.length ?? 0;
  let best: SplitCandidate | null = null;
  for (let f = 0; f < numFeatures; f++) {
    const candidate = bestSplitForFeature(points, f, parentImpurity);
    if (candidate && (best === null || candidate.impurityDecrease > best.impurityDecrease)) {
      best = candidate;
    }
  }
  return best;
}

function partition(points: Point[], featureIndex: number, threshold: number): [Point[], Point[]] {
  const left: Point[] = [];
  const right: Point[] = [];
  for (const p of points) {
    (p.features[featureIndex] <= threshold ? left : right).push(p);
  }
  return [left, right];
}

/**
 * Best-first tree growth: at each step, expand the frontier leaf whose
 * candidate split yields the largest impurity decrease. Growth stops only on
 * max depth, purity, or running out of valid splits — there's no leaf-count
 * cap since max depth is the only exposed control.
 */
export function trainDecisionTree(points: Point[], params: TreeHyperparams): TrainedTree {
  let nextId = 0;

  function makeNode(pts: Point[], depth: number, parentId: number | null): TreeNode {
    const counts = classCounts(pts);
    return {
      id: nextId++,
      depth,
      points: pts,
      classCounts: counts,
      prediction: majorityLabel(counts),
      impurity: giniFromCounts(counts, pts.length),
      isLeaf: true,
      parentId,
    };
  }

  const root = makeNode(points, 0, null);

  interface Frontier {
    node: TreeNode;
    candidate: SplitCandidate;
  }
  const frontier: Frontier[] = [];
  const permanentLeaves = new Set<number>();

  function tryEnqueue(node: TreeNode) {
    if (permanentLeaves.has(node.id)) return;
    const atDepthLimit = node.depth >= params.maxDepth;
    const tooFewSamples = node.points.length < MIN_SAMPLES_SPLIT;
    const isPure = node.impurity <= 1e-12;
    if (atDepthLimit || tooFewSamples || isPure) {
      permanentLeaves.add(node.id);
      return;
    }
    const candidate = findBestSplit(node.points, node.impurity);
    if (!candidate || candidate.impurityDecrease <= 1e-9) {
      permanentLeaves.add(node.id);
      return;
    }
    frontier.push({ node, candidate });
  }

  tryEnqueue(root);

  while (frontier.length > 0) {
    let bestIdx = 0;
    for (let i = 1; i < frontier.length; i++) {
      if (frontier[i].candidate.impurityDecrease > frontier[bestIdx].candidate.impurityDecrease) bestIdx = i;
    }
    const { node, candidate } = frontier[bestIdx];
    frontier.splice(bestIdx, 1);

    const [leftPoints, rightPoints] = partition(node.points, candidate.featureIndex, candidate.threshold);
    node.isLeaf = false;
    node.featureIndex = candidate.featureIndex;
    node.threshold = candidate.threshold;
    const left = makeNode(leftPoints, node.depth + 1, node.id);
    const right = makeNode(rightPoints, node.depth + 1, node.id);
    node.left = left;
    node.right = right;

    tryEnqueue(left);
    tryEnqueue(right);
  }

  let maxDepth = 0;
  let leaves = 0;
  let nodes = 0;
  (function walk(n: TreeNode) {
    nodes += 1;
    maxDepth = Math.max(maxDepth, n.depth);
    if (n.isLeaf) leaves += 1;
    if (n.left) walk(n.left);
    if (n.right) walk(n.right);
  })(root);

  return { root, nodeCount: nodes, leafCount: leaves, depth: maxDepth };
}

export function predict(root: TreeNode, point: Point): number {
  let node = root;
  while (!node.isLeaf && node.left && node.right) {
    const v = point.features[node.featureIndex as number];
    node = v <= (node.threshold as number) ? node.left : node.right;
  }
  return node.prediction;
}

/**
 * Standard CART feature importance: for every split, how much it reduced
 * impurity, weighted by the fraction of samples that passed through it —
 * summed per feature and normalized so the largest value is 1. This is what
 * drives the "which pixels does the tree look at" heatmap.
 */
export function featureImportance(tree: TrainedTree, numFeatures: number): Float64Array {
  const importance = new Float64Array(numFeatures);
  const totalSamples = tree.root.points.length;

  function walk(n: TreeNode) {
    if (n.isLeaf || !n.left || !n.right || n.featureIndex === undefined) return;
    const weight = n.points.length / totalSamples;
    const leftShare = n.left.points.length / n.points.length;
    const rightShare = n.right.points.length / n.points.length;
    const decrease = n.impurity - leftShare * n.left.impurity - rightShare * n.right.impurity;
    importance[n.featureIndex] += weight * decrease;
    walk(n.left);
    walk(n.right);
  }
  walk(tree.root);

  const max = importance.reduce((m, v) => Math.max(m, v), 1e-9);
  for (let i = 0; i < importance.length; i++) importance[i] /= max;
  return importance;
}

/** Level-order traversal of the fitted tree — used to sequence the growth animation. */
export function bfsOrder(root: TreeNode): TreeNode[] {
  const order: TreeNode[] = [];
  let queue: TreeNode[] = [root];
  while (queue.length) {
    const next: TreeNode[] = [];
    for (const n of queue) {
      order.push(n);
      if (n.left) next.push(n.left);
      if (n.right) next.push(n.right);
    }
    queue = next;
  }
  return order;
}
