import type { Point } from "./dataset";

export type Criterion = "gini" | "entropy" | "log_loss";
export type Splitter = "best" | "random";

export interface TreeHyperparams {
  criterion: Criterion;
  splitter: Splitter;
  maxDepth: number | null;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  minImpurityDecrease: number;
  maxLeafNodes: number | null;
}

export interface TreeNode {
  id: number;
  depth: number;
  points: Point[];
  classCounts: Record<number, number>;
  prediction: number;
  impurity: number;
  isLeaf: boolean;
  featureIndex?: 0 | 1;
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

function classCounts(points: Point[]): Record<number, number> {
  const counts: Record<number, number> = {};
  for (const p of points) counts[p.label] = (counts[p.label] ?? 0) + 1;
  return counts;
}

function impurityOf(counts: Record<number, number>, total: number, criterion: Criterion): number {
  if (total === 0) return 0;
  const ps = Object.values(counts).map((c) => c / total);
  if (criterion === "gini") {
    return 1 - ps.reduce((s, p) => s + p * p, 0);
  }
  // entropy and log_loss both reduce to Shannon entropy for a single-node classification split
  return -ps.reduce((s, p) => (p > 0 ? s + p * Math.log2(p) : s), 0);
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

interface Candidate {
  featureIndex: 0 | 1;
  threshold: number;
  impurityDecrease: number;
  leftPoints: Point[];
  rightPoints: Point[];
  leftImpurity: number;
  rightImpurity: number;
}

function evaluateThreshold(
  points: Point[],
  featureIndex: 0 | 1,
  threshold: number,
  parentImpurity: number,
  criterion: Criterion,
  minSamplesLeaf: number
): Candidate | null {
  const leftPoints: Point[] = [];
  const rightPoints: Point[] = [];
  for (const p of points) {
    const v = featureIndex === 0 ? p.x : p.y;
    if (v <= threshold) leftPoints.push(p);
    else rightPoints.push(p);
  }
  if (leftPoints.length < minSamplesLeaf || rightPoints.length < minSamplesLeaf) return null;

  const total = points.length;
  const leftImpurity = impurityOf(classCounts(leftPoints), leftPoints.length, criterion);
  const rightImpurity = impurityOf(classCounts(rightPoints), rightPoints.length, criterion);
  const weighted = (leftPoints.length / total) * leftImpurity + (rightPoints.length / total) * rightImpurity;
  const impurityDecrease = parentImpurity - weighted;

  return { featureIndex, threshold, impurityDecrease, leftPoints, rightPoints, leftImpurity, rightImpurity };
}

function bestSplitForFeature(
  points: Point[],
  featureIndex: 0 | 1,
  parentImpurity: number,
  criterion: Criterion,
  minSamplesLeaf: number
): Candidate | null {
  const values = Array.from(new Set(points.map((p) => (featureIndex === 0 ? p.x : p.y)))).sort((a, b) => a - b);
  if (values.length < 2) return null;

  let best: Candidate | null = null;
  for (let i = 0; i < values.length - 1; i++) {
    const threshold = (values[i] + values[i + 1]) / 2;
    const candidate = evaluateThreshold(points, featureIndex, threshold, parentImpurity, criterion, minSamplesLeaf);
    if (candidate && (!best || candidate.impurityDecrease > best.impurityDecrease)) {
      best = candidate;
    }
  }
  return best;
}

function randomSplitForFeature(
  points: Point[],
  featureIndex: 0 | 1,
  parentImpurity: number,
  criterion: Criterion,
  minSamplesLeaf: number,
  rng: () => number
): Candidate | null {
  const vals = points.map((p) => (featureIndex === 0 ? p.x : p.y));
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  if (max - min < 1e-9) return null;

  for (let attempt = 0; attempt < 8; attempt++) {
    const threshold = min + rng() * (max - min);
    const candidate = evaluateThreshold(points, featureIndex, threshold, parentImpurity, criterion, minSamplesLeaf);
    if (candidate) return candidate;
  }
  return null;
}

function candidateForFeature(
  points: Point[],
  featureIndex: 0 | 1,
  parentImpurity: number,
  params: TreeHyperparams,
  rng: () => number
): Candidate | null {
  if (params.splitter === "best") {
    return bestSplitForFeature(points, featureIndex, parentImpurity, params.criterion, params.minSamplesLeaf);
  }
  return randomSplitForFeature(points, featureIndex, parentImpurity, params.criterion, params.minSamplesLeaf, rng);
}

function findSplit(
  points: Point[],
  parentImpurity: number,
  params: TreeHyperparams,
  rng: () => number
): Candidate | null {
  const cx = candidateForFeature(points, 0, parentImpurity, params, rng);
  const cy = candidateForFeature(points, 1, parentImpurity, params, rng);

  let best: Candidate | null = cx;
  if (cy !== null && (best === null || cy.impurityDecrease > best.impurityDecrease)) best = cy;
  return best;
}

/**
 * Best-first tree growth: at each step, expand the frontier leaf whose
 * candidate split yields the largest impurity decrease. This lets max_leaf_nodes
 * cap growth naturally (sklearn does the same), and decouples build order from the
 * depth-order BFS used later purely for the reveal animation.
 */
export function trainDecisionTree(points: Point[], params: TreeHyperparams, seed = 1): TrainedTree {
  const rng = makeRng(seed);
  let nextId = 0;

  function makeNode(pts: Point[], depth: number, parentId: number | null): TreeNode {
    const counts = classCounts(pts);
    return {
      id: nextId++,
      depth,
      points: pts,
      classCounts: counts,
      prediction: majorityLabel(counts),
      impurity: impurityOf(counts, pts.length, params.criterion),
      isLeaf: true,
      parentId,
    };
  }

  const root = makeNode(points, 0, null);

  interface Frontier {
    node: TreeNode;
    candidate: Candidate;
  }
  const frontier: Frontier[] = [];
  const permanentLeaves = new Set<number>();

  function tryEnqueue(node: TreeNode) {
    if (permanentLeaves.has(node.id)) return;
    const atDepthLimit = params.maxDepth !== null && node.depth >= params.maxDepth;
    const tooFewSamples = node.points.length < params.minSamplesSplit;
    const isPure = node.impurity <= 1e-12;
    if (atDepthLimit || tooFewSamples || isPure) {
      permanentLeaves.add(node.id);
      return;
    }
    const candidate = findSplit(node.points, node.impurity, params, rng);
    if (!candidate || candidate.impurityDecrease < params.minImpurityDecrease) {
      permanentLeaves.add(node.id);
      return;
    }
    frontier.push({ node, candidate });
  }

  tryEnqueue(root);
  let leafCount = 1;

  while (frontier.length > 0) {
    if (params.maxLeafNodes !== null && leafCount >= params.maxLeafNodes) break;

    let bestIdx = 0;
    for (let i = 1; i < frontier.length; i++) {
      if (frontier[i].candidate.impurityDecrease > frontier[bestIdx].candidate.impurityDecrease) bestIdx = i;
    }
    const { node, candidate } = frontier[bestIdx];
    frontier.splice(bestIdx, 1);

    node.isLeaf = false;
    node.featureIndex = candidate.featureIndex;
    node.threshold = candidate.threshold;
    const left = makeNode(candidate.leftPoints, node.depth + 1, node.id);
    const right = makeNode(candidate.rightPoints, node.depth + 1, node.id);
    node.left = left;
    node.right = right;
    leafCount += 1; // net change: -1 (node) + 2 (children)

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
    const v = node.featureIndex === 0 ? point.x : point.y;
    node = v <= (node.threshold as number) ? node.left : node.right;
  }
  return node.prediction;
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
