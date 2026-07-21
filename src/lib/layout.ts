import type { TreeNode } from "./decisionTree";

export interface NodeLayout {
  x: number; // horizontal offset from the trunk, in "radius units" (multiply by level-height px)
  y: number; // vertical offset above the trunk top, in the same units
  depth: number;
}

export interface TreeLayoutResult {
  positions: Map<number, NodeLayout>;
  leafSlots: number;
  maxDepth: number;
  maxAbsX: number;
}

const CANOPY_SPREAD_DEG = 62;

/**
 * Radial fan layout: every node gets an angular slice proportional to its
 * subtree's leaf count (so siblings never overlap, exactly like a classic
 * leaf-slot diagram) but position is polar — angle * radius(=depth) — rather
 * than plain x/depth. That keeps shallow branches close to the trunk and
 * lets only the deepest twigs reach the outer edge, which is what makes the
 * silhouette read as a tapering tree canopy instead of a flat diagram.
 */
export function layoutTree(root: TreeNode): TreeLayoutResult {
  const slots = new Map<number, number>();
  const allNodes: TreeNode[] = [];
  let leafCounter = 0;
  let maxDepth = 0;

  function assignSlot(node: TreeNode): number {
    maxDepth = Math.max(maxDepth, node.depth);
    allNodes.push(node);
    if (node.isLeaf || !node.left || !node.right) {
      const slot = leafCounter;
      leafCounter += 1;
      slots.set(node.id, slot);
      return slot;
    }
    const lx = assignSlot(node.left);
    const rx = assignSlot(node.right);
    const mid = (lx + rx) / 2;
    slots.set(node.id, mid);
    return mid;
  }

  assignSlot(root);

  const leafSlots = Math.max(leafCounter, 1);
  const denom = Math.max(leafCounter - 1, 1);
  const maxSpreadRad = (CANOPY_SPREAD_DEG * Math.PI) / 180;

  const positions = new Map<number, NodeLayout>();
  let maxAbsX = 0;

  for (const node of allNodes) {
    const slot = slots.get(node.id)!;
    const normalized = leafCounter <= 1 ? 0.5 : slot / denom;
    const angle = -maxSpreadRad + normalized * 2 * maxSpreadRad;
    const radius = node.depth;
    const x = radius * Math.sin(angle);
    const y = radius * Math.cos(angle);
    positions.set(node.id, { x, y, depth: node.depth });
    maxAbsX = Math.max(maxAbsX, Math.abs(x));
  }

  return { positions, leafSlots, maxDepth, maxAbsX };
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

export interface RootSegment {
  level: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  width: number;
}

/**
 * Purely decorative mirror of the canopy: a procedurally branched root system
 * growing downward from the trunk base, scaled to roughly match the fitted
 * tree's depth/leaf count so denser trees also feel more "rooted".
 */
export function generateRootSystem(seed: number, levels: number, leaves: number): RootSegment[] {
  const rng = makeRng(seed * 7919 + 13);
  const segments: RootSegment[] = [];
  const branchesPerLevel = leaves > 6 ? 3 : 2;

  function grow(x: number, y: number, angleDeg: number, length: number, level: number) {
    if (level > levels) return;
    const spread = 22 + rng() * 14;
    const angleRad = (angleDeg * Math.PI) / 180;
    const midLen = length * (0.55 + rng() * 0.1);
    const endLen = length;
    const mx = x + Math.sin(angleRad) * midLen * 0.5 + (rng() - 0.5) * 0.04;
    const my = y + Math.cos(angleRad) * midLen;
    const ex = x + Math.sin(angleRad) * endLen;
    const ey = y + Math.cos(angleRad) * endLen;

    segments.push({
      level,
      x1: x,
      y1: y,
      x2: ex,
      y2: ey,
      cx: mx,
      cy: my,
      width: Math.max(0.6, 5 - level * 1.1),
    });

    if (level === levels) return;
    const children = level === 0 ? branchesPerLevel : rng() > 0.35 ? 2 : 1;
    for (let i = 0; i < children; i++) {
      const dir = children === 1 ? 0 : i === 0 ? -1 : 1;
      const childAngle = angleDeg + dir * spread + (rng() - 0.5) * 10;
      grow(ex, ey, childAngle, length * (0.72 + rng() * 0.1), level + 1);
    }
  }

  // angle convention: 0deg points straight down (+y), positive angles tilt right
  grow(0, 0, 0, 1, 0);
  return segments;
}
