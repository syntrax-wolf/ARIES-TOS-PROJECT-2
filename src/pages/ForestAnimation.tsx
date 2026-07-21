import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "../state/store";
import { bfsOrder, type TreeNode, type TrainedTree } from "../lib/decisionTree";
import { layoutTree, generateRootSystem } from "../lib/layout";
import { classColor } from "../lib/palette";

const LEVEL_HEIGHT = 34;
const TRUNK_BASE = 40;
const TOP_PADDING = 26;
const BOTTOM_PADDING = 18;
const MARGIN_X = 50;
const WIDE_SLOT = 260;
const TIGHT_SLOT = 168;

interface TreeGeometry {
  tree: TrainedTree;
  nodes: TreeNode[];
  edges: { parent: TreeNode; child: TreeNode }[];
  pixelOf: (node: TreeNode) => { x: number; y: number };
  rootPixel: { x: number; y: number };
  segments: ReturnType<typeof generateRootSystem>;
  wideX: number;
  tightX: number;
}

export function ForestAnimation() {
  const forest = useSylvaStore((s) => s.forest);
  const seed = useSylvaStore((s) => s.datasetConfig.seed);
  const setPage = useSylvaStore((s) => s.setPage);

  const [currentLevel, setCurrentLevel] = useState(-1);
  const [grown, setGrown] = useState(false);

  const totalLevels = useMemo(() => (forest ? Math.max(...forest.trees.map((t) => t.depth)) : 0), [forest]);

  const geometry = useMemo(() => {
    if (!forest) return null;
    const n = forest.trees.length;
    const contentWidth = n * WIDE_SLOT + 2 * MARGIN_X;
    const centerX = contentWidth / 2;

    const horizonY = Math.max(...forest.trees.map((t) => TRUNK_BASE + t.depth * LEVEL_HEIGHT + TOP_PADDING), 200);
    const rootAreaHeight = horizonY * 0.5;
    const totalHeight = horizonY + rootAreaHeight + BOTTOM_PADDING;

    const trees: TreeGeometry[] = forest.trees.map((tree, i) => {
      const layout = layoutTree(tree.root);
      const nodes = bfsOrder(tree.root);
      const edges: { parent: TreeNode; child: TreeNode }[] = [];
      for (const node of nodes) {
        if (node.left) edges.push({ parent: node, child: node.left });
        if (node.right) edges.push({ parent: node, child: node.right });
      }

      const pixelOf = (node: TreeNode) => {
        const l = layout.positions.get(node.id)!;
        return { x: l.x * LEVEL_HEIGHT, y: horizonY - TRUNK_BASE - l.y * LEVEL_HEIGHT };
      };
      const rootPixel = pixelOf(tree.root);

      const rootLevels = Math.max(2, Math.min(tree.depth, 5));
      const rawSegments = generateRootSystem(seed + i * 97 + 11, rootLevels, layout.leafSlots);
      const maxReach = rawSegments.reduce((m, s) => Math.max(m, s.y2), 0.001);
      const scale = (rootAreaHeight * 0.85) / maxReach;
      const xScale = scale * 0.82;
      const segments = rawSegments.map((s) => ({
        ...s,
        x1: rootPixel.x + s.x1 * xScale,
        y1: horizonY + s.y1 * scale,
        x2: rootPixel.x + s.x2 * xScale,
        y2: horizonY + s.y2 * scale,
        cx: rootPixel.x + s.cx * xScale,
        cy: horizonY + s.cy * scale,
      }));

      const wideX = MARGIN_X + WIDE_SLOT * (i + 0.5);
      const tightX = centerX + TIGHT_SLOT * (i - (n - 1) / 2);

      return { tree, nodes, edges, pixelOf, rootPixel, segments, wideX, tightX };
    });

    return { contentWidth, totalHeight, horizonY, trees };
  }, [forest, seed]);

  useEffect(() => {
    if (!forest) return;
    setCurrentLevel(-1);
    setGrown(false);
    const perLevel = Math.max(0.28, Math.min(0.85, 4.2 / (totalLevels + 1)));
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 0; i <= totalLevels; i++) {
      timers.push(setTimeout(() => setCurrentLevel(i), i * perLevel * 1000 + 250));
    }
    timers.push(setTimeout(() => setGrown(true), (totalLevels + 1) * perLevel * 1000 + 650));
    return () => timers.forEach(clearTimeout);
  }, [forest, totalLevels]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  if (!forest || !geometry) return null;

  const { contentWidth, totalHeight, horizonY, trees } = geometry;
  const perLevel = Math.max(0.28, Math.min(0.85, 4.2 / (totalLevels + 1)));

  return (
    <div className="fixed inset-0 overflow-hidden bg-sky">
      <svg viewBox={`0 0 ${contentWidth} ${totalHeight}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        <defs>
          <linearGradient id="fSkyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f2f7f1" />
            <stop offset="100%" stopColor="#e9f1ea" />
          </linearGradient>
          <linearGradient id="fSoilGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ece2cd" />
            <stop offset="100%" stopColor="#ddcba3" />
          </linearGradient>
          <filter id="fLeafShadow" x="-60%" y="-60%" width="220%" height="220%">
            <feDropShadow dx="0" dy="1" stdDeviation="1.3" floodColor="#1d2a1f" floodOpacity="0.18" />
          </filter>
          <radialGradient id="fVignette" cx="50%" cy="30%" r="75%">
            <stop offset="60%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="100%" stopColor="#1d2a1f" stopOpacity="0.05" />
          </radialGradient>
        </defs>

        <rect x={0} y={0} width={contentWidth} height={horizonY} fill="url(#fSkyGrad)" />
        <rect x={0} y={horizonY} width={contentWidth} height={totalHeight - horizonY} fill="url(#fSoilGrad)" />
        <line x1={0} y1={horizonY} x2={contentWidth} y2={horizonY} stroke="#c9b98c" strokeWidth={1.5} opacity={0.6} />
        {Array.from({ length: 5 }).map((_, i) => (
          <line
            key={i}
            x1={0}
            y1={horizonY + 10 + i * 12}
            x2={contentWidth}
            y2={horizonY + 10 + i * 12}
            stroke="#c9b98c"
            strokeWidth={0.6}
            opacity={0.15}
          />
        ))}

        {trees.map((t, i) => (
          <motion.g
            key={i}
            initial={{ x: t.wideX }}
            animate={{
              x: grown ? t.tightX : t.wideX,
              rotate: grown ? [0, -0.5, 0, 0.5, 0] : 0,
            }}
            transition={
              grown
                ? { x: { duration: 1.1, ease: "easeInOut" }, rotate: { duration: 6.5, repeat: Infinity, ease: "easeInOut", delay: 1.1 + i * 0.15 } }
                : { duration: 0.3 }
            }
            style={{ transformOrigin: `0px ${horizonY}px` }}
          >
            <g>
              {t.segments
                .filter((s) => s.level <= currentLevel)
                .map((s, si) => (
                  <motion.path
                    key={`root-${si}`}
                    d={`M ${s.x1} ${s.y1} Q ${s.cx} ${s.cy} ${s.x2} ${s.y2}`}
                    fill="none"
                    stroke="#a8875a"
                    strokeLinecap="round"
                    strokeWidth={s.width * 0.7}
                    opacity={0.5}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: perLevel * 0.85, ease: "easeOut" }}
                  />
                ))}
            </g>

            {currentLevel >= 0 && (
              <motion.path
                d={`M ${t.rootPixel.x} ${horizonY} L ${t.rootPixel.x} ${t.rootPixel.y}`}
                stroke="#8a6f45"
                strokeWidth={4}
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
              />
            )}

            {t.edges
              .filter((e) => e.child.depth <= currentLevel)
              .map((e) => {
                const p1 = t.pixelOf(e.parent);
                const p2 = t.pixelOf(e.child);
                const mx = (p1.x + p2.x) / 2;
                const my = p1.y - 6;
                const depthRatio = e.child.depth / Math.max(t.tree.depth, 1);
                const width = 3.6 - depthRatio * 2.4;
                return (
                  <motion.path
                    key={`edge-${e.child.id}`}
                    d={`M ${p1.x} ${p1.y} Q ${mx} ${my} ${p2.x} ${p2.y}`}
                    fill="none"
                    stroke="#5c7a5f"
                    strokeWidth={Math.max(0.9, width)}
                    strokeLinecap="round"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: perLevel * 0.85, ease: "easeOut" }}
                  />
                );
              })}

            {t.nodes
              .filter((n) => n.depth <= currentLevel)
              .map((n) => {
                const p = t.pixelOf(n);
                if (n.isLeaf) {
                  const total = Object.values(n.classCounts).reduce((a, b) => a + b, 0);
                  const purity = total ? (n.classCounts[n.prediction] ?? 0) / total : 1;
                  const r = 4.5 + purity * 2.2;
                  return (
                    <motion.circle
                      key={`node-${n.id}`}
                      cx={p.x}
                      cy={p.y}
                      r={r}
                      fill={classColor(n.prediction)}
                      filter="url(#fLeafShadow)"
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.35, delay: perLevel * 0.35, ease: "backOut" }}
                      style={{ transformOrigin: `${p.x}px ${p.y}px` }}
                    />
                  );
                }
                return (
                  <motion.circle
                    key={`node-${n.id}`}
                    cx={p.x}
                    cy={p.y}
                    r={2}
                    fill="#5c7a5f"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.25, ease: "backOut" }}
                    style={{ transformOrigin: `${p.x}px ${p.y}px` }}
                  />
                );
              })}
          </motion.g>
        ))}

        <rect x={0} y={0} width={contentWidth} height={totalHeight} fill="url(#fVignette)" pointerEvents="none" />
      </svg>

      <AnimatePresence>
        {grown && (
          <motion.div
            className="fixed inset-x-0 bottom-10 flex justify-center"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 14 }}
            transition={{ duration: 0.4, delay: 0.8 }}
          >
            <button
              onClick={() => setPage("results")}
              className="rounded-xl bg-brand px-7 py-3.5 text-base font-medium text-white shadow-soft transition-all hover:bg-brand-dark active:scale-[0.98]"
            >
              View results →
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
