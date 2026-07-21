import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "../state/store";
import { bfsOrder, type TreeNode } from "../lib/decisionTree";
import { layoutTree, generateRootSystem } from "../lib/layout";
import { classColor } from "../lib/palette";
import { averageImage, pixelsToDataUrl, RAW_SIZE } from "../lib/mnist";

const MARGIN_X = 70;
const LEVEL_HEIGHT = 92;
const TRUNK_BASE = 74;
const TOP_PADDING = 64;
const BOTTOM_PADDING = 36;

export function TreeAnimation() {
  const tree = useSylvaStore((s) => s.tree);
  const seed = useSylvaStore((s) => s.datasetConfig.seed);
  const setPage = useSylvaStore((s) => s.setPage);

  const [currentLevel, setCurrentLevel] = useState(-1);
  const [grown, setGrown] = useState(false);

  const nodes = useMemo(() => (tree ? bfsOrder(tree.root) : []), [tree]);
  const layout = useMemo(() => (tree ? layoutTree(tree.root) : null), [tree]);

  // a blurry "prototype" thumbnail per leaf — the average of every training digit that landed there
  const leafThumbnails = useMemo(() => {
    const map = new Map<number, string>();
    for (const n of nodes) {
      if (!n.isLeaf) continue;
      const avg = averageImage(n.points.map((p) => p.raw));
      map.set(n.id, pixelsToDataUrl(avg, RAW_SIZE, 2));
    }
    return map;
  }, [nodes]);

  const geometry = useMemo(() => {
    if (!tree || !layout) return null;
    const halfWidth = Math.max(430, layout.maxAbsX * LEVEL_HEIGHT + MARGIN_X);
    const contentWidth = halfWidth * 2;
    const centerX = halfWidth;
    const canopyHeight = TRUNK_BASE + tree.depth * LEVEL_HEIGHT + TOP_PADDING;
    const rootLevels = Math.max(2, Math.min(tree.depth, 5));
    const rootAreaHeight = canopyHeight * 0.55;
    const totalHeight = canopyHeight + rootAreaHeight + BOTTOM_PADDING;
    const horizonY = canopyHeight;

    const pixelOf = (node: TreeNode) => {
      const l = layout.positions.get(node.id)!;
      const x = centerX + l.x * LEVEL_HEIGHT;
      const y = horizonY - TRUNK_BASE - l.y * LEVEL_HEIGHT;
      return { x, y };
    };

    const rootPixel = pixelOf(tree.root);
    const rawSegments = generateRootSystem(seed, rootLevels, layout.leafSlots);
    const maxReach = rawSegments.reduce((m, s) => Math.max(m, s.y2), 0.001);
    const scale = (rootAreaHeight * 0.92) / maxReach;
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

    return { contentWidth, totalHeight, horizonY, pixelOf, rootPixel, segments };
  }, [tree, layout, seed]);

  useEffect(() => {
    if (!tree) return;
    setCurrentLevel(-1);
    setGrown(false);
    const totalLevels = tree.depth;
    const perLevel = Math.max(0.32, Math.min(1.0, 4.4 / (totalLevels + 1)));
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 0; i <= totalLevels; i++) {
      timers.push(setTimeout(() => setCurrentLevel(i), i * perLevel * 1000 + 250));
    }
    timers.push(
      setTimeout(() => setGrown(true), (totalLevels + 1) * perLevel * 1000 + 550)
    );
    return () => timers.forEach(clearTimeout);
  }, [tree]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  if (!tree || !geometry) return null;

  const { contentWidth, totalHeight, horizonY, pixelOf, rootPixel, segments } = geometry;
  const perLevel = Math.max(0.32, Math.min(1.0, 4.4 / (tree.depth + 1)));

  const edges: { parent: TreeNode; child: TreeNode }[] = [];
  for (const n of nodes) {
    if (n.left) edges.push({ parent: n, child: n.left });
    if (n.right) edges.push({ parent: n, child: n.right });
  }

  return (
    <div className="fixed inset-0 overflow-hidden bg-sky">
      <svg
        viewBox={`0 0 ${contentWidth} ${totalHeight}`}
        preserveAspectRatio="xMidYMid meet"
        className="h-full w-full"
      >
        <defs>
          <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f2f7f1" />
            <stop offset="100%" stopColor="#e9f1ea" />
          </linearGradient>
          <linearGradient id="soilGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ece2cd" />
            <stop offset="100%" stopColor="#ddcba3" />
          </linearGradient>
          <filter id="leafShadow" x="-60%" y="-60%" width="220%" height="220%">
            <feDropShadow dx="0" dy="1.5" stdDeviation="2" floodColor="#1d2a1f" floodOpacity="0.18" />
          </filter>
          <radialGradient id="vignette" cx="50%" cy="30%" r="75%">
            <stop offset="60%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="100%" stopColor="#1d2a1f" stopOpacity="0.05" />
          </radialGradient>
        </defs>

        <rect x={0} y={0} width={contentWidth} height={horizonY} fill="url(#skyGrad)" />
        <rect x={0} y={horizonY} width={contentWidth} height={totalHeight - horizonY} fill="url(#soilGrad)" />
        <line x1={0} y1={horizonY} x2={contentWidth} y2={horizonY} stroke="#c9b98c" strokeWidth={1.5} opacity={0.6} />

        {/* soil striations, purely decorative texture */}
        {Array.from({ length: 5 }).map((_, i) => (
          <line
            key={i}
            x1={0}
            y1={horizonY + 14 + i * 16}
            x2={contentWidth}
            y2={horizonY + 14 + i * 16}
            stroke="#c9b98c"
            strokeWidth={0.6}
            opacity={0.15}
          />
        ))}

        <motion.g
          style={{ transformOrigin: `${rootPixel.x}px ${horizonY}px` }}
          animate={grown ? { rotate: [0, -0.5, 0, 0.5, 0] } : { rotate: 0 }}
          transition={grown ? { duration: 7, repeat: Infinity, ease: "easeInOut" } : undefined}
        >
          {/* underground root system, mirrors the canopy's branching */}
          <g>
            {segments
              .filter((s) => s.level <= currentLevel)
              .map((s, i) => (
                <motion.path
                  key={`root-${i}`}
                  d={`M ${s.x1} ${s.y1} Q ${s.cx} ${s.cy} ${s.x2} ${s.y2}`}
                  fill="none"
                  stroke="#a8875a"
                  strokeLinecap="round"
                  strokeWidth={s.width}
                  opacity={0.55}
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: perLevel * 0.85, ease: "easeOut" }}
                />
              ))}
          </g>

          {/* trunk: ground to the root decision node */}
          {currentLevel >= 0 && (
            <motion.path
              d={`M ${rootPixel.x} ${horizonY} L ${rootPixel.x} ${rootPixel.y}`}
              stroke="#8a6f45"
              strokeWidth={7}
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          )}

          {/* branches */}
          {edges
            .filter((e) => e.child.depth <= currentLevel)
            .map((e) => {
              const p1 = pixelOf(e.parent);
              const p2 = pixelOf(e.child);
              const mx = (p1.x + p2.x) / 2;
              const my = p1.y - 14;
              const depthRatio = e.child.depth / Math.max(tree.depth, 1);
              const width = 6.5 - depthRatio * 4.5;
              return (
                <motion.path
                  key={`edge-${e.child.id}`}
                  d={`M ${p1.x} ${p1.y} Q ${mx} ${my} ${p2.x} ${p2.y}`}
                  fill="none"
                  stroke="#5c7a5f"
                  strokeWidth={Math.max(1.6, width)}
                  strokeLinecap="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: perLevel * 0.85, ease: "easeOut" }}
                />
              );
            })}

          {/* nodes: small junctions for internal splits, colored leaves at the ends */}
          {nodes
            .filter((n) => n.depth <= currentLevel)
            .map((n) => {
              const p = pixelOf(n);
              if (n.isLeaf) {
                const total = Object.values(n.classCounts).reduce((a, b) => a + b, 0);
                const purity = total ? (n.classCounts[n.prediction] ?? 0) / total : 1;
                const r = 12 + purity * 6;
                const thumb = leafThumbnails.get(n.id);
                return (
                  <motion.g
                    key={`node-${n.id}`}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.4, delay: perLevel * 0.35, ease: "backOut" }}
                    style={{ transformOrigin: `${p.x}px ${p.y}px` }}
                  >
                    <circle cx={p.x} cy={p.y} r={r + 2} fill={classColor(n.prediction)} filter="url(#leafShadow)" />
                    <circle cx={p.x} cy={p.y} r={r} fill="#ffffff" />
                    {thumb && (
                      <>
                        <clipPath id={`leaf-clip-${n.id}`}>
                          <circle cx={p.x} cy={p.y} r={r} />
                        </clipPath>
                        <image
                          href={thumb}
                          x={p.x - r}
                          y={p.y - r}
                          width={r * 2}
                          height={r * 2}
                          clipPath={`url(#leaf-clip-${n.id})`}
                          style={{ imageRendering: "pixelated" }}
                        />
                      </>
                    )}
                  </motion.g>
                );
              }
              return (
                <motion.circle
                  key={`node-${n.id}`}
                  cx={p.x}
                  cy={p.y}
                  r={4}
                  fill="#5c7a5f"
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3, ease: "backOut" }}
                  style={{ transformOrigin: `${p.x}px ${p.y}px` }}
                />
              );
            })}
        </motion.g>

        <rect x={0} y={0} width={contentWidth} height={totalHeight} fill="url(#vignette)" pointerEvents="none" />
      </svg>

      <AnimatePresence>
        {grown && (
          <motion.div
            className="fixed inset-x-0 bottom-10 flex justify-center"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 14 }}
            transition={{ duration: 0.4 }}
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
