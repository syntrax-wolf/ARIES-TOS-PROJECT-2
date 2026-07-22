import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "../state/store";
import type { NetworkSnapshot } from "../lib/neuralNetwork";
import { classColor } from "../lib/palette";
import { NUM_CLASSES } from "../lib/mnist";

const INPUT_DISPLAY = 10;
const COLUMN_GAP = 170;
const ROW_GAP = 32;
const MARGIN_X = 90;
const MARGIN_Y = 60;
const STAGGER = 0.09;
const PULSE_DUR = 0.6;
const CYCLE_MS = 950;
const BUILD_DELAY_MS = 300;

interface NodeSpec {
  layer: number;
  index: number;
  x: number;
  y: number;
  kind: "input" | "hidden" | "output";
}

interface EdgeSpec {
  id: string;
  fromLayer: number;
  i: number;
  j: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Coarsely averages the real 196 input weights down to 10 display buckets — an honest summary, not decoration. */
function bucketedInputWeights(W: number[][], inputSize: number, displayCount: number, outSize: number): number[][] {
  const bucketSize = Math.ceil(inputSize / displayCount);
  const sums = Array.from({ length: displayCount }, () => new Array(outSize).fill(0));
  const counts = new Array(displayCount).fill(0);
  for (let i = 0; i < inputSize; i++) {
    const bucket = Math.min(displayCount - 1, Math.floor(i / bucketSize));
    counts[bucket] += 1;
    for (let j = 0; j < outSize; j++) sums[bucket][j] += W[i][j];
  }
  for (let b = 0; b < displayCount; b++) {
    const c = Math.max(1, counts[b]);
    for (let j = 0; j < outSize; j++) sums[b][j] /= c;
  }
  return sums;
}

function displayWeightsFor(snapshot: NetworkSnapshot, inputSize: number, layerSizesDisplay: number[]): number[][][] {
  return snapshot.layers.map((layer, l) =>
    l === 0 ? bucketedInputWeights(layer.W, inputSize, layerSizesDisplay[0], layerSizesDisplay[1]) : layer.W
  );
}

function edgeStyle(weight: number, maxAbs: number) {
  const norm = maxAbs > 0 ? Math.min(1, Math.abs(weight) / maxAbs) : 0;
  return {
    stroke: weight >= 0 ? "#2f6b4f" : "#c07a3e",
    width: 0.6 + norm * 2.6,
    opacity: 0.12 + norm * 0.78,
  };
}

export function NeuralNetworkAnimation() {
  const network = useSylvaStore((s) => s.network);
  const setPage = useSylvaStore((s) => s.setPage);

  const [snapshotIndex, setSnapshotIndex] = useState(-1);
  const [grown, setGrown] = useState(false);

  const layerSizesDisplay = useMemo(() => {
    if (!network) return [];
    const hiddenSizes = network.layerSizes.slice(1, -1);
    return [INPUT_DISPLAY, ...hiddenSizes, NUM_CLASSES];
  }, [network]);

  const geometry = useMemo(() => {
    if (!network || layerSizesDisplay.length === 0) return null;
    const numLayers = layerSizesDisplay.length;
    const maxNodes = Math.max(...layerSizesDisplay);
    const contentWidth = MARGIN_X * 2 + (numLayers - 1) * COLUMN_GAP;
    const contentHeight = MARGIN_Y * 2 + (maxNodes - 1) * ROW_GAP;

    const nodePos = (l: number, k: number) => {
      const x = MARGIN_X + l * COLUMN_GAP;
      const count = layerSizesDisplay[l];
      const span = (count - 1) * ROW_GAP;
      const yStart = contentHeight / 2 - span / 2;
      return { x, y: yStart + k * ROW_GAP };
    };

    const nodes: NodeSpec[] = [];
    for (let l = 0; l < numLayers; l++) {
      for (let k = 0; k < layerSizesDisplay[l]; k++) {
        const { x, y } = nodePos(l, k);
        const kind = l === 0 ? "input" : l === numLayers - 1 ? "output" : "hidden";
        nodes.push({ layer: l, index: k, x, y, kind });
      }
    }

    const edges: EdgeSpec[] = [];
    for (let l = 0; l < numLayers - 1; l++) {
      for (let i = 0; i < layerSizesDisplay[l]; i++) {
        const p1 = nodePos(l, i);
        for (let j = 0; j < layerSizesDisplay[l + 1]; j++) {
          const p2 = nodePos(l + 1, j);
          edges.push({ id: `${l}-${i}-${j}`, fromLayer: l, i, j, x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
        }
      }
    }

    return { contentWidth, contentHeight, numLayers, nodes, edges };
  }, [network, layerSizesDisplay]);

  const inputSize = network?.layerSizes[0] ?? 0;

  const currentDisplayWeights = useMemo(() => {
    if (!network) return null;
    const idx = Math.max(0, snapshotIndex);
    return displayWeightsFor(network.snapshots[idx], inputSize, layerSizesDisplay);
  }, [network, snapshotIndex, inputSize, layerSizesDisplay]);

  const maxAbsWeight = useMemo(() => {
    if (!network) return 1;
    const final = displayWeightsFor(network.snapshots[network.snapshots.length - 1], inputSize, layerSizesDisplay);
    let max = 1e-9;
    for (const layerW of final) for (const row of layerW) for (const w of row) max = Math.max(max, Math.abs(w));
    return max;
  }, [network, inputSize, layerSizesDisplay]);

  useEffect(() => {
    if (!network) return;
    setSnapshotIndex(-1);
    setGrown(false);
    const n = network.snapshots.length;
    const timers: ReturnType<typeof setTimeout>[] = [];
    timers.push(setTimeout(() => setSnapshotIndex(0), BUILD_DELAY_MS));
    for (let i = 1; i < n; i++) {
      timers.push(setTimeout(() => setSnapshotIndex(i), BUILD_DELAY_MS + i * CYCLE_MS));
    }
    timers.push(setTimeout(() => setGrown(true), BUILD_DELAY_MS + n * CYCLE_MS + 300));
    return () => timers.forEach(clearTimeout);
  }, [network]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  if (!network || !geometry || !currentDisplayWeights) return null;

  const { contentWidth, contentHeight, nodes, edges } = geometry;
  const visible = snapshotIndex >= 0;

  return (
    <div className="fixed inset-0 overflow-hidden bg-bg">
      <svg viewBox={`0 0 ${contentWidth} ${contentHeight}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        <defs>
          <radialGradient id="nnVignette" cx="50%" cy="45%" r="75%">
            <stop offset="55%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="100%" stopColor="#1d2a1f" stopOpacity="0.045" />
          </radialGradient>
          <filter id="nnGlow" x="-80%" y="-80%" width="260%" height="260%">
            <feDropShadow dx="0" dy="0" stdDeviation="2.2" floodColor="#2f6b4f" floodOpacity="0.45" />
          </filter>
        </defs>

        <rect x={0} y={0} width={contentWidth} height={contentHeight} fill="url(#nnVignette)" pointerEvents="none" />

        {edges.map((e) => {
          const w = currentDisplayWeights[e.fromLayer][e.i][e.j];
          const style = edgeStyle(w, maxAbsWeight);
          return (
            <motion.line
              key={e.id}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke={style.stroke}
              initial={{ opacity: 0, strokeWidth: 0.4 }}
              animate={{ opacity: visible ? style.opacity : 0, strokeWidth: style.width }}
              transition={{ duration: 0.6, delay: e.fromLayer * STAGGER * 0.6, ease: "easeInOut" }}
            />
          );
        })}

        {nodes.map((n) => {
          const isOutput = n.kind === "output";
          const isInput = n.kind === "input";
          const fill = isOutput ? classColor(n.index) : isInput ? "#b9c2b2" : "#e5efe6";
          const r = isInput ? 6 : isOutput ? 10 : 9;

          return (
            <g key={`node-${n.layer}-${n.index}`}>
              {isInput ? (
                <motion.rect
                  x={n.x - r}
                  y={n.y - r}
                  width={r * 2}
                  height={r * 2}
                  rx={2}
                  fill={fill}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: visible ? 1 : 0, scale: 1 }}
                  transition={{ duration: 0.4, ease: "backOut" }}
                  style={{ transformOrigin: `${n.x}px ${n.y}px` }}
                />
              ) : (
                <motion.circle
                  cx={n.x}
                  cy={n.y}
                  r={r}
                  fill={fill}
                  stroke={isOutput ? "none" : "#2f6b4f"}
                  strokeWidth={isOutput ? 0 : 1.2}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: visible ? 1 : 0, scale: 1 }}
                  transition={{ duration: 0.4, ease: "backOut" }}
                  style={{ transformOrigin: `${n.x}px ${n.y}px` }}
                />
              )}
              {isOutput && (
                <text
                  x={n.x}
                  y={n.y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={11}
                  fontWeight={600}
                  fill="#ffffff"
                  pointerEvents="none"
                >
                  {n.index}
                </text>
              )}

              {visible && (
                <motion.circle
                  key={`pulse-${n.layer}-${n.index}-${snapshotIndex}`}
                  cx={n.x}
                  cy={n.y}
                  r={r}
                  fill="none"
                  stroke={isInput ? "#8a9484" : "#2f6b4f"}
                  strokeWidth={1.6}
                  filter={isInput ? undefined : "url(#nnGlow)"}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: [0, 1, 0.35, 0.85, 0], scale: [0.85, 1.3, 1.05, 1.22, 1] }}
                  transition={{ duration: PULSE_DUR, delay: n.layer * STAGGER, ease: "easeInOut" }}
                  style={{ transformOrigin: `${n.x}px ${n.y}px` }}
                />
              )}
            </g>
          );
        })}
      </svg>

      <AnimatePresence>
        {grown && (
          <motion.div
            className="fixed inset-x-0 bottom-10 flex justify-center"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 14 }}
            transition={{ duration: 0.4, delay: 0.3 }}
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
