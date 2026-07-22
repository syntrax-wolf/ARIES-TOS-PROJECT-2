import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "../state/store";
import { forwardCnn, type CnnForward, type Grid } from "../lib/cnn";
import { pixelsToDataUrl, RAW_SIZE, NUM_CLASSES } from "../lib/mnist";
import { classColor, heatColor } from "../lib/palette";

const NUM_QUERIES = 5;
const STAGE_GAP_MS = 520;
const HOLD_MS = 1300;
const CENTER_Y = 260;
const VIEW_W = 1360;
const VIEW_H = 520;

const STAGE_X = { input: 90, conv1: 270, pool1: 430, conv2: 590, pool2: 740, flatten: 890, dense: 1060, output: 1240 };
const STAGE_LABELS = ["Digit", "Convolve", "Pool", "Convolve", "Pool", "Flatten", "Dense", "Decide"];

interface MapSlot {
  originX: number;
  originY: number;
  mapSize: number;
}

function gridStackLayout(numMaps: number, gridSize: number, cellPx: number, gapPx: number, centerX: number): MapSlot[] {
  const mapSize = gridSize * cellPx;
  const totalH = numMaps * mapSize + (numMaps - 1) * gapPx;
  const startY = CENTER_Y - totalH / 2;
  return Array.from({ length: numMaps }, (_, i) => ({
    originX: centerX - mapSize / 2,
    originY: startY + i * (mapSize + gapPx),
    mapSize,
  }));
}

function maxAbsOf(grid: Grid): number {
  let m = 1e-9;
  for (const row of grid) for (const cell of row) for (const v of cell) m = Math.max(m, Math.abs(v));
  return m;
}

function GridStage({ grid, cellPx, gapPx, centerX, revealed, delay }: { grid: Grid; cellPx: number; gapPx: number; centerX: number; revealed: boolean; delay: number }) {
  const gridSize = grid.length;
  const numMaps = grid[0][0].length;
  const slots = useMemo(() => gridStackLayout(numMaps, gridSize, cellPx, gapPx, centerX), [numMaps, gridSize, cellPx, gapPx, centerX]);
  const maxAbs = useMemo(() => maxAbsOf(grid), [grid]);

  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: revealed ? 1 : 0, scale: revealed ? 1 : 0.85 }}
      transition={{ duration: 0.45, delay, ease: "backOut" }}
    >
      {slots.map((slot, mi) => (
        <g key={mi}>
          <rect x={slot.originX - 2} y={slot.originY - 2} width={slot.mapSize + 4} height={slot.mapSize + 4} rx={2} fill="none" stroke="#d8dccb" strokeWidth={1} />
          {grid.map((row, r) =>
            row.map((cell, c) => (
              <rect
                key={`${r}-${c}`}
                x={slot.originX + c * cellPx}
                y={slot.originY + r * cellPx}
                width={cellPx}
                height={cellPx}
                fill={heatColor(Math.abs(cell[mi]) / maxAbs)}
              />
            ))
          )}
        </g>
      ))}
    </motion.g>
  );
}

function FlattenStage({ values, centerX, revealed, delay }: { values: number[]; centerX: number; revealed: boolean; delay: number }) {
  const cols = 4;
  const rows = Math.ceil(values.length / cols);
  const cellPx = 14;
  const gapPx = 2;
  const width = cols * cellPx + (cols - 1) * gapPx;
  const height = rows * cellPx + (rows - 1) * gapPx;
  const originX = centerX - width / 2;
  const originY = CENTER_Y - height / 2;
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9);

  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: revealed ? 1 : 0, scale: revealed ? 1 : 0.85 }}
      transition={{ duration: 0.45, delay, ease: "backOut" }}
    >
      {values.map((v, i) => {
        const r = Math.floor(i / cols);
        const c = i % cols;
        return (
          <rect
            key={i}
            x={originX + c * (cellPx + gapPx)}
            y={originY + r * (cellPx + gapPx)}
            width={cellPx}
            height={cellPx}
            rx={1.5}
            fill={heatColor(Math.abs(v) / maxAbs)}
          />
        );
      })}
    </motion.g>
  );
}

function NodeColumn({
  values,
  centerX,
  spacing,
  radius,
  revealed,
  delay,
  colorFn,
  labelFn,
  highlightIndex,
}: {
  values: number[];
  centerX: number;
  spacing: number;
  radius: number | ((v: number) => number);
  revealed: boolean;
  delay: number;
  colorFn: (v: number, i: number) => { fill: string; opacity: number };
  labelFn?: (i: number) => string;
  highlightIndex?: number;
}) {
  const totalH = (values.length - 1) * spacing;
  const startY = CENTER_Y - totalH / 2;

  return (
    <motion.g initial={{ opacity: 0 }} animate={{ opacity: revealed ? 1 : 0 }} transition={{ duration: 0.4, delay }}>
      {values.map((v, i) => {
        const cy = startY + i * spacing;
        const r = typeof radius === "function" ? radius(v) : radius;
        const style = colorFn(v, i);
        const isHighlight = highlightIndex === i;
        return (
          <g key={i}>
            <motion.circle
              cx={centerX}
              cy={cy}
              r={r}
              fill={style.fill}
              opacity={style.opacity}
              initial={{ scale: 0 }}
              animate={{ scale: revealed ? 1 : 0 }}
              transition={{ duration: 0.35, delay: delay + i * 0.03, ease: "backOut" }}
              style={{ transformOrigin: `${centerX}px ${cy}px` }}
            />
            {isHighlight && revealed && (
              <motion.circle
                cx={centerX}
                cy={cy}
                r={r + 4}
                fill="none"
                stroke="#2f6b4f"
                strokeWidth={2}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.35, delay: delay + 0.3 }}
              />
            )}
            {labelFn && (
              <text x={centerX} y={cy} textAnchor="middle" dominantBaseline="central" fontSize={10} fontWeight={600} fill="#ffffff" pointerEvents="none">
                {labelFn(i)}
              </text>
            )}
          </g>
        );
      })}
    </motion.g>
  );
}

export function CnnAnimation() {
  const cnnWeights = useSylvaStore((s) => s.cnnWeights);
  const dataset = useSylvaStore((s) => s.dataset);
  const setPage = useSylvaStore((s) => s.setPage);

  const [queryIndex, setQueryIndex] = useState(-1);
  const [stage, setStage] = useState(-1);
  const [grown, setGrown] = useState(false);

  const scenes = useMemo(() => {
    if (!cnnWeights || !dataset) return [];
    return dataset.test.slice(0, NUM_QUERIES).map((point) => ({ point, fwd: forwardCnn(cnnWeights, point.features) }));
  }, [cnnWeights, dataset]);

  useEffect(() => {
    if (scenes.length === 0) return;
    setQueryIndex(-1);
    setStage(-1);
    setGrown(false);
    const timers: ReturnType<typeof setTimeout>[] = [];
    const cycleLength = 300 + STAGE_LABELS.length * STAGE_GAP_MS + HOLD_MS;

    for (let q = 0; q < scenes.length; q++) {
      const base = q * cycleLength;
      timers.push(setTimeout(() => setQueryIndex(q), base + 200));
      for (let s = 0; s < STAGE_LABELS.length; s++) {
        timers.push(setTimeout(() => setStage(s), base + 300 + s * STAGE_GAP_MS));
      }
    }
    timers.push(setTimeout(() => setGrown(true), scenes.length * cycleLength + 300));
    return () => timers.forEach(clearTimeout);
  }, [scenes.length]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  if (!cnnWeights || scenes.length === 0 || queryIndex < 0) {
    return <div className="fixed inset-0 bg-bg" />;
  }

  const scene = scenes[queryIndex];
  const fwd: CnnForward = scene.fwd;
  const inputThumb = pixelsToDataUrl(scene.point.raw, RAW_SIZE, 5);
  const correct = fwd.predicted === scene.point.label;

  const anchors = [
    STAGE_X.input,
    STAGE_X.conv1,
    STAGE_X.pool1,
    STAGE_X.conv2,
    STAGE_X.pool2,
    STAGE_X.flatten,
    STAGE_X.dense,
    STAGE_X.output,
  ];

  return (
    <div className="fixed inset-0 overflow-hidden bg-bg">
      <div className="absolute left-1/2 top-8 z-10 -translate-x-1/2 rounded-full bg-amber-soft px-4 py-1.5 text-[12px] font-medium uppercase tracking-wide text-amber">
        Pre-trained · for fun, not ranked
      </div>

      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        <defs>
          <radialGradient id="cnnVignette" cx="50%" cy="48%" r="75%">
            <stop offset="55%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="100%" stopColor="#1d2a1f" stopOpacity="0.045" />
          </radialGradient>
        </defs>
        <rect x={0} y={0} width={VIEW_W} height={VIEW_H} fill="url(#cnnVignette)" pointerEvents="none" />

        <AnimatePresence mode="wait">
          <g key={`scene-${queryIndex}`}>
            {/* connecting flow-line, drawn behind every stage */}
            {anchors.slice(0, -1).map((x, i) => (
              <motion.line
                key={i}
                x1={x + 34}
                y1={CENTER_Y}
                x2={anchors[i + 1] - 34}
                y2={CENTER_Y}
                stroke="#b9c2b2"
                strokeWidth={1.6}
                strokeDasharray="4 5"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: stage > i ? 1 : 0, opacity: stage > i ? 0.7 : 0 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
              />
            ))}

            {/* stage 0: the digit itself */}
            <motion.g
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: stage >= 0 ? 1 : 0, scale: stage >= 0 ? 1 : 0.8 }}
              transition={{ duration: 0.4, ease: "backOut" }}
            >
              <rect x={STAGE_X.input - 37} y={CENTER_Y - 37} width={74} height={74} rx={4} fill="#ffffff" style={{ filter: "drop-shadow(0 1px 3px rgba(29,42,31,0.18))" }} />
              <image href={inputThumb} x={STAGE_X.input - 35} y={CENTER_Y - 35} width={70} height={70} style={{ imageRendering: "pixelated" }} />
            </motion.g>

            {/* scanning window flourish while conv stages are being revealed */}
            {(stage === 1 || stage === 3) && (
              <motion.rect
                width={stage === 1 ? 15 : 12}
                height={stage === 1 ? 15 : 12}
                fill="none"
                stroke="#2f6b4f"
                strokeWidth={1.6}
                rx={2}
                initial={{ opacity: 0 }}
                animate={{
                  opacity: [0, 0.9, 0.9, 0],
                  x: stage === 1 ? [STAGE_X.input - 35, STAGE_X.input - 5, STAGE_X.input + 20, STAGE_X.input + 20] : [STAGE_X.pool1 - 18, STAGE_X.pool1, STAGE_X.pool1 + 10, STAGE_X.pool1 + 10],
                  y: stage === 1 ? [CENTER_Y - 35, CENTER_Y - 10, CENTER_Y + 15, CENTER_Y + 15] : [CENTER_Y - 18, CENTER_Y, CENTER_Y + 8, CENTER_Y + 8],
                }}
                transition={{ duration: STAGE_GAP_MS / 1000, ease: "easeInOut" }}
              />
            )}

            <GridStage grid={fwd.conv1} cellPx={3.4} gapPx={6} centerX={STAGE_X.conv1} revealed={stage >= 1} delay={0} />
            <GridStage grid={fwd.pool1} cellPx={6} gapPx={7} centerX={STAGE_X.pool1} revealed={stage >= 2} delay={0} />
            <GridStage grid={fwd.conv2} cellPx={3.8} gapPx={5} centerX={STAGE_X.conv2} revealed={stage >= 3} delay={0} />
            <GridStage grid={fwd.pool2} cellPx={7} gapPx={6} centerX={STAGE_X.pool2} revealed={stage >= 4} delay={0} />
            <FlattenStage values={fwd.pool2.flat(2)} centerX={STAGE_X.flatten} revealed={stage >= 5} delay={0} />

            <NodeColumn
              values={fwd.dense1}
              centerX={STAGE_X.dense}
              spacing={20}
              radius={6}
              revealed={stage >= 6}
              delay={0}
              colorFn={(v) => {
                const max = Math.max(...fwd.dense1, 1e-9);
                return { fill: heatColor(v / max), opacity: 1 };
              }}
            />

            <NodeColumn
              values={fwd.probs}
              centerX={STAGE_X.output}
              spacing={26}
              radius={(v) => 7 + v * 11}
              revealed={stage >= 7}
              delay={0.1}
              colorFn={(v, i) => ({ fill: classColor(i), opacity: 0.3 + v * 0.7 })}
              labelFn={(i) => String(i)}
              highlightIndex={fwd.predicted}
            />

            {stage >= 7 && (
              <motion.g initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.4 }}>
                <text x={STAGE_X.output} y={CENTER_Y - ((NUM_CLASSES - 1) * 26) / 2 - 34} textAnchor="middle" fontSize={14} fontWeight={700} fill={correct ? "#2f6b4f" : "#b8473d"}>
                  guesses {fwd.predicted}
                </text>
              </motion.g>
            )}

            {STAGE_LABELS.map((label, i) => (
              <text key={i} x={anchors[i]} y={VIEW_H - 34} textAnchor="middle" fontSize={12} fontWeight={500} fill={stage >= i ? "#55604f" : "#c3c9b8"}>
                {label}
              </text>
            ))}
          </g>
        </AnimatePresence>
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
