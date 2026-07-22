import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "../state/store";
import { kNearest, predict, type Neighbor } from "../lib/knn";
import type { Point } from "../lib/dataset";
import { pixelsToDataUrl, RAW_SIZE } from "../lib/mnist";
import { classColor } from "../lib/palette";

const NUM_QUERIES = 5;
const CYCLE_MS = 2400;
const SIZE = 760;
const CENTER = SIZE / 2;
const RING_RADIUS = 230;
const FAR_RADIUS = 430;
const QUERY_R = 62;
const NEIGHBOR_MAX_R = 32;
const NEIGHBOR_MIN_R = 21;

interface Scene {
  query: Point;
  neighbors: Neighbor[];
  predicted: number;
  correct: boolean;
}

export function KnnAnimation() {
  const knn = useSylvaStore((s) => s.knn);
  const dataset = useSylvaStore((s) => s.dataset);
  const setPage = useSylvaStore((s) => s.setPage);

  const [queryIndex, setQueryIndex] = useState(-1);
  const [grown, setGrown] = useState(false);

  const scenes = useMemo<Scene[]>(() => {
    if (!knn || !dataset) return [];
    return dataset.test.slice(0, NUM_QUERIES).map((query) => {
      const neighbors = kNearest(knn, query.features, knn.k);
      const predicted = predict(knn, query);
      return { query, neighbors, predicted, correct: predicted === query.label };
    });
  }, [knn, dataset]);

  useEffect(() => {
    if (scenes.length === 0) return;
    setQueryIndex(-1);
    setGrown(false);
    const timers: ReturnType<typeof setTimeout>[] = [];
    timers.push(setTimeout(() => setQueryIndex(0), 300));
    for (let i = 1; i < scenes.length; i++) {
      timers.push(setTimeout(() => setQueryIndex(i), 300 + i * CYCLE_MS));
    }
    timers.push(setTimeout(() => setGrown(true), 300 + scenes.length * CYCLE_MS));
    return () => timers.forEach(clearTimeout);
  }, [scenes.length]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  if (!knn || scenes.length === 0 || queryIndex < 0) {
    return <div className="fixed inset-0 bg-bg" />;
  }

  const scene = scenes[queryIndex];
  const k = scene.neighbors.length;
  const stagger = Math.min(0.09, 0.9 / Math.max(k, 1));
  const distances = scene.neighbors.map((n) => n.distance);
  const minD = Math.min(...distances);
  const maxD = Math.max(...distances, minD + 1e-6);
  const settleTime = 0.25 + (k - 1) * stagger + 0.7;

  const queryThumb = pixelsToDataUrl(scene.query.raw, RAW_SIZE, 3);

  return (
    <div className="fixed inset-0 overflow-hidden bg-bg">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        <defs>
          <radialGradient id="knnVignette" cx="50%" cy="48%" r="72%">
            <stop offset="55%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="100%" stopColor="#1d2a1f" stopOpacity="0.045" />
          </radialGradient>
          <filter id="knnShadow" x="-60%" y="-60%" width="220%" height="220%">
            <feDropShadow dx="0" dy="1.5" stdDeviation="2.2" floodColor="#1d2a1f" floodOpacity="0.2" />
          </filter>
        </defs>

        <rect x={0} y={0} width={SIZE} height={SIZE} fill="url(#knnVignette)" pointerEvents="none" />

        <AnimatePresence mode="wait">
          <motion.g key={`scene-${queryIndex}`}>
            {scene.neighbors.map((n, i) => {
              const angle = (i / k) * Math.PI * 2 - Math.PI / 2;
              const norm = (n.distance - minD) / (maxD - minD || 1);
              const closeness = 1 - norm;
              const ringX = CENTER + RING_RADIUS * Math.cos(angle);
              const ringY = CENTER + RING_RADIUS * Math.sin(angle);
              const delay = 0.25 + i * stagger;
              const r = NEIGHBOR_MIN_R + closeness * (NEIGHBOR_MAX_R - NEIGHBOR_MIN_R);
              const thumb = pixelsToDataUrl(n.point.raw, RAW_SIZE, 3);

              return (
                <g key={`n-${i}`}>
                  <motion.line
                    x1={CENTER}
                    y1={CENTER}
                    x2={ringX}
                    y2={ringY}
                    stroke="#8a9484"
                    strokeLinecap="round"
                    strokeWidth={0.8 + closeness * 2.2}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.3 + closeness * 0.45 }}
                    transition={{ duration: 0.6, delay, ease: "easeOut" }}
                  />
                  <motion.g
                    initial={{
                      x: CENTER + FAR_RADIUS * Math.cos(angle) - ringX,
                      y: CENTER + FAR_RADIUS * Math.sin(angle) - ringY,
                      opacity: 0,
                      scale: 0.55,
                    }}
                    animate={{ x: 0, y: 0, opacity: 1, scale: 1 }}
                    transition={{ duration: 0.75, delay, ease: "easeOut" }}
                  >
                    <clipPath id={`knn-clip-${i}`}>
                      <circle cx={ringX} cy={ringY} r={r} />
                    </clipPath>
                    <circle cx={ringX} cy={ringY} r={r + 2.5} fill={classColor(n.point.label)} filter="url(#knnShadow)" />
                    <image
                      href={thumb}
                      x={ringX - r}
                      y={ringY - r}
                      width={r * 2}
                      height={r * 2}
                      clipPath={`url(#knn-clip-${i})`}
                      style={{ imageRendering: "pixelated" }}
                    />
                  </motion.g>
                </g>
              );
            })}

            {/* vote tally: one dot per neighbor, closest first, colored by that neighbor's digit */}
            <motion.g
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: settleTime + 0.15 }}
            >
              {scene.neighbors.map((n, i) => (
                <circle
                  key={`vote-${i}`}
                  cx={CENTER - (k - 1) * 5 + i * 10}
                  cy={CENTER + QUERY_R + 26}
                  r={4}
                  fill={classColor(n.point.label)}
                />
              ))}
            </motion.g>

            {/* the query itself, with a neutral ring that resolves to green/red once the verdict lands */}
            <clipPath id="knn-query-clip">
              <circle cx={CENTER} cy={CENTER} r={QUERY_R} />
            </clipPath>
            <motion.circle
              cx={CENTER}
              cy={CENTER}
              r={QUERY_R + 4}
              fill="none"
              initial={{ opacity: 0, scale: 0.7, stroke: "#8a9484" }}
              animate={{
                opacity: 1,
                scale: 1,
                stroke: scene.correct ? "#2f6b4f" : "#b8473d",
              }}
              transition={{ opacity: { duration: 0.4 }, scale: { duration: 0.4 }, stroke: { duration: 0.4, delay: settleTime + 0.35 } }}
              strokeWidth={4}
            />
            <motion.circle
              cx={CENTER}
              cy={CENTER}
              r={QUERY_R}
              fill="#ffffff"
              filter="url(#knnShadow)"
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
            />
            <motion.image
              href={queryThumb}
              x={CENTER - QUERY_R}
              y={CENTER - QUERY_R}
              width={QUERY_R * 2}
              height={QUERY_R * 2}
              clipPath="url(#knn-query-clip)"
              style={{ imageRendering: "pixelated" }}
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
            />

            <motion.g
              initial={{ opacity: 0, scale: 0.4 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.35, delay: settleTime + 0.4, ease: "backOut" }}
              style={{ transformOrigin: `${CENTER + QUERY_R * 0.72}px ${CENTER + QUERY_R * 0.72}px` }}
            >
              <circle cx={CENTER + QUERY_R * 0.72} cy={CENTER + QUERY_R * 0.72} r={16} fill={classColor(scene.predicted)} filter="url(#knnShadow)" />
              <text
                x={CENTER + QUERY_R * 0.72}
                y={CENTER + QUERY_R * 0.72}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={14}
                fontWeight={700}
                fill="#ffffff"
              >
                {scene.predicted}
              </text>
            </motion.g>
          </motion.g>
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
