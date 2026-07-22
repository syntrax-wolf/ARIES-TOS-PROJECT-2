import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSylvaStore } from "../state/store";
import { boostingScores } from "../lib/gradientBoosting";
import { regressionBfsOrder } from "../lib/regressionTree";
import { classColor } from "../lib/palette";

const SIZE = 820;
const GROUND_Y = 560;
const TRUNK_X = SIZE / 2;
const TRUNK_TOP_Y = 330;
const CANOPY_RADIUS = 96;
const NURSERY_Y = GROUND_Y - 4;
const NURSERY_MARGIN = 70;
const SAMPLE_COUNT = 60;

type Phase = "sprout" | "combine" | "canopy" | "grown";

interface Sapling {
  x: number;
  leafCount: number;
}

export function GradientBoostingAnimation() {
  const boosting = useSylvaStore((s) => s.boosting);
  const dataset = useSylvaStore((s) => s.dataset);
  const setPage = useSylvaStore((s) => s.setPage);

  const [sproutIndex, setSproutIndex] = useState(-1);
  const [phase, setPhase] = useState<Phase>("sprout");

  const numRounds = boosting?.rounds.length ?? 0;

  const saplings = useMemo<Sapling[]>(() => {
    if (!boosting) return [];
    const usableWidth = SIZE - NURSERY_MARGIN * 2;
    return boosting.rounds.map((round, i) => {
      const representative = round.trees[i % round.trees.length];
      const leaves = regressionBfsOrder(representative.root).filter((n) => n.isLeaf).length;
      const x = numRounds === 1 ? SIZE / 2 : NURSERY_MARGIN + (usableWidth * i) / (numRounds - 1);
      return { x, leafCount: Math.max(1, Math.min(4, leaves)) };
    });
  }, [boosting, numRounds]);

  /** Average magnitude of each class's boosted score across a sample of real training points — genuine per-digit "how strongly did the ensemble learn this" signal, not decorative. */
  const canopyStats = useMemo(() => {
    if (!boosting || !dataset) return null;
    const sample = dataset.train.slice(0, SAMPLE_COUNT);
    const sums = new Array(10).fill(0);
    for (const p of sample) {
      const scores = boostingScores(boosting, p);
      for (let k = 0; k < 10; k++) sums[k] += Math.abs(scores[k]);
    }
    const means = sums.map((s) => s / Math.max(sample.length, 1));
    const max = Math.max(...means, 1e-6);
    return means.map((m) => m / max);
  }, [boosting, dataset]);

  useEffect(() => {
    if (numRounds === 0) return;
    setSproutIndex(-1);
    setPhase("sprout");
    const perSprout = Math.max(0.22, Math.min(0.55, 3.6 / (numRounds + 1)));
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 0; i < numRounds; i++) {
      timers.push(setTimeout(() => setSproutIndex(i), 300 + i * perSprout * 1000));
    }
    const sproutEnd = 300 + numRounds * perSprout * 1000;
    timers.push(setTimeout(() => setPhase("combine"), sproutEnd + 250));
    timers.push(setTimeout(() => setPhase("canopy"), sproutEnd + 1550));
    timers.push(setTimeout(() => setPhase("grown"), sproutEnd + 2650));
    return () => timers.forEach(clearTimeout);
  }, [numRounds]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  if (!boosting || saplings.length === 0) return <div className="fixed inset-0 bg-bg" />;

  const combining = phase === "combine" || phase === "canopy" || phase === "grown";
  const showCanopy = phase === "canopy" || phase === "grown";
  const grown = phase === "grown";

  return (
    <div className="fixed inset-0 overflow-hidden bg-sky">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        <defs>
          <linearGradient id="gbSkyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f2f7f1" />
            <stop offset="100%" stopColor="#e9f1ea" />
          </linearGradient>
          <linearGradient id="gbSoilGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ece2cd" />
            <stop offset="100%" stopColor="#ddcba3" />
          </linearGradient>
          <filter id="gbShadow" x="-60%" y="-60%" width="220%" height="220%">
            <feDropShadow dx="0" dy="1.5" stdDeviation="2" floodColor="#1d2a1f" floodOpacity="0.18" />
          </filter>
          <radialGradient id="gbVignette" cx="50%" cy="34%" r="75%">
            <stop offset="58%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="100%" stopColor="#1d2a1f" stopOpacity="0.05" />
          </radialGradient>
        </defs>

        <rect x={0} y={0} width={SIZE} height={GROUND_Y} fill="url(#gbSkyGrad)" />
        <rect x={0} y={GROUND_Y} width={SIZE} height={SIZE - GROUND_Y} fill="url(#gbSoilGrad)" />
        <line x1={0} y1={GROUND_Y} x2={SIZE} y2={GROUND_Y} stroke="#c9b98c" strokeWidth={1.5} opacity={0.6} />

        {/* the trunk: a stub that grows into the full trunk once the saplings start combining */}
        <motion.line
          x1={TRUNK_X}
          y1={GROUND_Y}
          x2={TRUNK_X}
          initial={{ y2: GROUND_Y - 26 }}
          animate={{ y2: combining ? TRUNK_TOP_Y : GROUND_Y - 26 }}
          transition={{ duration: 1.1, ease: "easeInOut" }}
          stroke="#7c5f3c"
          strokeWidth={9}
          strokeLinecap="round"
        />

        {/* nursery row: each weak learner sprouts as its own small sapling */}
        <AnimatePresence>
          {!combining &&
            saplings.map((s, i) => (
              <motion.g
                key={`sapling-${i}`}
                initial={{ opacity: 0, scale: 0.4 }}
                animate={i <= sproutIndex ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.4 }}
                exit={{ opacity: 0, scale: 0.5, transition: { duration: 0.4 } }}
                transition={{ duration: 0.45, ease: "backOut" }}
                style={{ transformOrigin: `${s.x}px ${NURSERY_Y}px` }}
              >
                <SaplingGlyph x={s.x} baseY={NURSERY_Y} leafCount={s.leafCount} active={i <= sproutIndex} />
              </motion.g>
            ))}
        </AnimatePresence>

        {/* combine: every sapling drifts up into the trunk top and dissolves */}
        <AnimatePresence>
          {phase === "combine" &&
            saplings.map((s, i) => (
              <motion.g
                key={`merge-${i}`}
                initial={{ opacity: 1, x: 0, y: 0, scale: 1 }}
                animate={{ opacity: 0, x: TRUNK_X - s.x, y: TRUNK_TOP_Y - NURSERY_Y + 20, scale: 0.25 }}
                transition={{ duration: 1.0, delay: i * 0.03, ease: "easeIn" }}
              >
                <SaplingGlyph x={s.x} baseY={NURSERY_Y} leafCount={s.leafCount} active />
              </motion.g>
            ))}
        </AnimatePresence>

        {/* final canopy: one leaf cluster per digit class, sized by how strongly the trained ensemble responds to that class */}
        {showCanopy && canopyStats && (
          <g>
            <motion.circle
              cx={TRUNK_X}
              cy={TRUNK_TOP_Y}
              r={10}
              fill="#7c5f3c"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3 }}
              style={{ transformOrigin: `${TRUNK_X}px ${TRUNK_TOP_Y}px` }}
            />
            {canopyStats.map((t, k) => {
              const angle = (k / canopyStats.length) * Math.PI * 2 - Math.PI / 2;
              const r = 8 + t * 15;
              const cx = TRUNK_X + CANOPY_RADIUS * Math.cos(angle);
              const cy = TRUNK_TOP_Y + CANOPY_RADIUS * 0.72 * Math.sin(angle) - 18;
              return (
                <motion.g key={k}>
                  <motion.line
                    x1={TRUNK_X}
                    y1={TRUNK_TOP_Y}
                    x2={cx}
                    y2={cy}
                    stroke="#8a9484"
                    strokeWidth={1.4}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.4 }}
                    transition={{ duration: 0.5, delay: k * 0.05, ease: "easeOut" }}
                  />
                  <motion.circle
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill={classColor(k)}
                    filter="url(#gbShadow)"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 0.55 + t * 0.45 }}
                    transition={{ duration: 0.4, delay: k * 0.05 + 0.15, ease: "backOut" }}
                    style={{ transformOrigin: `${cx}px ${cy}px` }}
                  />
                </motion.g>
              );
            })}
          </g>
        )}

        <rect x={0} y={0} width={SIZE} height={SIZE} fill="url(#gbVignette)" pointerEvents="none" />
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

function SaplingGlyph({ x, baseY, leafCount, active }: { x: number; baseY: number; leafCount: number; active: boolean }) {
  const height = 24 + leafCount * 6;
  const topY = baseY - height;
  return (
    <g>
      <motion.line
        x1={x}
        y1={baseY}
        x2={x}
        y2={topY}
        stroke="#5c7a5f"
        strokeWidth={2.4}
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: active ? 1 : 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      />
      {Array.from({ length: leafCount }).map((_, i) => {
        const t = (i + 1) / (leafCount + 0.4);
        const ly = baseY - height * t;
        const side = i % 2 === 0 ? 1 : -1;
        const lx = x + side * (5 + t * 6);
        return (
          <motion.ellipse
            key={i}
            cx={lx}
            cy={ly}
            rx={7}
            ry={4.2}
            fill="#6f9a6f"
            transform={`rotate(${side * 35} ${lx} ${ly})`}
            initial={{ scale: 0, opacity: 0 }}
            animate={active ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
            transition={{ duration: 0.35, delay: 0.15 + i * 0.08, ease: "backOut" }}
            style={{ transformOrigin: `${lx}px ${ly}px` }}
          />
        );
      })}
    </g>
  );
}
