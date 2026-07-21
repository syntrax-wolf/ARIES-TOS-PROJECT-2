import { useEffect, useMemo, useRef, useState } from "react";
import { StepHeader } from "../components/StepHeader";
import { Card } from "../components/Card";
import { Button } from "../components/Button";
import { useSylvaStore } from "../state/store";
import { featureImportance, predict } from "../lib/decisionTree";
import type { Point } from "../lib/dataset";
import { forestFeatureImportance, predictForest } from "../lib/randomForest";
import { evaluate } from "../lib/metrics";
import { heatColor } from "../lib/palette";
import { pixelsToDataUrl, FEATURE_PIXELS, FEATURE_SIZE, RAW_SIZE } from "../lib/mnist";
import { addLeaderboardEntry, getLeaderboard, rankOf, type LeaderboardEntry } from "../lib/leaderboard";

const HEAT_CELL = 20;
const GALLERY_COUNT = 12;

function PixelHeatmap({ importance }: { importance: Float64Array }) {
  const size = FEATURE_SIZE * HEAT_CELL;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="rounded-lg">
      {Array.from({ length: FEATURE_SIZE }).map((_, row) =>
        Array.from({ length: FEATURE_SIZE }).map((_, col) => {
          const t = importance[row * FEATURE_SIZE + col];
          return (
            <rect key={`${row}-${col}`} x={col * HEAT_CELL} y={row * HEAT_CELL} width={HEAT_CELL} height={HEAT_CELL} fill={heatColor(t)} />
          );
        })
      )}
    </svg>
  );
}

function SampleGallery({ predictOne }: { predictOne: (p: Point) => number }) {
  const dataset = useSylvaStore((s) => s.dataset);

  const samples = useMemo(() => {
    if (!dataset) return [];
    return dataset.test.slice(0, GALLERY_COUNT).map((p) => ({
      src: pixelsToDataUrl(p.raw, RAW_SIZE, 2),
      actual: p.label,
      predicted: predictOne(p),
    }));
  }, [dataset, predictOne]);

  return (
    <div className="grid grid-cols-6 gap-2">
      {samples.map((s, i) => {
        const correct = s.predicted === s.actual;
        return (
          <div key={i} className="flex flex-col items-center gap-1">
            <img
              src={s.src}
              alt=""
              className="aspect-square w-full rounded-md"
              style={{ boxShadow: `0 0 0 2px ${correct ? "#2f6b4f" : "#b8473d"}` }}
            />
            <span className={"text-[11px] font-medium tabular-nums " + (correct ? "text-brand-dark" : "text-danger")}>
              {s.predicted}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function Results() {
  const algorithm = useSylvaStore((s) => s.algorithm);
  const dataset = useSylvaStore((s) => s.dataset);
  const tree = useSylvaStore((s) => s.tree);
  const forest = useSylvaStore((s) => s.forest);
  const hyperparams = useSylvaStore((s) => s.hyperparams);
  const setPage = useSylvaStore((s) => s.setPage);
  const restart = useSylvaStore((s) => s.restart);

  const isForest = algorithm === "random-forest";
  const model = isForest ? forest : tree;

  const [board, setBoard] = useState<LeaderboardEntry[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const addedRef = useRef(false);

  const predictOne = useMemo(() => {
    if (isForest && forest) return (p: Point) => predictForest(forest, p);
    if (tree) return (p: Point) => predict(tree.root, p);
    return null;
  }, [isForest, forest, tree]);

  const importance = useMemo(() => {
    if (isForest && forest) return forestFeatureImportance(forest, FEATURE_PIXELS);
    if (tree) return featureImportance(tree, FEATURE_PIXELS);
    return null;
  }, [isForest, forest, tree]);

  const evaluation = useMemo(() => {
    if (!dataset || !predictOne) return null;
    const predictions = dataset.test.map(predictOne);
    return evaluate(dataset.test, predictions, dataset.classes);
  }, [dataset, predictOne]);

  useEffect(() => {
    if (addedRef.current || !model || !evaluation) return;
    addedRef.current = true;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const entry: LeaderboardEntry = isForest && forest
      ? {
          id,
          timestamp: Date.now(),
          algorithm: "Random Forest",
          accuracy: evaluation.accuracy,
          macroF1: evaluation.macroF1,
          depth: Math.round((forest.trees.reduce((s, t) => s + t.depth, 0) / forest.trees.length) * 10) / 10,
          leafCount: forest.trees.reduce((s, t) => s + t.leafCount, 0),
          summary: `${hyperparams.numTrees} trees, max depth ${hyperparams.maxDepth}`,
        }
      : {
          id,
          timestamp: Date.now(),
          algorithm: "Decision Tree",
          accuracy: evaluation.accuracy,
          macroF1: evaluation.macroF1,
          depth: tree!.depth,
          leafCount: tree!.leafCount,
          summary: `max depth ${hyperparams.maxDepth}, ${tree!.leafCount} leaves`,
        };

    const updated = addLeaderboardEntry(entry);
    setBoard(updated);
    setCurrentId(id);
  }, [model, evaluation, isForest, forest, tree, hyperparams.maxDepth, hyperparams.numTrees]);

  useEffect(() => {
    if (board.length === 0) setBoard(getLeaderboard());
  }, [board.length]);

  if (!dataset || !model || !evaluation || !predictOne || !importance) {
    return (
      <div className="min-h-screen">
        <StepHeader step={4} />
        <div className="px-8 py-20 text-center text-ink-soft">No trained model yet — go back and train one first.</div>
      </div>
    );
  }

  const rank = currentId ? rankOf(board, currentId) : null;

  const stats = isForest && forest
    ? [
        { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
        { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
        { label: "Trees", value: String(forest.trees.length) },
        { label: "Avg tree depth", value: (forest.trees.reduce((s, t) => s + t.depth, 0) / forest.trees.length).toFixed(1) },
      ]
    : [
        { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
        { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
        { label: "Tree depth", value: String(tree!.depth) },
        { label: "Leaf nodes", value: String(tree!.leafCount) },
      ];

  return (
    <div className="min-h-screen">
      <StepHeader step={4} />

      <main className="mx-auto max-w-5xl px-6 pb-24 pt-2 sm:px-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl text-ink sm:text-[38px]">Your {isForest ? "forest" : "tree"}, evaluated</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            Performance on handwritten digits {isForest ? "the forest" : "the tree"} never saw during training.
          </p>
        </div>

        <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {stats.map((stat) => (
            <Card key={stat.label} className="p-5">
              <div className="text-2xl font-semibold text-brand-dark">{stat.value}</div>
              <div className="mt-1 text-sm text-ink-soft">{stat.label}</div>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[auto_1fr]">
          <div className="flex flex-col gap-6">
            <Card className="p-6">
              <h2 className="mb-1 text-lg font-semibold text-ink">Pixels that matter</h2>
              <p className="mb-4 text-sm text-ink-soft">
                Where on the digit {isForest ? "the forest" : "the tree"} chose to look.
              </p>
              <PixelHeatmap importance={importance} />
            </Card>

            <Card className="p-6">
              <h2 className="mb-1 text-lg font-semibold text-ink">Test predictions</h2>
              <p className="mb-4 text-sm text-ink-soft">Green ring: correct. Red ring: missed.</p>
              <SampleGallery predictOne={predictOne} />
            </Card>
          </div>

          <div className="flex flex-col gap-6">
            <Card className="p-6">
              <h2 className="mb-4 text-lg font-semibold text-ink">Confusion matrix</h2>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-xs">
                  <thead>
                    <tr>
                      <th className="p-1.5 text-left font-medium text-ink-faint"> </th>
                      {dataset.classes.map((c) => (
                        <th key={c} className="p-1.5 text-center font-medium text-ink-faint">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.classes.map((actual) => (
                      <tr key={actual}>
                        <td className="p-1.5 font-medium text-ink-faint">{actual}</td>
                        {dataset.classes.map((pred) => {
                          const count = evaluation.confusion.get(actual)?.get(pred) ?? 0;
                          return (
                            <td
                              key={pred}
                              className={
                                "p-1.5 text-center tabular-nums " +
                                (actual === pred
                                  ? "rounded-md bg-brand-soft font-semibold text-brand-dark"
                                  : count > 0
                                    ? "text-ink-soft"
                                    : "text-ink-faint/50")
                              }
                            >
                              {count}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card className="flex-1 p-6">
              <div className="mb-4 flex items-baseline justify-between">
                <h2 className="text-lg font-semibold text-ink">Leaderboard</h2>
                {rank && (
                  <span className="text-sm font-medium text-brand-dark">
                    You placed #{rank} of {board.length}
                  </span>
                )}
              </div>
              <div className="max-h-64 overflow-y-auto">
                <table className="w-full border-collapse text-sm">
                  <tbody>
                    {board.slice(0, 12).map((entry, i) => (
                      <tr
                        key={entry.id}
                        className={entry.id === currentId ? "bg-brand-soft/70" : i % 2 === 1 ? "bg-surface-soft/60" : ""}
                      >
                        <td className="rounded-l-md py-2 pl-2 pr-1 font-medium text-ink-faint">#{i + 1}</td>
                        <td className="py-2 pr-2 font-semibold tabular-nums text-ink">
                          {(entry.accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-2 text-ink-faint">{entry.algorithm}</td>
                        <td className="rounded-r-md py-2 pr-2 text-ink-faint">{entry.summary}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>

        <div className="mt-10 flex items-center justify-between">
          <Button variant="ghost" onClick={restart}>
            ← Choose another algorithm
          </Button>
          <Button variant="secondary" size="lg" onClick={() => setPage("hyperparams")}>
            Try different hyperparameters
          </Button>
        </div>
      </main>
    </div>
  );
}
