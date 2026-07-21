import { useEffect, useMemo, useRef, useState } from "react";
import { StepHeader } from "../components/StepHeader";
import { Card } from "../components/Card";
import { Button } from "../components/Button";
import { useSylvaStore } from "../state/store";
import { predict } from "../lib/decisionTree";
import { evaluate } from "../lib/metrics";
import { classColor, classColorSoft } from "../lib/palette";
import { addLeaderboardEntry, getLeaderboard, rankOf, type LeaderboardEntry } from "../lib/leaderboard";

const CANVAS_SIZE = 420;
const GRID_RES = 90;

function DecisionBoundary() {
  const dataset = useSylvaStore((s) => s.dataset);
  const tree = useSylvaStore((s) => s.tree);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!dataset || !tree || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_SIZE * dpr;
    canvas.height = CANVAS_SIZE * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const { bounds } = dataset;
    const cell = CANVAS_SIZE / GRID_RES;

    for (let gx = 0; gx < GRID_RES; gx++) {
      for (let gy = 0; gy < GRID_RES; gy++) {
        const dataX = bounds.xMin + ((gx + 0.5) / GRID_RES) * (bounds.xMax - bounds.xMin);
        const dataY = bounds.yMax - ((gy + 0.5) / GRID_RES) * (bounds.yMax - bounds.yMin);
        const pred = predict(tree.root, { x: dataX, y: dataY, label: 0 });
        ctx.fillStyle = classColorSoft(pred);
        ctx.fillRect(gx * cell, gy * cell, cell + 0.5, cell + 0.5);
      }
    }

    const project = (x: number, y: number) => [
      ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * CANVAS_SIZE,
      CANVAS_SIZE - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * CANVAS_SIZE,
    ];

    for (const p of dataset.train) {
      const [px, py] = project(p.x, p.y);
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = classColor(p.label);
      ctx.globalAlpha = 0.28;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    for (const p of dataset.test) {
      const pred = predict(tree.root, p);
      const [px, py] = project(p.x, p.y);
      ctx.beginPath();
      ctx.arc(px, py, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = classColor(p.label);
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      if (pred !== p.label) {
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = "#b8473d";
        ctx.stroke();
      }
    }
  }, [dataset, tree]);

  return (
    <div className="overflow-hidden rounded-xl border border-border-soft bg-surface-soft">
      <canvas ref={canvasRef} style={{ width: CANVAS_SIZE, height: CANVAS_SIZE, display: "block" }} />
    </div>
  );
}

export function Results() {
  const dataset = useSylvaStore((s) => s.dataset);
  const tree = useSylvaStore((s) => s.tree);
  const hyperparams = useSylvaStore((s) => s.hyperparams);
  const datasetConfig = useSylvaStore((s) => s.datasetConfig);
  const setPage = useSylvaStore((s) => s.setPage);
  const restart = useSylvaStore((s) => s.restart);

  const [board, setBoard] = useState<LeaderboardEntry[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const addedRef = useRef(false);

  const evaluation = useMemo(() => {
    if (!dataset || !tree) return null;
    const predictions = dataset.test.map((p) => predict(tree.root, p));
    return evaluate(dataset.test, predictions, dataset.classes);
  }, [dataset, tree]);

  useEffect(() => {
    if (addedRef.current || !tree || !evaluation) return;
    addedRef.current = true;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const entry: LeaderboardEntry = {
      id,
      timestamp: Date.now(),
      algorithm: "Decision Tree",
      datasetShape: datasetConfig.shape,
      accuracy: evaluation.accuracy,
      macroF1: evaluation.macroF1,
      depth: tree.depth,
      leafCount: tree.leafCount,
      summary: `${hyperparams.criterion}, depth ${tree.depth}, ${tree.leafCount} leaves`,
    };
    const updated = addLeaderboardEntry(entry);
    setBoard(updated);
    setCurrentId(id);
  }, [tree, evaluation, datasetConfig.shape, hyperparams.criterion]);

  useEffect(() => {
    if (board.length === 0) setBoard(getLeaderboard());
  }, [board.length]);

  if (!dataset || !tree || !evaluation) {
    return (
      <div className="min-h-screen">
        <StepHeader step={4} />
        <div className="px-8 py-20 text-center text-ink-soft">No trained model yet — go back and train one first.</div>
      </div>
    );
  }

  const rank = currentId ? rankOf(board, currentId) : null;

  return (
    <div className="min-h-screen">
      <StepHeader step={4} />

      <main className="mx-auto max-w-5xl px-6 pb-24 pt-2 sm:px-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl text-ink sm:text-[38px]">Your tree, evaluated</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            Performance on the held-out test set your tree never saw during training.
          </p>
        </div>

        <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
            { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
            { label: "Tree depth", value: String(tree.depth) },
            { label: "Leaf nodes", value: String(tree.leafCount) },
          ].map((stat) => (
            <Card key={stat.label} className="p-5">
              <div className="text-2xl font-semibold text-brand-dark">{stat.value}</div>
              <div className="mt-1 text-sm text-ink-soft">{stat.label}</div>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[auto_1fr]">
          <Card className="p-6">
            <h2 className="mb-4 text-lg font-semibold text-ink">Decision boundary</h2>
            <DecisionBoundary />
            <div className="mt-4 flex flex-wrap items-center gap-4 text-xs text-ink-soft">
              {dataset.classes.map((c) => (
                <span key={c} className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: classColor(c) }} />
                  Class {c}
                </span>
              ))}
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full border-2 border-danger" />
                Misclassified
              </span>
            </div>
          </Card>

          <div className="flex flex-col gap-6">
            <Card className="p-6">
              <h2 className="mb-4 text-lg font-semibold text-ink">Confusion matrix</h2>
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    <th className="p-2 text-left font-medium text-ink-faint"> </th>
                    {dataset.classes.map((c) => (
                      <th key={c} className="p-2 text-center font-medium text-ink-faint">
                        pred {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.classes.map((actual) => (
                    <tr key={actual}>
                      <td className="p-2 font-medium text-ink-faint">actual {actual}</td>
                      {dataset.classes.map((pred) => {
                        const count = evaluation.confusion.get(actual)?.get(pred) ?? 0;
                        return (
                          <td
                            key={pred}
                            className={
                              "p-2 text-center tabular-nums " +
                              (actual === pred ? "rounded-md bg-brand-soft font-semibold text-brand-dark" : "text-ink-soft")
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
                        <td className="py-2 pr-2 capitalize text-ink-soft">{entry.datasetShape}</td>
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
