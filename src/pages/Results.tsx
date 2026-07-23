import { useEffect, useMemo, useRef, useState } from "react";
import { StepHeader } from "../components/StepHeader";
import { Card } from "../components/Card";
import { Button } from "../components/Button";
import { useSylvaStore, UNRATED_ALGORITHMS } from "../state/store";
import { featureImportance, predict } from "../lib/decisionTree";
import type { Point } from "../lib/dataset";
import { forestFeatureImportance, predictForest } from "../lib/randomForest";
import { predict as predictNetwork, type TrainedNetwork } from "../lib/neuralNetwork";
import { kNearest, predict as predictKnn, type TrainedKnn } from "../lib/knn";
import { boostingFeatureImportance, predictBoosting, LEARNER_MAX_DEPTH } from "../lib/gradientBoosting";
import { predictCnn, type CnnWeights } from "../lib/cnn";
import { evaluate } from "../lib/metrics";
import { classColor, heatColor, weightColor } from "../lib/palette";
import { pixelsToDataUrl, FEATURE_PIXELS, FEATURE_SIZE, RAW_SIZE } from "../lib/mnist";
import { ApiError, fetchLeaderboard, submitRun, type BoardRow, type SubmitResult } from "../lib/api";

const HEAT_CELL = 20;
const GALLERY_COUNT = 12;
const FIELD_CELL = 4;

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

/** Each hidden neuron's incoming weight vector, reshaped back into a 14x14 grid — a little "what pattern fires this unit" filter. */
function ReceptiveFields({ network }: { network: TrainedNetwork }) {
  const layer0 = network.snapshots[network.snapshots.length - 1].layers[0];
  const inputSize = network.layerSizes[0];
  const hiddenCount = network.layerSizes[1];
  const size = FEATURE_SIZE * FIELD_CELL;

  const maxAbs = useMemo(() => {
    let m = 1e-9;
    for (let i = 0; i < inputSize; i++) for (let j = 0; j < hiddenCount; j++) m = Math.max(m, Math.abs(layer0.W[i][j]));
    return m;
  }, [layer0, inputSize, hiddenCount]);

  return (
    <div className="grid grid-cols-4 gap-2">
      {Array.from({ length: hiddenCount }).map((_, j) => (
        <svg key={j} viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="rounded-md border border-border-soft">
          {Array.from({ length: FEATURE_SIZE }).map((_, row) =>
            Array.from({ length: FEATURE_SIZE }).map((_, col) => {
              const i = row * FEATURE_SIZE + col;
              const w = layer0.W[i][j] / maxAbs;
              return <rect key={`${row}-${col}`} x={col * FIELD_CELL} y={row * FIELD_CELL} width={FIELD_CELL} height={FIELD_CELL} fill={weightColor(w)} />;
            })
          )}
        </svg>
      ))}
    </div>
  );
}

/** One example digit alongside the k training images it matched most closely — KNN's "reasoning" made visible. */
function NearestNeighborsPanel({ knn, example }: { knn: TrainedKnn; example: Point }) {
  const neighbors = useMemo(() => kNearest(knn, example.features, knn.k), [knn, example]);
  const queryThumb = useMemo(() => pixelsToDataUrl(example.raw, RAW_SIZE, 2), [example]);

  return (
    <div className="flex flex-wrap items-center gap-5">
      <div className="flex flex-col items-center gap-1.5">
        <img src={queryThumb} alt="" className="h-16 w-16 rounded-full" style={{ boxShadow: "0 0 0 3px #2f6b4f" }} />
        <span className="text-[11px] font-medium text-ink-faint">query</span>
      </div>
      <div className="grid grid-cols-5 gap-2.5">
        {neighbors.map((n, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <img
              src={pixelsToDataUrl(n.point.raw, RAW_SIZE, 2)}
              alt=""
              className="h-12 w-12 rounded-full"
              style={{ boxShadow: `0 0 0 2px ${classColor(n.point.label)}` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

const FILTER_CELL = 13;

/** The CNN's own first-layer filters — literally the weights it slides across every digit, not a stand-in visualization. */
function CnnFilters({ weights }: { weights: CnnWeights }) {
  const filters = weights.conv1.filters; // [numFilters][3][3][1]
  const maxAbs = useMemo(() => {
    let m = 1e-9;
    for (const f of filters) for (const row of f) for (const cell of row) m = Math.max(m, Math.abs(cell[0]));
    return m;
  }, [filters]);
  const size = 3 * FILTER_CELL;

  return (
    <div className="flex flex-wrap gap-3">
      {filters.map((f, fi) => (
        <svg key={fi} viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="rounded-md border border-border-soft">
          {f.map((row, r) =>
            row.map((cell, c) => (
              <rect key={`${r}-${c}`} x={c * FILTER_CELL} y={r * FILTER_CELL} width={FILTER_CELL} height={FILTER_CELL} fill={weightColor(cell[0] / maxAbs)} />
            ))
          )}
        </svg>
      ))}
    </div>
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

type Kind = "tree" | "forest" | "network" | "knn" | "boosting" | "cnn";

export function Results() {
  const algorithm = useSylvaStore((s) => s.algorithm);
  const dataset = useSylvaStore((s) => s.dataset);
  const tree = useSylvaStore((s) => s.tree);
  const forest = useSylvaStore((s) => s.forest);
  const network = useSylvaStore((s) => s.network);
  const knn = useSylvaStore((s) => s.knn);
  const boosting = useSylvaStore((s) => s.boosting);
  const cnnWeights = useSylvaStore((s) => s.cnnWeights);
  const hyperparams = useSylvaStore((s) => s.hyperparams);
  const student = useSylvaStore((s) => s.student);
  const trainedAt = useSylvaStore((s) => s.trainedAt);
  const setPage = useSylvaStore((s) => s.setPage);
  const restart = useSylvaStore((s) => s.restart);

  const kind: Kind =
    algorithm === "random-forest"
      ? "forest"
      : algorithm === "neural-net"
        ? "network"
        : algorithm === "knn"
          ? "knn"
          : algorithm === "gradient-boosting"
            ? "boosting"
            : algorithm === "cnn"
              ? "cnn"
              : "tree";
  const model =
    kind === "forest" ? forest : kind === "network" ? network : kind === "knn" ? knn : kind === "boosting" ? boosting : kind === "cnn" ? cnnWeights : tree;
  const unrated = algorithm !== null && UNRATED_ALGORITHMS.has(algorithm);

  const [board, setBoard] = useState<BoardRow[]>([]);
  const [submission, setSubmission] = useState<SubmitResult | null>(null);
  const [boardError, setBoardError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const submittedRef = useRef(false);

  const predictOne = useMemo(() => {
    if (kind === "forest" && forest) return (p: Point) => predictForest(forest, p);
    if (kind === "network" && network) {
      const finalSnapshot = network.snapshots[network.snapshots.length - 1];
      return (p: Point) => predictNetwork(finalSnapshot, p);
    }
    if (kind === "knn" && knn) return (p: Point) => predictKnn(knn, p);
    if (kind === "boosting" && boosting) return (p: Point) => predictBoosting(boosting, p);
    if (kind === "cnn" && cnnWeights) return (p: Point) => predictCnn(cnnWeights, p);
    if (kind === "tree" && tree) return (p: Point) => predict(tree.root, p);
    return null;
  }, [kind, forest, network, knn, boosting, cnnWeights, tree]);

  const importance = useMemo(() => {
    if (kind === "forest" && forest) return forestFeatureImportance(forest, FEATURE_PIXELS);
    if (kind === "boosting" && boosting) return boostingFeatureImportance(boosting, FEATURE_PIXELS);
    if (kind === "tree" && tree) return featureImportance(tree, FEATURE_PIXELS);
    return null;
  }, [kind, forest, boosting, tree]);

  const evaluation = useMemo(() => {
    if (!dataset || !predictOne) return null;
    const predictions = dataset.test.map(predictOne);
    return evaluate(dataset.test, predictions, dataset.classes);
  }, [dataset, predictOne]);

  // The browser's model drove the animation; it does not decide the score.
  // Submit what was configured and let the server retrain and rank it.
  //
  // submittedRef alone guards this — deliberately no "cancelled" flag. React's
  // StrictMode mounts, cleans up, then remounts in dev; a cleanup flag would
  // discard the response of the one request we fired and strand the UI on
  // "Scoring your run…". Setting state after unmount is a harmless no-op.
  useEffect(() => {
    if (submittedRef.current || !model || !evaluation || !student || trainedAt === null) return;
    submittedRef.current = true;

    async function run() {
      if (unrated) {
        try {
          setBoard(await fetchLeaderboard());
        } catch (err) {
          setBoardError(err instanceof ApiError ? err.message : "Could not load the leaderboard.");
        }
        return;
      }

      setSubmitting(true);
      try {
        const result = await submitRun({
          studentId: student!.student_id,
          algorithm: algorithm!,
          seed: trainedAt!,
          hyperparams: { ...hyperparams },
          clientAccuracy: evaluation!.accuracy,
          clientMacroF1: evaluation!.macroF1,
        });
        setSubmission(result);
        setBoard(await fetchLeaderboard());
      } catch (err) {
        setBoardError(err instanceof ApiError ? err.message : "Could not submit your run.");
      } finally {
        setSubmitting(false);
      }
    }

    void run();
  }, [model, evaluation, student, trainedAt, algorithm, hyperparams, unrated]);

  if (!dataset || !model || !evaluation || !predictOne) {
    return (
      <div className="min-h-screen">
        <StepHeader step={4} />
        <div className="px-8 py-20 text-center text-ink-soft">No trained model yet — go back and train one first.</div>
      </div>
    );
  }

  const rank = submission?.position ?? null;
  const currentRunId = submission?.run_id ?? null;
  const subject =
    kind === "forest"
      ? "forest"
      : kind === "network"
        ? "network"
        : kind === "knn"
          ? "k-NN classifier"
          : kind === "boosting"
            ? "boosting ensemble"
            : kind === "cnn"
              ? "convolutional network"
              : "tree";

  const stats =
    kind === "forest" && forest
      ? [
          { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
          { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
          { label: "Trees", value: String(forest.trees.length) },
          { label: "Avg tree depth", value: (forest.trees.reduce((s, t) => s + t.depth, 0) / forest.trees.length).toFixed(1) },
        ]
      : kind === "network" && network
        ? [
            { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
            { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
            { label: "Hidden layers", value: String(network.layerSizes.length - 2) },
            { label: "Nodes per layer", value: String(network.layerSizes[1]) },
          ]
        : kind === "knn"
          ? [
              { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
              { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
              { label: "Neighbors (k)", value: String(hyperparams.k) },
              { label: "Distance metric", value: hyperparams.metric === "euclidean" ? "Euclidean" : "Manhattan" },
            ]
          : kind === "boosting" && boosting
            ? [
                { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
                { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
                { label: "Learners", value: String(boosting.rounds.length) },
                { label: "Learner depth", value: String(LEARNER_MAX_DEPTH) },
              ]
            : kind === "cnn" && cnnWeights
              ? [
                  { label: "Test accuracy", value: `${(evaluation.accuracy * 100).toFixed(1)}%` },
                  { label: "Macro F1", value: evaluation.macroF1.toFixed(3) },
                  { label: "Conv layers", value: "2" },
                  { label: "Learned filters", value: String(cnnWeights.conv1.filters.length + cnnWeights.conv2.filters.length) },
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
          <h1 className="text-4xl text-ink sm:text-[38px]">Your {subject}, evaluated</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            Performance on handwritten digits the {subject} never saw during training.
          </p>
          {unrated && (
            <span className="mt-3 inline-block rounded-full bg-amber-soft px-3 py-1 text-[12px] font-medium uppercase tracking-wide text-amber">
              Unrated · just for fun
            </span>
          )}
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
              <h2 className="mb-1 text-lg font-semibold text-ink">
                {kind === "network"
                  ? "What each neuron looks for"
                  : kind === "knn"
                    ? "Its nearest neighbors"
                    : kind === "cnn"
                      ? "What its filters look for"
                      : "Pixels that matter"}
              </h2>
              <p className="mb-4 text-sm text-ink-soft">
                {kind === "network"
                  ? "Each tile is one hidden neuron's learned pattern — green where it looks for ink, amber where it looks for blank paper."
                  : kind === "knn"
                    ? "For one example digit, these are the training images it matched most closely."
                    : kind === "cnn"
                      ? "Each tile is one of its learned 3×3 filters, slid across every position of the image — green where it looks for more ink, amber for less."
                      : `Where on the digit the ${subject} chose to look.`}
              </p>
              {kind === "network" && network ? (
                <ReceptiveFields network={network} />
              ) : kind === "knn" && knn ? (
                <NearestNeighborsPanel knn={knn} example={dataset.test[0]} />
              ) : kind === "cnn" && cnnWeights ? (
                <CnnFilters weights={cnnWeights} />
              ) : (
                importance && <PixelHeatmap importance={importance} />
              )}
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

            {submission && (
              <Card className="p-6">
                <div className="flex items-baseline justify-between">
                  <h2 className="text-lg font-semibold text-ink">Verified score</h2>
                  <span className="text-[13px] text-ink-faint">seed {submission.seed}</span>
                </div>
                <p className="mt-1 text-sm text-ink-soft">
                  The server retrained this configuration on the same {submission.official.n_train}/
                  {submission.official.n_test} split and scored it independently. This is the number that
                  ranks — not the one your browser computed.
                </p>
                <div className="mt-4 flex flex-wrap items-end gap-x-8 gap-y-3">
                  <div>
                    <div className="text-3xl font-semibold text-brand-dark tabular-nums">
                      {submission.official.macro_f1.toFixed(3)}
                    </div>
                    <div className="mt-0.5 text-sm text-ink-soft">Macro F1 · official</div>
                  </div>
                  <div>
                    <div className="text-3xl font-semibold text-ink tabular-nums">
                      {(submission.official.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="mt-0.5 text-sm text-ink-soft">Accuracy · official</div>
                  </div>
                  <div>
                    <div className="text-lg font-medium text-ink-faint tabular-nums">
                      {evaluation.macroF1.toFixed(3)} / {(evaluation.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="mt-0.5 text-sm text-ink-faint">Your browser's model</div>
                  </div>
                </div>
              </Card>
            )}

            <Card className="flex-1 p-6">
              <div className="mb-4 flex items-baseline justify-between">
                <h2 className="text-lg font-semibold text-ink">Leaderboard</h2>
                {unrated ? (
                  <span className="text-sm font-medium text-ink-faint">Unrated runs don't appear here</span>
                ) : submitting ? (
                  <span className="text-sm font-medium text-ink-faint">Scoring your run…</span>
                ) : rank ? (
                  <span className="text-sm font-medium text-brand-dark">
                    You placed #{rank} of {submission?.total ?? board.length}
                  </span>
                ) : submission ? (
                  <span className="text-sm font-medium text-ink-faint">
                    Not your best run — your earlier score still stands
                  </span>
                ) : null}
              </div>

              {boardError && (
                <p className="mb-3 rounded-xl bg-amber-soft px-3.5 py-2.5 text-sm text-danger">{boardError}</p>
              )}

              <div className="max-h-64 overflow-y-auto">
                <table className="w-full border-collapse text-sm">
                  <tbody>
                    {board.map((row, i) => (
                      <tr
                        key={row.run_id}
                        className={
                          row.run_id === currentRunId
                            ? "bg-brand-soft/70"
                            : i % 2 === 1
                              ? "bg-surface-soft/60"
                              : ""
                        }
                      >
                        <td className="rounded-l-md py-2 pl-2 pr-1 font-medium text-ink-faint">#{row.position}</td>
                        <td className="py-2 pr-2 font-medium text-ink">{row.name}</td>
                        <td className="py-2 pr-2 font-mono text-[12px] text-ink-faint">{row.entry_number}</td>
                        <td className="py-2 pr-2 font-semibold tabular-nums text-ink">{row.macro_f1.toFixed(3)}</td>
                        <td className="rounded-r-md py-2 pr-2 text-ink-faint">{row.summary}</td>
                      </tr>
                    ))}
                    {board.length === 0 && !boardError && (
                      <tr>
                        <td className="py-3 text-sm text-ink-faint">No runs on the board yet.</td>
                      </tr>
                    )}
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
