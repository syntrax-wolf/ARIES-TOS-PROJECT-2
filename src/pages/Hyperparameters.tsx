import { useEffect, useMemo, useState } from "react";
import { StepHeader } from "../components/StepHeader";
import { Card } from "../components/Card";
import { Button } from "../components/Button";
import { Slider } from "../components/Slider";
import { SegmentedControl } from "../components/SegmentedControl";
import { useSylvaStore } from "../state/store";
import { generateDataset, type Dataset } from "../lib/dataset";
import { pixelsToDataUrl } from "../lib/mnist";

const PREVIEW_COUNT = 24;

function DatasetPreview() {
  const datasetConfig = useSylvaStore((s) => s.datasetConfig);
  const [dataset, setDataset] = useState<Dataset | null>(null);

  useEffect(() => {
    let cancelled = false;
    generateDataset(datasetConfig).then((ds) => {
      if (!cancelled) setDataset(ds);
    });
    return () => {
      cancelled = true;
    };
  }, [datasetConfig]);

  const thumbnails = useMemo(() => {
    if (!dataset) return [];
    const pool = [...dataset.train, ...dataset.test].slice(0, PREVIEW_COUNT);
    return pool.map((p) => pixelsToDataUrl(p.raw, 28, 2));
  }, [dataset]);

  if (!dataset) {
    return <div className="flex h-full items-center justify-center text-sm text-ink-faint">Loading digits…</div>;
  }

  return (
    <div className="grid grid-cols-6 gap-1.5">
      {thumbnails.map((src, i) => (
        <img key={i} src={src} alt="" className="aspect-square w-full rounded-md bg-surface-soft" />
      ))}
    </div>
  );
}

type Kind = "tree" | "forest" | "network" | "knn" | "boosting" | "cnn";

const TITLES: Record<Kind, string> = {
  tree: "decision tree",
  forest: "random forest",
  network: "neural network",
  knn: "k-NN classifier",
  boosting: "gradient boosting ensemble",
  cnn: "convolutional network",
};

const INTROS: Record<Kind, string> = {
  tree: "It will learn to recognize handwritten digits from the MNIST dataset.",
  forest: "A forest of trees will each learn from a random slice of the MNIST digits, then vote on the answer.",
  network: "A small neural network will learn the MNIST digits through backpropagation.",
  knn: "It will classify each digit by finding its closest matches among the training images — no training step required.",
  boosting: "A short sequence of small trees will train one after another, each one focused on fixing the last one's mistakes.",
  cnn: "This model arrives already trained. It will slide learned filters across each digit and make its own call — just for fun, it doesn't join the leaderboard.",
};

const INTRO_SUFFIXES: Record<Kind, string> = {
  tree: "Choose how it's allowed to grow.",
  forest: "Choose how it's allowed to grow.",
  network: "Choose its shape.",
  knn: "Choose how it measures closeness.",
  boosting: "Choose how many learners join the ensemble.",
  cnn: "There's nothing to tune — just pick which digits to show it.",
};

const TRAIN_LABELS: Record<Kind, string> = {
  tree: "Train & grow tree",
  forest: "Train & grow forest",
  network: "Train & grow network",
  knn: "Meet the neighbors",
  boosting: "Train & combine learners",
  cnn: "Watch it think",
};

export function Hyperparameters() {
  const { algorithm, hyperparams, updateHyperparams, regenerateSeed, runTraining, setPage, isTraining } = useSylvaStore();
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

  return (
    <div className="min-h-screen">
      <StepHeader step={2} />

      <main className="mx-auto max-w-4xl px-6 pb-28 pt-2 sm:px-8">
        <div className="mb-10 text-center">
          <h1 className="text-4xl text-ink sm:text-[38px]">Configure your {TITLES[kind]}</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            {INTROS[kind]} {INTRO_SUFFIXES[kind]}
          </p>
        </div>

        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[1fr_1fr]">
          <Card className="p-6 sm:p-7">
            <h2 className="mb-1 text-lg font-semibold text-ink">Dataset</h2>
            <p className="mb-5 text-sm text-ink-soft">A standard sample of real handwritten digits, 0 through 9.</p>
            <div className="mb-6">
              <DatasetPreview />
            </div>
            <Button variant="secondary" size="md" onClick={regenerateSeed} className="w-full">
              Shuffle new sample
            </Button>
          </Card>

          <Card className="p-6 sm:p-7">
            <h2 className="mb-1 text-lg font-semibold text-ink">{kind === "cnn" ? "About this model" : `Hyperparameter${kind === "tree" ? "" : "s"}`}</h2>
            <p className="mb-5 text-sm text-ink-soft">
              {kind === "tree"
                ? "The one dial that matters most for a decision tree."
                : kind === "forest"
                  ? "The two dials that matter most for a forest."
                  : kind === "network"
                    ? "The two dials that matter most for a neural network."
                    : kind === "knn"
                      ? "The two dials that matter most for k-nearest neighbors."
                      : kind === "boosting"
                        ? "The one dial that matters most for gradient boosting."
                        : "Its filters were fit ahead of time — nothing here to adjust."}
            </p>
            {kind === "cnn" && (
              <div className="rounded-xl border border-border-soft bg-surface-soft/60 p-4 text-sm leading-relaxed text-ink-soft">
                A convolutional network learns small filters that slide across the image looking for edges and strokes, pools
                the results down, and repeats — building up from raw pixels to a final digit guess. This one's weights were
                trained once, offline; running it here is pure inference. Because it isn't trained live like the others, it
                sits outside the leaderboard.
              </div>
            )}
            <div className="space-y-6">
              {kind === "boosting" && (
                <Slider
                  label="Number of learners"
                  value={hyperparams.numLearners}
                  min={3}
                  max={16}
                  onChange={(numLearners) => updateHyperparams({ numLearners })}
                  hint="Each learner is small and alone barely beats guessing — but stacking their corrections steadily sharpens the ensemble."
                />
              )}
              {kind === "forest" && (
                <Slider
                  label="Number of trees"
                  value={hyperparams.numTrees}
                  min={3}
                  max={9}
                  onChange={(numTrees) => updateHyperparams({ numTrees })}
                  hint="More trees vote together, which usually smooths out mistakes any single tree would make."
                />
              )}
              {kind === "network" && (
                <>
                  <Slider
                    label="Hidden layers"
                    value={hyperparams.hiddenLayers}
                    min={1}
                    max={3}
                    onChange={(hiddenLayers) => updateHyperparams({ hiddenLayers })}
                    hint="More layers let the network combine what earlier layers detected into more abstract ideas."
                  />
                  <Slider
                    label="Nodes per layer"
                    value={hyperparams.nodesPerLayer}
                    min={4}
                    max={16}
                    onChange={(nodesPerLayer) => updateHyperparams({ nodesPerLayer })}
                    hint="How many units each hidden layer has to work with."
                  />
                </>
              )}
              {(kind === "tree" || kind === "forest") && (
                <Slider
                  label="Max depth"
                  value={hyperparams.maxDepth}
                  min={1}
                  max={15}
                  onChange={(maxDepth) => updateHyperparams({ maxDepth })}
                  hint={
                    kind === "forest"
                      ? "How many questions deep each tree in the forest can go."
                      : "How many questions deep the tree can go. Shallow trees generalize better; deep trees can memorize the training digits too closely."
                  }
                />
              )}
              {kind === "knn" && (
                <>
                  <Slider
                    label="Neighbors (k)"
                    value={hyperparams.k}
                    min={1}
                    max={15}
                    onChange={(k) => updateHyperparams({ k })}
                    hint="How many closest training images get a vote. Small k follows local quirks; large k smooths toward the overall majority."
                  />
                  <div>
                    <SegmentedControl
                      label="Distance metric"
                      options={[
                        { value: "euclidean", label: "Euclidean" },
                        { value: "manhattan", label: "Manhattan" },
                      ]}
                      value={hyperparams.metric}
                      onChange={(metric) => updateHyperparams({ metric })}
                    />
                    <p className="mt-1.5 text-xs text-ink-faint">
                      How "closeness" between two images is measured — straight-line distance, or a city-block sum across pixels.
                    </p>
                  </div>
                </>
              )}
            </div>
          </Card>
        </div>

        <div className="mt-10 flex items-center justify-between">
          <Button variant="ghost" onClick={() => setPage("select")}>
            ← Back
          </Button>
          <Button variant="primary" size="lg" onClick={runTraining} disabled={isTraining}>
            {isTraining ? "Training…" : TRAIN_LABELS[kind]}
          </Button>
        </div>
      </main>
    </div>
  );
}
