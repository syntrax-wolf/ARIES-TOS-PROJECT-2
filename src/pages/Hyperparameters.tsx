import { useEffect, useMemo, useState } from "react";
import { StepHeader } from "../components/StepHeader";
import { Card } from "../components/Card";
import { Button } from "../components/Button";
import { Slider } from "../components/Slider";
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

export function Hyperparameters() {
  const { datasetConfig, hyperparams, updateDatasetConfig, updateHyperparams, regenerateSeed, runTraining, setPage, isTraining } =
    useSylvaStore();

  return (
    <div className="min-h-screen">
      <StepHeader step={2} />

      <main className="mx-auto max-w-4xl px-6 pb-28 pt-2 sm:px-8">
        <div className="mb-10 text-center">
          <h1 className="text-4xl text-ink sm:text-[38px]">Configure your decision tree</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            It will learn to recognize handwritten digits from the MNIST dataset. Choose how much data to show it,
            and how deep it's allowed to grow.
          </p>
        </div>

        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[1fr_1fr]">
          <Card className="p-6 sm:p-7">
            <h2 className="mb-1 text-lg font-semibold text-ink">Dataset</h2>
            <p className="mb-5 text-sm text-ink-soft">A sample of real handwritten digits, 0 through 9.</p>
            <div className="mb-6">
              <DatasetPreview />
            </div>
            <div className="space-y-6">
              <Slider
                label="Samples"
                value={datasetConfig.samples}
                min={200}
                max={4000}
                step={100}
                onChange={(samples) => updateDatasetConfig({ samples })}
              />
              <Slider
                label="Test split"
                value={datasetConfig.testSize}
                min={0.1}
                max={0.5}
                step={0.05}
                formatValue={(v) => `${Math.round(v * 100)}%`}
                onChange={(testSize) => updateDatasetConfig({ testSize })}
                hint="Held out to score the tree after training."
              />
              <Button variant="secondary" size="md" onClick={regenerateSeed} className="w-full">
                Shuffle new sample
              </Button>
            </div>
          </Card>

          <Card className="p-6 sm:p-7">
            <h2 className="mb-1 text-lg font-semibold text-ink">Hyperparameter</h2>
            <p className="mb-5 text-sm text-ink-soft">The one dial that matters most for a decision tree.</p>
            <Slider
              label="Max depth"
              value={hyperparams.maxDepth}
              min={1}
              max={15}
              onChange={(maxDepth) => updateHyperparams({ maxDepth })}
              hint="How many questions deep the tree can go. Shallow trees generalize better; deep trees can memorize the training digits too closely."
            />
          </Card>
        </div>

        <div className="mt-10 flex items-center justify-between">
          <Button variant="ghost" onClick={() => setPage("select")}>
            ← Back
          </Button>
          <Button variant="primary" size="lg" onClick={runTraining} disabled={isTraining}>
            {isTraining ? "Training…" : "Train & grow tree"}
          </Button>
        </div>
      </main>
    </div>
  );
}
