import { useMemo } from "react";
import { StepHeader } from "../components/StepHeader";
import { Card } from "../components/Card";
import { Button } from "../components/Button";
import { Slider } from "../components/Slider";
import { SegmentedControl } from "../components/SegmentedControl";
import { useSylvaStore } from "../state/store";
import { generateDataset, type DatasetShape } from "../lib/dataset";
import { classColor } from "../lib/palette";

const SHAPE_OPTIONS: { value: DatasetShape; label: string }[] = [
  { value: "moons", label: "Moons" },
  { value: "circles", label: "Circles" },
  { value: "blobs", label: "Blobs" },
  { value: "xor", label: "XOR" },
  { value: "linear", label: "Linear" },
];

function DatasetPreview() {
  const datasetConfig = useSylvaStore((s) => s.datasetConfig);
  const dataset = useMemo(() => generateDataset(datasetConfig), [datasetConfig]);
  const { bounds } = dataset;
  const size = 240;

  const project = (x: number, y: number) => {
    const px = ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * size;
    const py = size - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * size;
    return [px, py];
  };

  const all = [...dataset.train, ...dataset.test];

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-full w-full rounded-xl bg-surface-soft">
      {all.map((p, i) => {
        const [px, py] = project(p.x, p.y);
        return <circle key={i} cx={px} cy={py} r={2.1} fill={classColor(p.label)} opacity={0.75} />;
      })}
    </svg>
  );
}

export function Hyperparameters() {
  const { datasetConfig, hyperparams, updateDatasetConfig, updateHyperparams, regenerateSeed, runTraining, setPage } =
    useSylvaStore();

  return (
    <div className="min-h-screen">
      <StepHeader step={2} />

      <main className="mx-auto max-w-5xl px-6 pb-28 pt-2 sm:px-8">
        <div className="mb-10 text-center">
          <h1 className="text-4xl text-ink sm:text-[38px]">Configure your decision tree</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            Shape the data it will learn from, then tune the hyperparameters that control how it grows.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.1fr_1fr]">
          <Card className="p-6 sm:p-7">
            <h2 className="mb-5 text-lg font-semibold text-ink">Dataset</h2>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-[1fr_auto] sm:gap-8">
              <div className="space-y-6">
                <SegmentedControl
                  label="Shape"
                  options={SHAPE_OPTIONS}
                  value={datasetConfig.shape}
                  onChange={(shape) => updateDatasetConfig({ shape })}
                />
                <Slider
                  label="Noise"
                  value={datasetConfig.noise}
                  min={0}
                  max={0.5}
                  step={0.01}
                  formatValue={(v) => v.toFixed(2)}
                  onChange={(noise) => updateDatasetConfig({ noise })}
                />
                <Slider
                  label="Samples"
                  value={datasetConfig.samples}
                  min={100}
                  max={2000}
                  step={20}
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
                />
                <Button variant="secondary" size="md" onClick={regenerateSeed} className="w-full">
                  Shuffle new sample
                </Button>
              </div>
              <div className="flex flex-col items-center gap-2">
                <div className="h-[240px] w-[240px]">
                  <DatasetPreview />
                </div>
                <span className="text-xs text-ink-faint">live preview</span>
              </div>
            </div>
          </Card>

          <Card className="p-6 sm:p-7">
            <h2 className="mb-5 text-lg font-semibold text-ink">Hyperparameters</h2>
            <div className="space-y-6">
              <SegmentedControl
                label="Split quality criterion"
                options={[
                  { value: "gini", label: "Gini" },
                  { value: "entropy", label: "Entropy" },
                  { value: "log_loss", label: "Log loss" },
                ]}
                value={hyperparams.criterion}
                onChange={(criterion) => updateHyperparams({ criterion })}
              />

              <SegmentedControl
                label="Splitter"
                options={[
                  { value: "best", label: "Best split" },
                  { value: "random", label: "Random split" },
                ]}
                value={hyperparams.splitter}
                onChange={(splitter) => updateHyperparams({ splitter })}
              />

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-medium text-ink">Max depth</span>
                  <button
                    type="button"
                    onClick={() => updateHyperparams({ maxDepth: hyperparams.maxDepth === null ? 5 : null })}
                    className="text-xs font-medium text-brand-dark underline-offset-2 hover:underline"
                  >
                    {hyperparams.maxDepth === null ? "set a limit" : "make unlimited"}
                  </button>
                </div>
                <Slider
                  label=""
                  value={hyperparams.maxDepth ?? 20}
                  min={1}
                  max={20}
                  onChange={(maxDepth) => updateHyperparams({ maxDepth })}
                  formatValue={() => (hyperparams.maxDepth === null ? "Unlimited" : String(hyperparams.maxDepth))}
                />
              </div>

              <Slider
                label="Min samples to split a node"
                value={hyperparams.minSamplesSplit}
                min={2}
                max={50}
                onChange={(minSamplesSplit) => updateHyperparams({ minSamplesSplit })}
              />

              <Slider
                label="Min samples per leaf"
                value={hyperparams.minSamplesLeaf}
                min={1}
                max={50}
                onChange={(minSamplesLeaf) => updateHyperparams({ minSamplesLeaf })}
              />

              <Slider
                label="Min impurity decrease"
                value={hyperparams.minImpurityDecrease}
                min={0}
                max={0.3}
                step={0.005}
                formatValue={(v) => v.toFixed(3)}
                onChange={(minImpurityDecrease) => updateHyperparams({ minImpurityDecrease })}
              />

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-medium text-ink">Max leaf nodes</span>
                  <button
                    type="button"
                    onClick={() =>
                      updateHyperparams({ maxLeafNodes: hyperparams.maxLeafNodes === null ? 20 : null })
                    }
                    className="text-xs font-medium text-brand-dark underline-offset-2 hover:underline"
                  >
                    {hyperparams.maxLeafNodes === null ? "set a limit" : "make unlimited"}
                  </button>
                </div>
                <Slider
                  label=""
                  value={hyperparams.maxLeafNodes ?? 100}
                  min={2}
                  max={100}
                  onChange={(maxLeafNodes) => updateHyperparams({ maxLeafNodes })}
                  formatValue={() => (hyperparams.maxLeafNodes === null ? "Unlimited" : String(hyperparams.maxLeafNodes))}
                />
              </div>
            </div>
          </Card>
        </div>

        <div className="mt-10 flex items-center justify-between">
          <Button variant="ghost" onClick={() => setPage("select")}>
            ← Back
          </Button>
          <Button variant="primary" size="lg" onClick={runTraining}>
            Train &amp; grow tree
          </Button>
        </div>
      </main>
    </div>
  );
}
