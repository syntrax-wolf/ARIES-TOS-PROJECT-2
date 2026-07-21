import { create } from "zustand";
import { generateDataset, type Dataset, type DatasetConfig } from "../lib/dataset";
import { trainDecisionTree, type TrainedTree, type TreeHyperparams } from "../lib/decisionTree";

export type Page = "select" | "hyperparams" | "animation" | "results";
export type AlgorithmId = "decision-tree" | "random-forest" | "logistic-regression" | "knn" | "svm" | "neural-net";

export const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  shape: "moons",
  noise: 0.25,
  samples: 320,
  testSize: 0.25,
  seed: 42,
};

export const DEFAULT_HYPERPARAMS: TreeHyperparams = {
  criterion: "gini",
  splitter: "best",
  maxDepth: 5,
  minSamplesSplit: 2,
  minSamplesLeaf: 3,
  minImpurityDecrease: 0,
  maxLeafNodes: null,
};

interface SylvaState {
  page: Page;
  algorithm: AlgorithmId | null;
  datasetConfig: DatasetConfig;
  hyperparams: TreeHyperparams;
  dataset: Dataset | null;
  tree: TrainedTree | null;
  trainedAt: number | null;

  setPage: (page: Page) => void;
  selectAlgorithm: (algorithm: AlgorithmId) => void;
  updateDatasetConfig: (partial: Partial<DatasetConfig>) => void;
  updateHyperparams: (partial: Partial<TreeHyperparams>) => void;
  regenerateSeed: () => void;
  runTraining: () => void;
  restart: () => void;
}

export const useSylvaStore = create<SylvaState>((set, get) => ({
  page: "select",
  algorithm: null,
  datasetConfig: DEFAULT_DATASET_CONFIG,
  hyperparams: DEFAULT_HYPERPARAMS,
  dataset: null,
  tree: null,
  trainedAt: null,

  setPage: (page) => set({ page }),

  selectAlgorithm: (algorithm) => set({ algorithm, page: "hyperparams" }),

  updateDatasetConfig: (partial) => set({ datasetConfig: { ...get().datasetConfig, ...partial } }),

  updateHyperparams: (partial) => set({ hyperparams: { ...get().hyperparams, ...partial } }),

  regenerateSeed: () =>
    set({ datasetConfig: { ...get().datasetConfig, seed: Math.floor(get().datasetConfig.seed * 9301 + 49297) % 233280 } }),

  runTraining: () => {
    const { datasetConfig, hyperparams } = get();
    const dataset = generateDataset(datasetConfig);
    const tree = trainDecisionTree(dataset.train, hyperparams, datasetConfig.seed);
    set({ dataset, tree, trainedAt: datasetConfig.seed, page: "animation" });
  },

  restart: () =>
    set({
      page: "select",
      algorithm: null,
      dataset: null,
      tree: null,
      trainedAt: null,
    }),
}));
