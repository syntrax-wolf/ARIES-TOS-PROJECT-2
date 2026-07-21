import { create } from "zustand";
import { generateDataset, type Dataset, type DatasetConfig } from "../lib/dataset";
import { trainDecisionTree, type TrainedTree } from "../lib/decisionTree";
import { trainRandomForest, type TrainedForest } from "../lib/randomForest";

export type Page = "select" | "hyperparams" | "animation" | "results";
export type AlgorithmId = "decision-tree" | "random-forest" | "logistic-regression" | "knn" | "svm" | "neural-net";

export interface Hyperparams {
  maxDepth: number;
  numTrees: number; // only used by random forest
}

export const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  seed: 42,
};

export const DEFAULT_HYPERPARAMS: Hyperparams = {
  maxDepth: 6,
  numTrees: 5,
};

interface SylvaState {
  page: Page;
  algorithm: AlgorithmId | null;
  datasetConfig: DatasetConfig;
  hyperparams: Hyperparams;
  dataset: Dataset | null;
  tree: TrainedTree | null;
  forest: TrainedForest | null;
  trainedAt: number | null;
  isTraining: boolean;

  setPage: (page: Page) => void;
  selectAlgorithm: (algorithm: AlgorithmId) => void;
  updateHyperparams: (partial: Partial<Hyperparams>) => void;
  regenerateSeed: () => void;
  runTraining: () => Promise<void>;
  restart: () => void;
}

export const useSylvaStore = create<SylvaState>((set, get) => ({
  page: "select",
  algorithm: null,
  datasetConfig: DEFAULT_DATASET_CONFIG,
  hyperparams: DEFAULT_HYPERPARAMS,
  dataset: null,
  tree: null,
  forest: null,
  trainedAt: null,
  isTraining: false,

  setPage: (page) => set({ page }),

  selectAlgorithm: (algorithm) => set({ algorithm, page: "hyperparams" }),

  updateHyperparams: (partial) => set({ hyperparams: { ...get().hyperparams, ...partial } }),

  regenerateSeed: () =>
    set({ datasetConfig: { ...get().datasetConfig, seed: Math.floor(get().datasetConfig.seed * 9301 + 49297) % 233280 } }),

  runTraining: async () => {
    set({ isTraining: true });
    const { datasetConfig, hyperparams, algorithm } = get();
    const dataset = await generateDataset(datasetConfig);

    if (algorithm === "random-forest") {
      const forest = trainRandomForest(dataset.train, hyperparams, datasetConfig.seed);
      set({ dataset, forest, tree: null, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    } else {
      const tree = trainDecisionTree(dataset.train, { maxDepth: hyperparams.maxDepth });
      set({ dataset, tree, forest: null, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    }
  },

  restart: () =>
    set({
      page: "select",
      algorithm: null,
      dataset: null,
      tree: null,
      forest: null,
      trainedAt: null,
    }),
}));
