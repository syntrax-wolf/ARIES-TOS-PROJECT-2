import { create } from "zustand";
import { generateDataset, type Dataset, type DatasetConfig } from "../lib/dataset";
import { trainDecisionTree, type TrainedTree } from "../lib/decisionTree";
import { trainRandomForest, type TrainedForest } from "../lib/randomForest";
import { trainNeuralNetwork, type TrainedNetwork } from "../lib/neuralNetwork";
import { trainKnn, type DistanceMetric, type TrainedKnn } from "../lib/knn";
import { trainGradientBoosting, type TrainedBoosting } from "../lib/gradientBoosting";
import { loadCnnWeights, type CnnWeights } from "../lib/cnn";

import type { Student } from "../lib/api";

export type Page = "register" | "select" | "hyperparams" | "animation" | "results";
export type AlgorithmId = "decision-tree" | "random-forest" | "gradient-boosting" | "knn" | "neural-net" | "cnn";

/** Algorithms whose accuracy isn't posted to the shared leaderboard. */
export const UNRATED_ALGORITHMS: ReadonlySet<AlgorithmId> = new Set(["cnn"]);

export interface Hyperparams {
  maxDepth: number;
  numTrees: number; // only used by random forest
  hiddenLayers: number; // only used by neural network
  nodesPerLayer: number; // only used by neural network
  k: number; // only used by knn
  metric: DistanceMetric; // only used by knn
  numLearners: number; // only used by gradient boosting
}

export const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  seed: 42,
};

export const DEFAULT_HYPERPARAMS: Hyperparams = {
  maxDepth: 6,
  numTrees: 5,
  hiddenLayers: 1,
  nodesPerLayer: 12,
  k: 5,
  metric: "euclidean",
  numLearners: 8,
};

interface SylvaState {
  page: Page;
  student: Student | null;
  algorithm: AlgorithmId | null;
  datasetConfig: DatasetConfig;
  hyperparams: Hyperparams;
  dataset: Dataset | null;
  tree: TrainedTree | null;
  forest: TrainedForest | null;
  network: TrainedNetwork | null;
  knn: TrainedKnn | null;
  boosting: TrainedBoosting | null;
  cnnWeights: CnnWeights | null;
  trainedAt: number | null;
  isTraining: boolean;

  setPage: (page: Page) => void;
  setStudent: (student: Student) => void;
  signOut: () => void;
  selectAlgorithm: (algorithm: AlgorithmId) => void;
  updateHyperparams: (partial: Partial<Hyperparams>) => void;
  regenerateSeed: () => void;
  runTraining: () => Promise<void>;
  restart: () => void;
}

const CLEARED_MODELS = { tree: null, forest: null, network: null, knn: null, boosting: null, cnnWeights: null };

export const useSylvaStore = create<SylvaState>((set, get) => ({
  page: "register",
  student: null,
  algorithm: null,
  datasetConfig: DEFAULT_DATASET_CONFIG,
  hyperparams: DEFAULT_HYPERPARAMS,
  dataset: null,
  tree: null,
  forest: null,
  network: null,
  knn: null,
  boosting: null,
  cnnWeights: null,
  trainedAt: null,
  isTraining: false,

  setPage: (page) => set({ page }),

  setStudent: (student) => set({ student }),

  signOut: () =>
    set({ student: null, page: "register", algorithm: null, dataset: null, ...CLEARED_MODELS, trainedAt: null }),

  selectAlgorithm: (algorithm) => set({ algorithm, page: "hyperparams" }),

  updateHyperparams: (partial) => set({ hyperparams: { ...get().hyperparams, ...partial } }),

  regenerateSeed: () =>
    set({ datasetConfig: { ...get().datasetConfig, seed: Math.floor(get().datasetConfig.seed * 9301 + 49297) % 233280 } }),

  runTraining: async () => {
    set({ isTraining: true });
    const { datasetConfig, hyperparams, algorithm } = get();

    if (algorithm === "cnn") {
      const [dataset, cnnWeights] = await Promise.all([generateDataset(datasetConfig), loadCnnWeights()]);
      set({ dataset, ...CLEARED_MODELS, cnnWeights, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
      return;
    }

    const dataset = await generateDataset(datasetConfig);

    if (algorithm === "random-forest") {
      const forest = trainRandomForest(dataset.train, hyperparams, datasetConfig.seed);
      set({ dataset, ...CLEARED_MODELS, forest, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    } else if (algorithm === "neural-net") {
      const network = trainNeuralNetwork(dataset.train, hyperparams, datasetConfig.seed);
      set({ dataset, ...CLEARED_MODELS, network, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    } else if (algorithm === "knn") {
      const knn = trainKnn(dataset.train, hyperparams);
      set({ dataset, ...CLEARED_MODELS, knn, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    } else if (algorithm === "gradient-boosting") {
      const boosting = trainGradientBoosting(dataset.train, hyperparams);
      set({ dataset, ...CLEARED_MODELS, boosting, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    } else {
      const tree = trainDecisionTree(dataset.train, { maxDepth: hyperparams.maxDepth });
      set({ dataset, ...CLEARED_MODELS, tree, trainedAt: datasetConfig.seed, page: "animation", isTraining: false });
    }
  },

  restart: () =>
    set({
      page: "select",
      algorithm: null,
      dataset: null,
      ...CLEARED_MODELS,
      trainedAt: null,
    }),
}));
