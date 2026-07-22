import type { Point } from "./dataset";

export type DistanceMetric = "euclidean" | "manhattan";

export interface KnnHyperparams {
  k: number;
  metric: DistanceMetric;
}

export interface TrainedKnn {
  points: Point[];
  k: number;
  metric: DistanceMetric;
}

export interface Neighbor {
  point: Point;
  distance: number;
}

function distance(a: Float32Array, b: Float32Array, metric: DistanceMetric): number {
  if (metric === "manhattan") {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum;
  }
  let sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq);
}

/** KNN has no real training step — it just keeps the labeled examples around to compare against later. */
export function trainKnn(points: Point[], params: KnnHyperparams): TrainedKnn {
  return { points, k: params.k, metric: params.metric };
}

export function kNearest(model: TrainedKnn, features: Float32Array, k: number = model.k): Neighbor[] {
  const distances = model.points.map((point) => ({ point, distance: distance(features, point.features, model.metric) }));
  distances.sort((a, b) => a.distance - b.distance);
  return distances.slice(0, k);
}

/** Majority vote among the k nearest, ties broken in favor of whichever tied class has the closer representative. */
export function predict(model: TrainedKnn, point: Point): number {
  const neighbors = kNearest(model, point.features);
  const counts = new Map<number, number>();
  const firstIndex = new Map<number, number>();

  neighbors.forEach((n, idx) => {
    counts.set(n.point.label, (counts.get(n.point.label) ?? 0) + 1);
    if (!firstIndex.has(n.point.label)) firstIndex.set(n.point.label, idx);
  });

  let bestLabel = 0;
  let bestCount = -1;
  let bestFirst = Infinity;
  for (const [label, count] of counts) {
    const first = firstIndex.get(label)!;
    if (count > bestCount || (count === bestCount && first < bestFirst)) {
      bestLabel = label;
      bestCount = count;
      bestFirst = first;
    }
  }
  return bestLabel;
}
