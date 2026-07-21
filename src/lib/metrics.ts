import type { Point } from "./dataset";

export interface ClassMetric {
  label: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Evaluation {
  accuracy: number;
  macroF1: number;
  confusion: Map<number, Map<number, number>>; // confusion.get(actual).get(predicted)
  perClass: ClassMetric[];
  correct: boolean[]; // aligned to input points, for plotting
}

export function evaluate(points: Point[], predictions: number[], classes: number[]): Evaluation {
  const confusion = new Map<number, Map<number, number>>();
  for (const a of classes) {
    confusion.set(a, new Map(classes.map((p) => [p, 0])));
  }

  let correctCount = 0;
  const correct: boolean[] = [];
  points.forEach((p, i) => {
    const pred = predictions[i];
    confusion.get(p.label)!.set(pred, (confusion.get(p.label)!.get(pred) ?? 0) + 1);
    const isCorrect = pred === p.label;
    correct.push(isCorrect);
    if (isCorrect) correctCount += 1;
  });

  const perClass: ClassMetric[] = classes.map((c) => {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (const a of classes) {
      const row = confusion.get(a)!;
      for (const p of classes) {
        const count = row.get(p) ?? 0;
        if (a === c) support += count;
        if (a === c && p === c) tp += count;
        if (a !== c && p === c) fp += count;
        if (a === c && p !== c) fn += count;
      }
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { label: c, precision, recall, f1, support };
  });

  const macroF1 = perClass.reduce((s, c) => s + c.f1, 0) / (perClass.length || 1);

  return {
    accuracy: points.length === 0 ? 0 : correctCount / points.length,
    macroF1,
    confusion,
    perClass,
    correct,
  };
}
