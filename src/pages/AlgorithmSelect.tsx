import { motion } from "framer-motion";
import type { ReactElement, ReactNode } from "react";
import { StepHeader } from "../components/StepHeader";
import { useSylvaStore, type AlgorithmId } from "../state/store";

interface AlgorithmDef {
  id: AlgorithmId;
  name: string;
  description: string;
  icon: ReactElement;
  /** Shown as a small pill in place of the (removed) "coming soon" tag — currently only the unrated CNN demo uses it. */
  badge?: string;
}

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 40 40" className="h-9 w-9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  );
}

const ALGORITHMS: AlgorithmDef[] = [
  {
    id: "decision-tree",
    name: "Decision Tree",
    description: "Watch a tree recursively split pixels to recognize handwritten digits.",
    icon: (
      <Icon>
        <path d="M20 6v10M20 16l-8 8M20 16l8 8M12 24v6M28 24v6M12 24l-4 6M12 24l4 6M28 24l-4 6M28 24l4 6" />
        <circle cx="20" cy="6" r="2.4" fill="currentColor" stroke="none" />
      </Icon>
    ),
  },
  {
    id: "random-forest",
    name: "Random Forest",
    description: "Many trees, each grown on a random slice of the digits, voting together.",
    icon: (
      <Icon>
        <path d="M10 34V20M10 20l-4 6M10 20l4 6M20 34V16M20 16l-5 7M20 16l5 7M30 34V22M30 22l-4 6M30 22l4 6" />
      </Icon>
    ),
  },
  {
    id: "gradient-boosting",
    name: "Gradient Boosting",
    description: "A sequence of small, imperfect learners each correcting the last one's mistakes.",
    icon: (
      <Icon>
        <path d="M20 34V9" />
        <path d="M20 15c-4 0-6.5-2.6-6.5-6" />
        <path d="M20 22c4.5 0 7-2.8 7-6.5" />
        <path d="M20 29c-3.5 0-5.5-2.2-5.5-5" />
        <circle cx="20" cy="8" r="2.2" fill="currentColor" stroke="none" />
      </Icon>
    ),
  },
  {
    id: "knn",
    name: "k-Nearest Neighbors",
    description: "Classify a digit by polling its closest matches among the training images.",
    icon: (
      <Icon>
        <circle cx="20" cy="20" r="3" fill="currentColor" stroke="none" />
        <circle cx="12" cy="14" r="2" />
        <circle cx="28" cy="13" r="2" />
        <circle cx="10" cy="27" r="2" />
        <circle cx="30" cy="27" r="2" />
        <path d="M20 20L12 14M20 20L28 13M20 20L10 27M20 20L30 27" strokeDasharray="2 2" />
      </Icon>
    ),
  },
  {
    id: "neural-net",
    name: "Neural Network",
    description: "Watch backpropagation update its weights, beat by beat, into a trained network.",
    icon: (
      <Icon>
        <circle cx="8" cy="12" r="2" />
        <circle cx="8" cy="20" r="2" />
        <circle cx="8" cy="28" r="2" />
        <circle cx="20" cy="8" r="2" />
        <circle cx="20" cy="20" r="2" />
        <circle cx="20" cy="32" r="2" />
        <circle cx="32" cy="14" r="2" />
        <circle cx="32" cy="26" r="2" />
        <path d="M8 12l12-4M8 12l12 8M8 20l12-12M8 20l12 12M8 28l12-8M8 28l12 4M20 8l12 6M20 20l12-6M20 20l12 6M20 32l12-6" opacity="0.6" />
      </Icon>
    ),
  },
  {
    id: "cnn",
    name: "Convolutional Network",
    description: "A pre-trained model slides filters across the image, then makes its call.",
    badge: "For fun · unrated",
    icon: (
      <Icon>
        <path
          d="M8 8h24M8 16h24M8 24h24M8 32h24M8 8v24M16 8v24M24 8v24M32 8v24"
          opacity="0.5"
        />
        <rect x="8" y="8" width="16" height="16" rx="1.5" strokeWidth="2.2" />
      </Icon>
    ),
  },
];

export function AlgorithmSelect() {
  const selectAlgorithm = useSylvaStore((s) => s.selectAlgorithm);

  return (
    <div className="min-h-screen">
      <StepHeader step={1} />

      <main className="mx-auto max-w-5xl px-6 pb-24 pt-6 sm:px-8">
        <div className="mb-12 text-center">
          <h1 className="text-4xl text-ink sm:text-[42px]">Choose an algorithm to explore</h1>
          <p className="mx-auto mt-3 max-w-xl text-[17px] leading-relaxed text-ink-soft">
            Pick a model, tune how it learns, then watch it train step by step on handwritten digits.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {ALGORITHMS.map((algo, i) => (
            <motion.button
              key={algo.id}
              type="button"
              onClick={() => selectAlgorithm(algo.id)}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.35, ease: "easeOut" }}
              whileHover={{ y: -3 }}
              className="group relative flex cursor-pointer flex-col items-start gap-4 rounded-2xl border border-border bg-surface p-6 text-left shadow-card transition-all hover:border-brand/40 hover:shadow-soft"
            >
              {algo.badge && (
                <span className="absolute right-4 top-4 rounded-full bg-amber-soft px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-amber">
                  {algo.badge}
                </span>
              )}
              <span className="flex h-14 w-14 items-center justify-center rounded-xl bg-brand-soft text-brand-dark">
                {algo.icon}
              </span>
              <div>
                <h3 className="text-lg font-semibold text-ink">{algo.name}</h3>
                <p className="mt-1 text-sm leading-relaxed text-ink-soft">{algo.description}</p>
              </div>
            </motion.button>
          ))}
        </div>
      </main>
    </div>
  );
}
