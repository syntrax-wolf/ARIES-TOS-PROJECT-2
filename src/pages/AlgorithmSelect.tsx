import { motion } from "framer-motion";
import type { ReactElement, ReactNode } from "react";
import clsx from "clsx";
import { StepHeader } from "../components/StepHeader";
import { useSylvaStore, type AlgorithmId } from "../state/store";

interface AlgorithmDef {
  id: AlgorithmId;
  name: string;
  description: string;
  available: boolean;
  icon: ReactElement;
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
    available: true,
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
    description: "An ensemble of decision trees voting together.",
    available: false,
    icon: (
      <Icon>
        <path d="M10 34V20M10 20l-4 6M10 20l4 6M20 34V16M20 16l-5 7M20 16l5 7M30 34V22M30 22l-4 6M30 22l4 6" />
      </Icon>
    ),
  },
  {
    id: "logistic-regression",
    name: "Logistic Regression",
    description: "A smooth probability curve separating two classes.",
    available: false,
    icon: (
      <Icon>
        <path d="M6 30c6 0 8-20 14-20s8 20 14 20" />
      </Icon>
    ),
  },
  {
    id: "knn",
    name: "k-Nearest Neighbors",
    description: "Classify a point by polling its closest neighbors.",
    available: false,
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
    id: "svm",
    name: "Support Vector Machine",
    description: "Find the widest possible margin between classes.",
    available: false,
    icon: (
      <Icon>
        <path d="M6 12l28 16M6 20l28 16M6 4l28 16" strokeDasharray="0 0 100 0" />
        <path d="M4 16l32 0" transform="rotate(20 20 20)" />
      </Icon>
    ),
  },
  {
    id: "neural-net",
    name: "Neural Network",
    description: "Layers of connected units learning nonlinear patterns.",
    available: false,
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
              disabled={!algo.available}
              onClick={() => algo.available && selectAlgorithm(algo.id)}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.35, ease: "easeOut" }}
              whileHover={algo.available ? { y: -3 } : undefined}
              className={clsx(
                "group relative flex flex-col items-start gap-4 rounded-2xl border p-6 text-left transition-all",
                algo.available
                  ? "cursor-pointer border-border bg-surface shadow-card hover:border-brand/40 hover:shadow-soft"
                  : "cursor-not-allowed border-border-soft bg-surface-soft/60 opacity-70"
              )}
            >
              {!algo.available && (
                <span className="absolute right-4 top-4 rounded-full bg-surface-soft px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
                  Coming soon
                </span>
              )}
              <span
                className={clsx(
                  "flex h-14 w-14 items-center justify-center rounded-xl",
                  algo.available ? "bg-brand-soft text-brand-dark" : "bg-surface text-ink-faint"
                )}
              >
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
