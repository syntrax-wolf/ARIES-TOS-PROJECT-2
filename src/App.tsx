import { lazy, Suspense } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "./state/store";
import { AlgorithmSelect } from "./pages/AlgorithmSelect";
import { Hyperparameters } from "./pages/Hyperparameters";
import { TreeAnimation } from "./pages/TreeAnimation";
import { ForestAnimation } from "./pages/ForestAnimation";
import { NeuralNetworkAnimation } from "./pages/NeuralNetworkAnimation";
import { KnnAnimation } from "./pages/KnnAnimation";
import { GradientBoostingAnimation } from "./pages/GradientBoostingAnimation";
import { Results } from "./pages/Results";

// The CNN page pulls in three.js + @react-three/fiber for its 3D forward-pass
// visualization — lazy-load it so everyone choosing a non-CNN algorithm never
// downloads that weight.
const CnnAnimation = lazy(() => import("./pages/CnnAnimation").then((m) => ({ default: m.CnnAnimation })));

function App() {
  const page = useSylvaStore((s) => s.page);
  const algorithm = useSylvaStore((s) => s.algorithm);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={page}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
      >
        {page === "select" && <AlgorithmSelect />}
        {page === "hyperparams" && <Hyperparameters />}
        {page === "animation" &&
          (algorithm === "random-forest" ? (
            <ForestAnimation />
          ) : algorithm === "neural-net" ? (
            <NeuralNetworkAnimation />
          ) : algorithm === "knn" ? (
            <KnnAnimation />
          ) : algorithm === "gradient-boosting" ? (
            <GradientBoostingAnimation />
          ) : algorithm === "cnn" ? (
            <Suspense fallback={<div className="fixed inset-0 bg-bg" />}>
              <CnnAnimation />
            </Suspense>
          ) : (
            <TreeAnimation />
          ))}
        {page === "results" && <Results />}
      </motion.div>
    </AnimatePresence>
  );
}

export default App;
