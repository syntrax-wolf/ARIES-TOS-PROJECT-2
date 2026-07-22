import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "./state/store";
import { AlgorithmSelect } from "./pages/AlgorithmSelect";
import { Hyperparameters } from "./pages/Hyperparameters";
import { TreeAnimation } from "./pages/TreeAnimation";
import { ForestAnimation } from "./pages/ForestAnimation";
import { NeuralNetworkAnimation } from "./pages/NeuralNetworkAnimation";
import { KnnAnimation } from "./pages/KnnAnimation";
import { GradientBoostingAnimation } from "./pages/GradientBoostingAnimation";
import { CnnAnimation } from "./pages/CnnAnimation";
import { Results } from "./pages/Results";

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
            <CnnAnimation />
          ) : (
            <TreeAnimation />
          ))}
        {page === "results" && <Results />}
      </motion.div>
    </AnimatePresence>
  );
}

export default App;
