import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "./state/store";
import { Register } from "./pages/Register";
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
  const student = useSylvaStore((s) => s.student);
  const algorithm = useSylvaStore((s) => s.algorithm);

  // Every run is recorded against an entry number, so nothing past registration
  // is reachable without one.
  const resolved: typeof page = student ? page : "register";

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={resolved}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
      >
        {resolved === "register" && <Register />}
        {resolved === "select" && <AlgorithmSelect />}
        {resolved === "hyperparams" && <Hyperparameters />}
        {resolved === "animation" &&
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
        {resolved === "results" && <Results />}
      </motion.div>
    </AnimatePresence>
  );
}

export default App;
