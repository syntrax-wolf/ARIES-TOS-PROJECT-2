import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "./state/store";
import { AlgorithmSelect } from "./pages/AlgorithmSelect";
import { Hyperparameters } from "./pages/Hyperparameters";
import { TreeAnimation } from "./pages/TreeAnimation";
import { Results } from "./pages/Results";

function App() {
  const page = useSylvaStore((s) => s.page);

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
        {page === "animation" && <TreeAnimation />}
        {page === "results" && <Results />}
      </motion.div>
    </AnimatePresence>
  );
}

export default App;
