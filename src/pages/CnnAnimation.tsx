import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { useSylvaStore } from "../state/store";
import { forwardCnn, type CnnForward } from "../lib/cnn";
import { downsampleImage } from "../lib/mnist";
import { DigitCanvas, type DigitCanvasHandle } from "../components/DigitCanvas";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { CnnScene3D } from "../three/CnnScene3D";
import { STAGE_LABELS } from "../three/cnnGeometry";

const STAGE_GAP_MS = 650;
const HOLD_MS = 1500;
const START_DELAY_MS = 250;

export function CnnAnimation() {
  const cnnWeights = useSylvaStore((s) => s.cnnWeights);
  const setPage = useSylvaStore((s) => s.setPage);

  const canvasRef = useRef<DigitCanvasHandle>(null);
  const [hasInk, setHasInk] = useState(false);
  const [fwd, setFwd] = useState<CnnForward | null>(null);
  const [stage, setStage] = useState(-1);
  const [grown, setGrown] = useState(false);
  const [runId, setRunId] = useState(0);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPage("hyperparams");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPage]);

  useEffect(() => {
    if (!fwd) return;
    setStage(-1);
    setGrown(false);
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let s = 0; s < STAGE_LABELS.length; s++) {
      timers.push(setTimeout(() => setStage(s), START_DELAY_MS + s * STAGE_GAP_MS));
    }
    timers.push(setTimeout(() => setGrown(true), START_DELAY_MS + STAGE_LABELS.length * STAGE_GAP_MS + HOLD_MS));
    return () => timers.forEach(clearTimeout);
    // runId forces a fresh timeline even if the same digit is predicted twice in a row
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  function handlePredict() {
    if (!cnnWeights || !canvasRef.current) return;
    const raw = canvasRef.current.getRaw();
    const features = downsampleImage(raw);
    setFwd(forwardCnn(cnnWeights, features));
    setRunId((id) => id + 1);
  }

  function handleClear() {
    canvasRef.current?.clear();
    setHasInk(false);
    setFwd(null);
    setStage(-1);
    setGrown(false);
  }

  if (!cnnWeights) {
    return <div className="fixed inset-0 bg-bg" />;
  }

  return (
    <div className="fixed inset-0 overflow-hidden bg-bg">
      <div className="absolute left-1/2 top-6 z-20 -translate-x-1/2 rounded-full bg-amber-soft px-4 py-1.5 text-[12px] font-medium uppercase tracking-wide text-amber">
        Pre-trained · for fun, not ranked
      </div>

      {fwd ? (
        <CnnScene3D key={runId} fwd={fwd} stage={stage} />
      ) : (
        <div className="flex h-full items-center justify-center px-6 text-center">
          <p className="max-w-sm text-[15px] leading-relaxed text-ink-faint">
            Draw a digit on the pad, then watch it travel through every convolution, pool, and dense layer in 3D.
          </p>
        </div>
      )}

      <motion.div
        layout
        transition={{ duration: 0.5, ease: "easeInOut" }}
        className={clsx("absolute z-20", fwd ? "left-6 top-24 w-60" : "left-1/2 top-1/2 w-80 -translate-x-1/2 -translate-y-1/2")}
      >
        <Card className="p-5">
          <h2 className="mb-3 text-base font-semibold text-ink">Draw a digit</h2>
          <DigitCanvas ref={canvasRef} onChange={setHasInk} className="mb-4" />
          <div className="flex gap-2">
            <Button variant="secondary" size="md" onClick={handleClear} className="flex-1">
              Clear
            </Button>
            <Button variant="primary" size="md" onClick={handlePredict} disabled={!hasInk} className="flex-1">
              Predict
            </Button>
          </div>
        </Card>
      </motion.div>

      <AnimatePresence>
        {grown && (
          <motion.div
            className="fixed inset-x-0 bottom-10 z-20 flex justify-center gap-3"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 14 }}
            transition={{ duration: 0.4 }}
          >
            <Button variant="secondary" size="lg" onClick={handleClear}>
              Draw another
            </Button>
            <Button variant="primary" size="lg" onClick={() => setPage("results")}>
              View results →
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
