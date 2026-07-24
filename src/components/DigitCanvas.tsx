import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import clsx from "clsx";
import { RAW_SIZE } from "../lib/mnist";

// Internal drawing resolution — big enough for smooth strokes under the
// pointer; getRaw() then re-renders the ink down to RAW_SIZE (28) using the
// same crop/scale/center-of-mass normalization the real MNIST dataset was
// built with, since the model only ever saw digits laid out that way.
const CANVAS_SIZE = 280;
const BRUSH_WIDTH = 22;
const INK_THRESHOLD = 24;
// MNIST digits are scaled so their longest side fills 20 of the 28 output pixels.
const TARGET_DIGIT_SPAN = 20;

export interface DigitCanvasHandle {
  clear: () => void;
  /** Crops to the drawn ink, rescales, and re-centers by center of mass into a 28x28 buffer, MNIST-style. */
  getRaw: () => Uint8Array;
}

/**
 * Reproduces MNIST's own normalization: find the drawn ink's bounding box,
 * scale it so its longest side is ~20px, then translate it so its intensity
 * centroid sits at the center of a 28x28 field. Without this, a hand-drawn
 * digit's size and position rarely match what the model was trained on, and
 * predictions degrade badly even though the strokes look fine to a human.
 */
function normalizeToMnistLayout(source: HTMLCanvasElement): Uint8Array {
  const out = new Uint8Array(RAW_SIZE * RAW_SIZE);
  const srcCtx = source.getContext("2d");
  if (!srcCtx) return out;

  const { data } = srcCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  let minX = CANVAS_SIZE;
  let minY = CANVAS_SIZE;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < CANVAS_SIZE; y++) {
    for (let x = 0; x < CANVAS_SIZE; x++) {
      if (data[(y * CANVAS_SIZE + x) * 4] > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return out; // nothing drawn

  // Pad the bounding box a little so strokes right at its edge don't get clipped.
  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const marginX = Math.round(boxW * 0.08) + 2;
  const marginY = Math.round(boxH * 0.08) + 2;
  const bx = Math.max(0, minX - marginX);
  const by = Math.max(0, minY - marginY);
  const bw = Math.min(CANVAS_SIZE, maxX + marginX + 1) - bx;
  const bh = Math.min(CANVAS_SIZE, maxY + marginY + 1) - by;

  // Scale straight from the high-res crop down to a TARGET_DIGIT_SPAN-tall/wide
  // box in 28px output space, letting the canvas do the (smoothed) resampling.
  const scale = TARGET_DIGIT_SPAN / Math.max(bw, bh);
  const destW = Math.max(1, Math.round(bw * scale));
  const destH = Math.max(1, Math.round(bh * scale));

  const staged = document.createElement("canvas");
  staged.width = RAW_SIZE;
  staged.height = RAW_SIZE;
  const stagedCtx = staged.getContext("2d")!;
  stagedCtx.fillStyle = "#000000";
  stagedCtx.fillRect(0, 0, RAW_SIZE, RAW_SIZE);
  stagedCtx.imageSmoothingEnabled = true;
  stagedCtx.imageSmoothingQuality = "high";
  // A touch of blur nudges our hard vector strokes toward MNIST's naturally
  // anti-aliased ones, which the model's filters were tuned against.
  stagedCtx.filter = "blur(1px)";
  stagedCtx.drawImage(source, bx, by, bw, bh, (RAW_SIZE - destW) / 2, (RAW_SIZE - destH) / 2, destW, destH);
  stagedCtx.filter = "none";

  // Re-center by intensity centroid, same as MNIST's own construction.
  const stagedData = stagedCtx.getImageData(0, 0, RAW_SIZE, RAW_SIZE).data;
  let mass = 0;
  let momentX = 0;
  let momentY = 0;
  for (let y = 0; y < RAW_SIZE; y++) {
    for (let x = 0; x < RAW_SIZE; x++) {
      const v = stagedData[(y * RAW_SIZE + x) * 4];
      mass += v;
      momentX += v * x;
      momentY += v * y;
    }
  }
  let dx = 0;
  let dy = 0;
  if (mass > 0) {
    dx = Math.round(RAW_SIZE / 2 - momentX / mass);
    dy = Math.round(RAW_SIZE / 2 - momentY / mass);
  }

  const final = document.createElement("canvas");
  final.width = RAW_SIZE;
  final.height = RAW_SIZE;
  const finalCtx = final.getContext("2d")!;
  finalCtx.fillStyle = "#000000";
  finalCtx.fillRect(0, 0, RAW_SIZE, RAW_SIZE);
  finalCtx.drawImage(staged, dx, dy);

  const finalData = finalCtx.getImageData(0, 0, RAW_SIZE, RAW_SIZE).data;
  for (let i = 0; i < RAW_SIZE * RAW_SIZE; i++) out[i] = finalData[i * 4];
  return out;
}

interface DigitCanvasProps {
  onChange: (hasInk: boolean) => void;
  className?: string;
}

export const DigitCanvas = forwardRef<DigitCanvasHandle, DigitCanvasProps>(function DigitCanvas(
  { onChange, className },
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const [isEmpty, setIsEmpty] = useState(true);

  const paintBackground = useCallback((ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) paintBackground(ctx);
  }, [paintBackground]);

  function getPos(e: ReactPointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * CANVAS_SIZE,
      y: ((e.clientY - rect.top) / rect.height) * CANVAS_SIZE,
    };
  }

  function handlePointerDown(e: ReactPointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);
    drawingRef.current = true;
    const pos = getPos(e);
    lastPointRef.current = pos;
    setIsEmpty(false);
    onChange(true);

    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function handlePointerMove(e: ReactPointerEvent<HTMLCanvasElement>) {
    if (!drawingRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const pos = getPos(e);
    const last = lastPointRef.current ?? pos;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = BRUSH_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastPointRef.current = pos;
  }

  function endStroke(e: ReactPointerEvent<HTMLCanvasElement>) {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastPointRef.current = null;
    if (canvasRef.current?.hasPointerCapture(e.pointerId)) {
      canvasRef.current.releasePointerCapture(e.pointerId);
    }
  }

  useImperativeHandle(ref, () => ({
    clear() {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      paintBackground(ctx);
      setIsEmpty(true);
      onChange(false);
    },
    getRaw() {
      const canvas = canvasRef.current;
      if (!canvas) return new Uint8Array(RAW_SIZE * RAW_SIZE);
      return normalizeToMnistLayout(canvas);
    },
  }));

  return (
    <div className={clsx("relative", className)}>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endStroke}
        onPointerLeave={endStroke}
        onPointerCancel={endStroke}
        className="aspect-square w-full touch-none rounded-2xl bg-black shadow-soft"
        style={{ cursor: "crosshair" }}
      />
      {isEmpty && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-medium text-white/25">Draw a digit here</span>
        </div>
      )}
    </div>
  );
});
