import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { Grid } from "../lib/cnn";
import { activationColor, cellPosition, easeOutBack, gridDims, inkColor, maxAbsOf, PAD_COLOR } from "./cnnGeometry";

const POP_DURATION = 0.5; // seconds for a single bar's pop-in
const MAX_STAGGER = 0.45; // seconds spread across the whole grid's diagonal wave
const PARKED_SCALE = 0.0001;

export type GridVariant = "input" | "activation" | "pad";

interface GridLayerProps {
  grid: Grid;
  stageZ: number;
  revealed: boolean;
  variant: GridVariant;
  /** For variant "pad": how the non-border cells (real data) should be colored. */
  innerVariant?: "input" | "activation";
  /** For variant "pad": border width in cells that counts as zero-padding. */
  pad?: number;
}

const geometry = new THREE.BoxGeometry(1, 1, 1);
geometry.translate(0, 0.5, 0);

interface CellInfo {
  row: number;
  col: number;
  ch: number;
  value: number;
  isPad: boolean;
}

export function GridLayer({ grid, stageZ, revealed, variant, innerVariant = "activation", pad = 1 }: GridLayerProps) {
  const { H, W, C } = gridDims(grid);
  const count = H * W * C;
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const revealTimeRef = useRef<number | null>(null);
  const doneRef = useRef(false);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  const maxAbs = useMemo(() => maxAbsOf(grid), [grid]);

  const cells = useMemo(() => {
    const list: CellInfo[] = [];
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        for (let ch = 0; ch < C; ch++) {
          const isPad = variant === "pad" && (r < pad || r >= H - pad || c < pad || c >= W - pad);
          list.push({ row: r, col: c, ch, value: grid[r][c][ch], isPad });
        }
      }
    }
    return list;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid, H, W, C, variant, pad]);

  // Park every instance at its final x/y/z but zero scale, so nothing pops
  // into view at the origin before its reveal animation begins.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    cells.forEach((cell, i) => {
      const { x, y, z } = cellPosition(cell.row, cell.col, cell.ch, H, W, C, stageZ);
      dummy.position.set(x, y, z);
      dummy.scale.set(PARKED_SCALE, PARKED_SCALE, PARKED_SCALE);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    doneRef.current = false;
    revealTimeRef.current = null;
  }, [cells, H, W, C, stageZ, dummy]);

  useEffect(() => {
    if (revealed && revealTimeRef.current === null) revealTimeRef.current = performance.now();
  }, [revealed]);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || !revealed || doneRef.current) return;
    const started = revealTimeRef.current ?? performance.now();
    const elapsed = (performance.now() - started) / 1000;

    let allDone = true;
    cells.forEach((cell, i) => {
      const stagger = ((cell.row + cell.col) / (H + W)) * MAX_STAGGER;
      const local = Math.max(0, Math.min(1, (elapsed - stagger) / POP_DURATION));
      if (local < 1) allDone = false;
      const eased = easeOutBack(local);
      if (eased <= 0) return;

      const { x, y, z, pitch } = cellPosition(cell.row, cell.col, cell.ch, H, W, C, stageZ);
      const t = Math.abs(cell.value) / maxAbs;
      const footprint = pitch * 0.84;

      let heightFactor: number;
      let color: THREE.Color;
      if (cell.isPad) {
        heightFactor = 0.12;
        color = PAD_COLOR;
      } else {
        heightFactor = 0.15 + t * 1.35;
        const useInk = variant === "input" || (variant === "pad" && innerVariant === "input");
        color = useInk ? inkColor(cell.value) : activationColor(t);
      }

      dummy.position.set(x, y, z);
      dummy.scale.set(Math.max(footprint * eased, 0.0001), Math.max(heightFactor * eased, 0.0001), Math.max(footprint * eased, 0.0001));
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      tmpColor.copy(color);
      mesh.setColorAt(i, tmpColor);
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (allDone) doneRef.current = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[geometry, undefined, count]} frustumCulled={false}>
      <meshStandardMaterial roughness={0.55} metalness={0.04} />
    </instancedMesh>
  );
}
