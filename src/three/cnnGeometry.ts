import * as THREE from "three";
import type { Grid } from "../lib/cnn";
import { classColor, heatColor } from "../lib/palette";

// Shared layout constants for the 3D CNN forward-pass scene. Stages line up
// along -Z (the "tunnel" the camera looks down); within a stage, channels
// stack vertically as floors of a tower (Y), and each floor is an X/Z grid of
// bars whose height encodes activation strength.
export const STAGE_GAP = 7;
export const FOOTPRINT = 8.6;
export const TOWER_HEIGHT = 9;
export const STAGE_LABELS = ["Digit", "Pad", "Convolve", "Pool", "Pad", "Convolve", "Pool", "Flatten", "Dense", "Decide"];
/** From this stage on, the camera swings from its top-down "watch the filters" view to a side profile. */
export const FLATTEN_STAGE_INDEX = STAGE_LABELS.indexOf("Flatten");

export function gridDims(grid: Grid) {
  return { H: grid.length, W: grid[0].length, C: grid[0][0].length };
}

export function maxAbsOf(grid: Grid): number {
  let m = 1e-9;
  for (const row of grid) for (const cell of row) for (const v of cell) m = Math.max(m, Math.abs(v));
  return m;
}

function floorY(channel: number, C: number): number {
  if (C <= 1) return 0;
  return (channel - (C - 1) / 2) * (TOWER_HEIGHT / (C - 1));
}

export function cellPosition(row: number, col: number, channel: number, H: number, W: number, C: number, stageZ: number) {
  const pitch = FOOTPRINT / Math.max(H, W);
  const x = (col - (W - 1) / 2) * pitch;
  const z = stageZ + (row - (H - 1) / 2) * pitch;
  const y = floorY(channel, C);
  return { x, y, z, pitch };
}

/** Reshapes a 14x14 (or similar) plain intensity grid into a single-channel Grid. */
export function toSingleChannelGrid(values: number[][]): Grid {
  return values.map((row) => row.map((v) => [v]));
}

/** Reshapes a flat vector into a roughly-square single-channel grid, padding with zeros if needed. */
export function reshapeToGrid(values: number[], cols: number): Grid {
  const rows = Math.ceil(values.length / cols);
  const grid: Grid = [];
  for (let r = 0; r < rows; r++) {
    const row: number[][] = [];
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      row.push([idx < values.length ? values[idx] : 0]);
    }
    grid.push(row);
  }
  return grid;
}

export const PAD_COLOR = new THREE.Color("#cdd1c2");
const INK_LOW = new THREE.Color("#f4efe4");
const INK_HIGH = new THREE.Color("#241f19");

export function inkColor(t: number): THREE.Color {
  return INK_LOW.clone().lerp(INK_HIGH, Math.max(0, Math.min(1, t)));
}

const heatColorCache = new Map<number, THREE.Color>();
export function activationColor(t: number): THREE.Color {
  const key = Math.round(Math.max(0, Math.min(1, t)) * 255);
  let color = heatColorCache.get(key);
  if (!color) {
    color = new THREE.Color(heatColor(key / 255));
    heatColorCache.set(key, color);
  }
  return color;
}

const classColorCache = new Map<number, THREE.Color>();
export function classThreeColor(i: number): THREE.Color {
  let color = classColorCache.get(i);
  if (!color) {
    color = new THREE.Color(classColor(i));
    classColorCache.set(i, color);
  }
  return color;
}

export function easeOutBack(x: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  const t = Math.max(0, Math.min(1, x));
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}
