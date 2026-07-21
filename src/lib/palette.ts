// One color per digit 0-9 — muted and earthy so it sits comfortably in the tree/soil scene.
export const CLASS_COLORS = [
  "#1a7a72", // 0 teal
  "#d98544", // 1 amber
  "#6a6fb0", // 2 indigo
  "#c1554a", // 3 terracotta
  "#b8963a", // 4 mustard
  "#966a94", // 5 mauve
  "#4f7fa3", // 6 steel blue
  "#8a7548", // 7 olive brown
  "#bf7a8c", // 8 rose
  "#5c6670", // 9 slate grey
];
export function classColor(label: number): string {
  return CLASS_COLORS[((label % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];
}

const HEAT_COLD: [number, number, number] = [0xf4, 0xef, 0xe4];
const HEAT_HOT: [number, number, number] = [0x9a, 0x4a, 0x1f];

/** Sequential "how much does the tree look at this pixel" scale, t in [0,1]. */
export function heatColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const [r1, g1, b1] = HEAT_COLD;
  const [r2, g2, b2] = HEAT_HOT;
  const r = Math.round(r1 + (r2 - r1) * clamped);
  const g = Math.round(g1 + (g2 - g1) * clamped);
  const b = Math.round(b1 + (b2 - b1) * clamped);
  return `rgb(${r}, ${g}, ${b})`;
}
