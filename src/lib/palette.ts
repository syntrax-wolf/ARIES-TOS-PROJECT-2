export const CLASS_COLORS = ["#17796f", "#d97b3f", "#5b6bb0", "#b8473d"];
export const CLASS_COLORS_SOFT = ["#dcefec", "#fbe9da", "#e6e8f7", "#f6dedb"];

export function classColor(label: number): string {
  return CLASS_COLORS[((label % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];
}

export function classColorSoft(label: number): string {
  return CLASS_COLORS_SOFT[((label % CLASS_COLORS_SOFT.length) + CLASS_COLORS_SOFT.length) % CLASS_COLORS_SOFT.length];
}
