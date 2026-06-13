/** RGB tuples mirrored from src/config.py */
export const theme = {
  background: [18, 18, 24] as const,
  panel: [24, 24, 32] as const,
  gridLine: [40, 40, 52] as const,
  snakeHead: [38, 118, 72] as const,
  snakeTail: [96, 210, 132] as const,
  snakeEye: [232, 248, 236] as const,
  snakePupil: [16, 32, 24] as const,
  food: [220, 58, 52] as const,
  appleShadow: [150, 28, 32] as const,
  appleHighlight: [255, 170, 158] as const,
  appleStem: [92, 58, 28] as const,
  appleLeaf: [56, 138, 62] as const,
  appleOutline: [120, 24, 28] as const,
  text: [220, 220, 230] as const,
  textDim: [120, 120, 140] as const,
  controlActive: [80, 180, 255] as const,
  gameOver: [255, 100, 100] as const,
  win: [72, 220, 118] as const,
  rayWall: [180, 120, 80] as const,
  rayBody: [80, 180, 120] as const,
  rayFood: [240, 100, 100] as const,
  accent: [100, 200, 255] as const,
  controlBorder: [70, 70, 90] as const,
  controlActiveGlow: [120, 210, 255] as const,
  neuronInactive: [50, 50, 65] as const,
  neuronActive: [100, 200, 255] as const,
  neuronInputWall: [180, 120, 80] as const,
  neuronInputBody: [80, 180, 120] as const,
  neuronInputFood: [240, 100, 100] as const,
};

/** Snake layout fractions from src/config.py */
export const snakeLayout = {
  segmentWidthHeadFrac: 0.44,
  segmentWidthTailFrac: 0.16,
  stubFrac: 0.2,
  headNoseFrac: 0.28,
  jointRadiusScale: 1.05,
} as const;

export function rgb([r, g, b]: readonly [number, number, number], alpha = 1): string {
  return alpha === 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function lerpColor(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  t: number,
): readonly [number, number, number] {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}
