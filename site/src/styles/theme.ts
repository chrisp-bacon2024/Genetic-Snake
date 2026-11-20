/**
 * Canvas colours and fonts.
 *
 * Two kinds of colour:
 * - Brand colours (board, panel, text, controls) come from the
 *   christophermpalmer.com theme, read from its CSS variables at runtime, so
 *   a change to the shared theme also recolours the canvases. The fallbacks
 *   are the theme's values, used if the stylesheet hasn't loaded.
 * - Game colours (snake, apple, vision rays, win/lose) belong to this
 *   project (BRAND_GUIDELINES 7.7) and are defined here. They started as a
 *   copy of src/config.py (the pygame app keeps its dark look); the snake,
 *   apple and ray hues are unchanged, with a few darkened to stay legible on
 *   a light background.
 */

type Rgb = readonly [number, number, number];

const BRAND_FALLBACKS = {
  "--color-surface": "#ffffff",
  "--color-slate-50": "#f8fafc",
  "--color-rule": "#e2e8f0",
  "--color-border": "#cbd5e1",
  "--color-text-muted": "#94a3b8",
  "--color-text-ui": "#334155",
  "--color-text-secondary": "#64748b",
  "--color-navy": "#284a63",
  "--color-sky": "#65b5e8",
  "--color-success-strong": "#15803d",
  "--font-serif": '"Times New Roman", Times, "Liberation Serif", "Nimbus Roman", serif',
} as const;

type BrandToken = keyof typeof BRAND_FALLBACKS;

function parseColor(value: string): Rgb | null {
  const hex = value.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const digits = hex[1].length === 3 ? [...hex[1]].map((d) => d + d).join("") : hex[1];
    return [0, 2, 4].map((i) => parseInt(digits.slice(i, i + 2), 16)) as unknown as Rgb;
  }
  const fn = value.trim().match(/^rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/i);
  return fn ? [Number(fn[1]), Number(fn[2]), Number(fn[3])] : null;
}

let brandCache: Map<BrandToken, string> | null = null;

/** Reads the theme's tokens once (after the stylesheet has applied). */
function token(name: BrandToken): string {
  if (!brandCache) {
    const style = getComputedStyle(document.documentElement);
    brandCache = new Map(
      (Object.keys(BRAND_FALLBACKS) as BrandToken[]).map((key) => [
        key,
        style.getPropertyValue(key).trim() || BRAND_FALLBACKS[key],
      ]),
    );
  }
  return brandCache.get(name)!;
}

function brandColor(name: BrandToken): Rgb {
  return parseColor(token(name)) ?? (parseColor(BRAND_FALLBACKS[name]) as Rgb);
}

export const theme = {
  // ---- Brand (from the shared theme) ----
  get background(): Rgb { return brandColor("--color-slate-50"); },   // board
  get panel(): Rgb { return brandColor("--color-surface"); },         // network panel
  get gridLine(): Rgb { return brandColor("--color-rule"); },
  get text(): Rgb { return brandColor("--color-text-ui"); },
  get textDim(): Rgb { return brandColor("--color-text-secondary"); },
  get controlActive(): Rgb { return brandColor("--color-navy"); },
  get controlActiveGlow(): Rgb { return brandColor("--color-sky"); },
  get controlBorder(): Rgb { return brandColor("--color-border"); },
  get rayFaint(): Rgb { return brandColor("--color-text-muted"); },     // distant vision rays
  get neuronInactive(): Rgb { return brandColor("--color-rule"); },
  get neuronActive(): Rgb { return brandColor("--color-navy"); },
  get accent(): Rgb { return brandColor("--color-sky"); },
  get win(): Rgb { return brandColor("--color-success-strong"); },

  // ---- Game (this project's own palette) ----
  snakeHead: [38, 118, 72] as Rgb,
  snakeTail: [96, 210, 132] as Rgb,
  snakeEye: [232, 248, 236] as Rgb,
  snakePupil: [16, 32, 24] as Rgb,
  food: [220, 58, 52] as Rgb,
  appleShadow: [150, 28, 32] as Rgb,
  appleHighlight: [255, 170, 158] as Rgb,
  appleStem: [92, 58, 28] as Rgb,
  appleLeaf: [56, 138, 62] as Rgb,
  appleOutline: [120, 24, 28] as Rgb,
  gameOver: [185, 28, 28] as Rgb,
  rayWall: [166, 104, 60] as Rgb,
  rayBody: [46, 150, 90] as Rgb,
  rayFood: [220, 58, 52] as Rgb,
  neuronInputWall: [166, 104, 60] as Rgb,
  neuronInputBody: [46, 150, 90] as Rgb,
  neuronInputFood: [220, 58, 52] as Rgb,
};

/** Canvas fonts. Titles and the score use the theme's serif. Labels under
 *  12px use the system sans (BRAND_GUIDELINES 7.8 exception for small
 *  readouts in a dense data display; see site/README.md). */
export const fonts = {
  serif: (weight: number, px: number): string => `${weight} ${px}px ${token("--font-serif")}`,
  small: (weight: number, px: number): string =>
    `${weight} ${px}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`,
};

/** Translucent veil drawn over the board when a run ends. */
export const overlayVeil = "rgba(255, 255, 255, 0.78)";

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
