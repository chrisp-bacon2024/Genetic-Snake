import { lerpColor, theme } from "../styles/theme";

type Rgb = readonly [number, number, number];

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Match pygame ui/input_feature_color.py — rows 0=wall, 1=food, 2=body. */
export function inputFeatureColor(featureRow: number, value: number): Rgb {
  const bases = [theme.neuronInputWall, theme.neuronInputFood, theme.neuronInputBody];
  return lerpColor(theme.neuronInactive, bases[featureRow] ?? theme.neuronInactive, clamp(value));
}

/** Vision ray on the board: like inputFeatureColor, but fades toward a
 *  mid grey rather than the inactive colour, so distant rays stay visible
 *  on the light board. */
export function rayColor(featureRow: number, value: number): Rgb {
  const bases = [theme.neuronInputWall, theme.neuronInputFood, theme.neuronInputBody];
  return lerpColor(theme.rayFaint, bases[featureRow] ?? theme.rayFaint, clamp(value));
}

export function proximityActivation(steps: number | null | undefined): number {
  if (steps === null || steps === undefined || steps <= 0) {
    return 0;
  }
  return 1 / steps;
}
