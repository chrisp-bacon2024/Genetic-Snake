export type DirectionName = "UP" | "DOWN" | "LEFT" | "RIGHT";

export interface LiteReplayFrame {
  tick: number;
  direction: DirectionName;
  snake: [number, number][];
  food: [number, number];
  score: number;
  alive: boolean;
  ate_food: boolean;
  died: boolean;
  starved?: boolean;
  inputs?: number[];
  hidden_layers?: number[][];
  rnn_hidden?: number[];
  outputs?: number[];
}

export interface ReplayDocument {
  version: number;
  grid: { cols: number; rows: number };
  ticks_per_second: number;
  genome?: number[];
  frame_count: number;
  frames: LiteReplayFrame[];
  generation?: number;
  saved_score?: number;
  death_cause?: string;
  lite?: boolean;
}

export interface GenerationEntry {
  generation: number;
  score: number;
  grid_cols: number;
  grid_rows: number;
  death_cause: string;
  frame_count: number;
  ticks_per_second: number;
  lite: boolean;
  path: string;
  narrative: string;
}

export interface SiteManifest {
  grid_generations: GenerationEntry[];
  featured_generations: GenerationEntry[];
  default_featured_generation: number;
}

export interface MetricRow {
  generation: number;
  grid_cols: number;
  grid_rows: number;
  best_fitness: number | null;
  avg_fitness: number | null;
  best_score: number;
  max_score: number;
  avg_score: number;
  best_ever_score: number;
  death_cause: string;
  win_count?: number;
  win_needed?: number;
}

export const OUTPUT_LABELS = ["UP", "DOWN", "LEFT", "RIGHT"] as const;

export const DIRECTION_DELTA: Record<DirectionName, [number, number]> = {
  UP: [0, -1],
  DOWN: [0, 1],
  LEFT: [-1, 0],
  RIGHT: [1, 0],
};

export function relativeRayDeltas(facing: DirectionName): [number, number][] {
  const [forwardX, forwardY] = DIRECTION_DELTA[facing];
  const rightX = -forwardY;
  const rightY = forwardX;
  const backX = -forwardX;
  const backY = -forwardY;
  const leftX = forwardY;
  const leftY = -forwardX;

  const combine = (ax: number, ay: number, bx: number, by: number): [number, number] => {
    let dx = ax + bx;
    let dy = ay + by;
    if (dx !== 0) dx = dx > 0 ? 1 : -1;
    if (dy !== 0) dy = dy > 0 ? 1 : -1;
    return [dx, dy];
  };

  return [
    [forwardX, forwardY],
    combine(forwardX, forwardY, rightX, rightY),
    [rightX, rightY],
    combine(backX, backY, rightX, rightY),
    [backX, backY],
    combine(backX, backY, leftX, leftY),
    [leftX, leftY],
    combine(forwardX, forwardY, leftX, leftY),
  ];
}
