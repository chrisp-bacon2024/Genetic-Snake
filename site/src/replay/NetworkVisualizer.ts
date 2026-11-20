import { fonts, lerpColor, rgb, theme } from "../styles/theme";
import { inputFeatureColor } from "./neuronColors";
import type { DirectionName, LiteReplayFrame } from "./types";

type Rgb = readonly [number, number, number];

const RAY_COUNT = 8;
const FEATURE_OFFSETS = {
  rays: [0, 24] as const,
  food: [24, 5] as const,
  head: [29, 4] as const,
  tail: [33, 4] as const,
  lookahead: [37, 4] as const,
  space: [41, 3] as const,
};

// Colours are read when drawing (not at import), so they follow the theme.
const FEATURE_SECTIONS = [
  { key: "food" as const, label: "Food", color: (): Rgb => theme.neuronInputFood },
  { key: "head" as const, label: "Head dir", color: (): Rgb => theme.controlActive },
  { key: "tail" as const, label: "Tail dir", color: (): Rgb => theme.controlActive },
  { key: "lookahead" as const, label: "Lookahead", color: (): Rgb => theme.neuronInputFood },
  { key: "space" as const, label: "Space", color: (): Rgb => theme.neuronInputBody },
];

/** When the featured player stacks into one column (the 800px breakpoint
 *  in styles/layout.css), the panel sits under the board, so it uses the
 *  compact layout to fit on screen with it. Side by side, it uses the full
 *  layout, which matches the board's height. */
const STACKED = "(max-width: 800px)";

const OUTPUT_DIRECTIONS: DirectionName[] = ["UP", "DOWN", "LEFT", "RIGHT"];
const INPUT_LABELS = ["Wall", "Food", "Body"];

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function featureRowColor(base: Rgb, value: number): Rgb {
  return lerpColor(theme.neuronInactive, base, clamp(value));
}

function activationColor(value: number): Rgb {
  return lerpColor(theme.neuronInactive, theme.neuronActive, clamp(value));
}

function directionGlyph(direction: DirectionName): string {
  switch (direction) {
    case "UP":
      return "^";
    case "DOWN":
      return "v";
    case "LEFT":
      return "<";
    case "RIGHT":
      return ">";
  }
}

export class NetworkVisualizer {
  private readonly ctx: CanvasRenderingContext2D;
  /** Size in CSS pixels; the canvas itself is scaled by the pixel ratio. */
  private panelWidth = 280;
  private panelHeight = 0;
  private pixelRatio = 1;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    this.ctx = ctx;
  }

  private get compact(): boolean {
    return window.matchMedia(STACKED).matches;
  }

  resize(width: number): void {
    this.panelWidth = Math.max(220, Math.round(width));
    this.setHeight(this.heightFor(64), true);
  }

  /** Resizes the backing store only when needed (resizing clears the canvas
   *  and resets its transform). Draws at the device's pixel ratio, capped at
   *  2, so the panel stays sharp on phones. */
  private setHeight(height: number, force = false): void {
    const ratio = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    if (!force && height === this.panelHeight && ratio === this.pixelRatio) return;
    this.panelHeight = height;
    this.pixelRatio = ratio;
    this.canvas.width = Math.round(this.panelWidth * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  private heightFor(hiddenCount: number): number {
    return this.compact ? this.compactHeight(hiddenCount) : this.estimateHeight(hiddenCount);
  }

  draw(frame: LiteReplayFrame): void {
    const { ctx } = this;
    const width = this.panelWidth;
    const margin = 12;
    const usable = width - margin * 2;

    if (!frame.inputs?.length || !frame.outputs?.length) {
      this.setHeight(this.heightFor(64));
      const height = this.panelHeight;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = rgb(theme.panel);
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = rgb(theme.textDim);
      ctx.font = fonts.serif(400, 13);
      ctx.textAlign = "center";
      ctx.fillText("Neural data unavailable", width / 2, height / 2);
      ctx.fillText("(use full replay export)", width / 2, height / 2 + 18);
      return;
    }

    const hidden = frame.hidden_layers?.[0] ?? [];
    this.setHeight(this.heightFor(hidden.length));

    ctx.clearRect(0, 0, width, this.panelHeight);
    ctx.fillStyle = rgb(theme.panel);
    ctx.fillRect(0, 0, width, this.panelHeight);

    if (this.compact) {
      this.drawCompact(frame.inputs, hidden, frame.outputs, frame.direction);
      return;
    }

    let y = 10;
    y = this.drawTitle(y, width);
    y = this.drawRaySection(frame.inputs, y, usable, margin);
    for (const section of FEATURE_SECTIONS) {
      const [start, count] = FEATURE_OFFSETS[section.key];
      y = this.drawFeatureSection(frame.inputs, section.label, count, start, section.color(), y, usable, margin);
    }
    y = this.drawHiddenSection(hidden, y, usable, margin);
    this.drawOutputSection(frame.outputs, frame.direction, y, width);
  }

  private estimateHeight(hiddenCount: number): number {
    const hiddenRows = Math.ceil(hiddenCount / Math.max(1, Math.floor((this.panelWidth - 24) / 12)));
    return 520 + hiddenRows * 10;
  }

  // ---- Compact layout (phones) ----
  //
  //  Neural net               Stronger colour = stronger signal
  //  Rays (8)            Other inputs
  //  Wall  o o o o o     Food      o o o o o
  //  Food  o o o o o     Head dir  o o o o
  //  Body  o o o o o     ...
  //  Hidden layer (64)
  //  oooooooooooooooooooooooooooooooo
  //  Output        [^]         Navy = the move it makes
  //             [<][v][>]

  private static readonly C = {
    margin: 12,
    titleH: 24,
    labelH: 13,
    dotPitch: 11,
    dotRadius: 3.5,
    rowH: 12,
    hiddenPitch: 8,
    hiddenRadius: 2.5,
    arrow: 22,
    arrowGap: 4,
    sectionGap: 10,
  };

  private compactHiddenPerRow(): number {
    const { margin, hiddenPitch } = NetworkVisualizer.C;
    return Math.max(1, Math.floor((this.panelWidth - margin * 2 + 2) / hiddenPitch));
  }

  private compactHeight(hiddenCount: number): number {
    const c = NetworkVisualizer.C;
    const inputs = c.labelH + FEATURE_SECTIONS.length * c.rowH;
    const hiddenRows = Math.ceil(hiddenCount / this.compactHiddenPerRow());
    const hidden = c.labelH + hiddenRows * c.hiddenPitch;
    const output = c.labelH + c.arrow * 2 + c.arrowGap;
    return Math.ceil(8 + c.titleH + inputs + c.sectionGap + hidden + c.sectionGap + output + 10);
  }

  private drawCompact(inputs: number[], hidden: number[], outputs: number[], chosen: DirectionName): void {
    const { ctx } = this;
    const c = NetworkVisualizer.C;
    const width = this.panelWidth;
    const left = c.margin;
    const right = width - c.margin;
    const label = (text: string, x: number, y: number, align: CanvasTextAlign = "left", weight = 600): void => {
      ctx.fillStyle = rgb(theme.textDim);
      ctx.font = fonts.small(weight, 10);
      ctx.textAlign = align;
      ctx.fillText(text, x, y);
    };

    // Title and key.
    let y = 8;
    ctx.fillStyle = rgb(theme.text);
    ctx.font = fonts.serif(700, 14);
    ctx.textAlign = "left";
    ctx.fillText("Neural net", left, y + 13);
    label("Stronger colour = stronger signal", right, y + 12, "right", 400);
    y += c.titleH;

    // Inputs: rays on the left, the other inputs on the right.
    const inputsTop = y;
    label(`Rays (${RAY_COUNT})`, left, inputsTop + 9);
    const rayStartX = left + 28 + c.dotRadius;
    for (let row = 0; row < 3; row += 1) {
      const cy = inputsTop + c.labelH + row * c.rowH + c.rowH / 2;
      label(INPUT_LABELS[row], left, cy + 3.5, "left", 400);
      for (let ray = 0; ray < RAY_COUNT; ray += 1) {
        const value = inputs[ray * 3 + row] ?? 0;
        this.fillCircle(rayStartX + ray * c.dotPitch, cy, c.dotRadius, inputFeatureColor(row, value));
      }
    }

    const maxDots = Math.max(...FEATURE_SECTIONS.map((section) => FEATURE_OFFSETS[section.key][1]));
    const featureX = right - (maxDots - 1) * c.dotPitch - c.dotRadius;
    const featureLabelX = featureX - c.dotRadius - 6;
    label("Other inputs", featureLabelX, inputsTop + 9, "right");
    FEATURE_SECTIONS.forEach((section, row) => {
      const [start, count] = FEATURE_OFFSETS[section.key];
      const cy = inputsTop + c.labelH + row * c.rowH + c.rowH / 2;
      label(section.label, featureLabelX, cy + 3.5, "right", 400);
      const color = section.color();
      for (let i = 0; i < count; i += 1) {
        this.fillCircle(featureX + i * c.dotPitch, cy, c.dotRadius, featureRowColor(color, inputs[start + i] ?? 0));
      }
    });
    y = inputsTop + c.labelH + FEATURE_SECTIONS.length * c.rowH + c.sectionGap;

    // Hidden layer, full width.
    label(`Hidden layer (${hidden.length})`, left, y + 9);
    y += c.labelH;
    const perRow = this.compactHiddenPerRow();
    for (let i = 0; i < hidden.length; i += 1) {
      const row = Math.floor(i / perRow);
      const col = i % perRow;
      this.fillCircle(
        left + c.hiddenRadius + col * c.hiddenPitch,
        y + c.hiddenRadius + row * c.hiddenPitch,
        c.hiddenRadius,
        activationColor(hidden[i]),
      );
    }
    y += Math.ceil(hidden.length / perRow) * c.hiddenPitch + c.sectionGap;

    // Output arrows, centred, with their key.
    label("Output", left, y + 9);
    label("Navy = the move it makes", right, y + 9, "right", 400);
    y += c.labelH;
    const centerX = width / 2;
    const step = c.arrow + c.arrowGap;
    const positions: Record<DirectionName, { x: number; y: number }> = {
      UP: { x: centerX, y: y + c.arrow / 2 },
      LEFT: { x: centerX - step, y: y + step + c.arrow / 2 },
      DOWN: { x: centerX, y: y + step + c.arrow / 2 },
      RIGHT: { x: centerX + step, y: y + step + c.arrow / 2 },
    };
    const maxOutput = Math.max(...outputs, 0.001);
    OUTPUT_DIRECTIONS.forEach((direction, index) => {
      const normalized = Math.max(0, (outputs[index] ?? 0) / maxOutput);
      const { x, y: cy } = positions[direction];
      this.drawArrowButton(x, cy, c.arrow, direction, normalized, direction === chosen);
    });
  }

  // ---- Full layout ----

  private drawTitle(y: number, width: number): number {
    const { ctx } = this;
    ctx.fillStyle = rgb(theme.text);
    ctx.font = fonts.serif(700, 14);
    ctx.textAlign = "center";
    ctx.fillText("Neural Net", width / 2, y + 12);
    return y + 24;
  }

  private drawLayerLabel(text: string, y: number, width: number, suffix?: string): number {
    const { ctx } = this;
    ctx.fillStyle = rgb(theme.textDim);
    ctx.font = fonts.small(600, 11);
    ctx.textAlign = "center";
    ctx.fillText(text, width / 2, y + 10);
    let bottom = y + 14;
    if (suffix) {
      ctx.font = fonts.small(400, 9);
      ctx.fillText(suffix, width / 2, bottom + 10);
      bottom += 12;
    }
    return bottom;
  }

  private drawRaySection(inputs: number[], y: number, usable: number, margin: number): number {
    const { ctx } = this;
    const width = this.panelWidth;
    y = this.drawLayerLabel(`Rays (${RAY_COUNT})`, y, width, "wall / food / body");
    y += 6;

    const radius = 4;
    const rowHeight = 15;
    const colGap = 6;
    const gridWidth = RAY_COUNT * (radius * 2 + colGap) - colGap;
    const startX = margin + (usable - gridWidth) / 2 + radius;
    const firstRowCenterY = y + radius;

    ctx.font = fonts.small(400, 9);
    ctx.fillStyle = rgb(theme.textDim);
    ctx.textAlign = "right";
    const legendX = startX - radius - 6;
    ctx.fillText("Proximity to", legendX, firstRowCenterY - rowHeight / 2 - 2);
    for (let row = 0; row < 3; row += 1) {
      ctx.fillText(INPUT_LABELS[row], legendX, firstRowCenterY + row * rowHeight + 3);
    }

    for (let ray = 0; ray < RAY_COUNT; ray += 1) {
      const x = startX + ray * (radius * 2 + colGap);
      for (let row = 0; row < 3; row += 1) {
        const index = ray * 3 + row;
        const value = inputs[index] ?? 0;
        this.fillCircle(x, firstRowCenterY + row * rowHeight, radius, inputFeatureColor(row, value));
      }
    }

    return firstRowCenterY + rowHeight * 2 + radius * 2 + 12;
  }

  private drawFeatureSection(
    inputs: number[],
    label: string,
    count: number,
    startIndex: number,
    baseColor: Rgb,
    y: number,
    usable: number,
    margin: number,
  ): number {
    const width = this.panelWidth;
    y = this.drawLayerLabel(`${label} (${count})`, y, width);
    y += 8;

    const radius = 4;
    const colGap = 6;
    const pitch = radius * 2 + colGap;
    const rowWidth = count * pitch - colGap;
    const startX = margin + (usable - rowWidth) / 2 + radius;
    const centerY = y + radius;

    for (let i = 0; i < count; i += 1) {
      const value = inputs[startIndex + i] ?? 0;
      this.fillCircle(startX + i * pitch, centerY, radius, featureRowColor(baseColor, value));
    }

    return centerY + radius + 12;
  }

  private drawHiddenSection(hidden: number[], y: number, usable: number, margin: number): number {
    const width = this.panelWidth;
    y = this.drawLayerLabel(`Hidden 1 (${hidden.length})`, y, width);
    y += 8;

    const radius = 3;
    const gap = 3;
    const pitch = radius * 2 + gap;
    const perRow = Math.max(1, Math.floor((usable + gap) / pitch));

    for (let i = 0; i < hidden.length; i += 1) {
      const row = Math.floor(i / perRow);
      const col = i % perRow;
      const countInRow = Math.min(perRow, hidden.length - row * perRow);
      const rowWidth = countInRow * pitch - gap;
      const startX = margin + (usable - rowWidth) / 2 + radius;
      const cx = startX + col * pitch;
      const cy = y + row * pitch + radius;
      this.fillCircle(cx, cy, radius, activationColor(hidden[i]));
    }

    const rows = Math.ceil(hidden.length / perRow);
    return y + rows * pitch + 10;
  }

  private drawOutputSection(
    outputs: number[],
    chosen: DirectionName,
    y: number,
    width: number,
  ): number {
    y = this.drawLayerLabel("Output", y, width);
    y += 8;

    const arrowSize = 26;
    const arrowGap = 5;
    const centerX = width / 2;
    const clusterTop = y;

    const layouts: Record<DirectionName, { x: number; y: number }> = {
      UP: { x: centerX, y: clusterTop + arrowSize / 2 },
      DOWN: { x: centerX, y: clusterTop + arrowSize + arrowGap + arrowSize / 2 },
      LEFT: { x: centerX - arrowSize - arrowGap, y: clusterTop + arrowSize + arrowGap + arrowSize / 2 },
      RIGHT: { x: centerX + arrowSize + arrowGap, y: clusterTop + arrowSize + arrowGap + arrowSize / 2 },
    };

    const maxOutput = Math.max(...outputs, 0.001);

    for (const direction of OUTPUT_DIRECTIONS) {
      const index = OUTPUT_DIRECTIONS.indexOf(direction);
      const value = outputs[index] ?? 0;
      const normalized = Math.max(0, value / maxOutput);
      const isChosen = direction === chosen;
      const { x, y: cy } = layouts[direction];
      this.drawArrowButton(x, cy, arrowSize, direction, normalized, isChosen);
    }

    return clusterTop + arrowSize * 2 + arrowGap + 10;
  }

  private drawArrowButton(
    cx: number,
    cy: number,
    size: number,
    direction: DirectionName,
    activation: number,
    isChosen: boolean,
  ): void {
    const { ctx } = this;
    const half = size / 2;
    const x = cx - half;
    const y = cy - half;
    const radius = Math.max(4, size / 7);

    if (isChosen) {
      ctx.fillStyle = rgb(theme.controlActiveGlow, 0.35);
      ctx.beginPath();
      ctx.roundRect(x - 2, y - 2, size + 4, size + 4, radius + 2);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.roundRect(x, y, size, size, radius);

    if (isChosen) {
      ctx.fillStyle = rgb(theme.controlActive);
      ctx.fill();
      ctx.fillStyle = rgb(theme.background);
    } else {
      // Moves it didn't make shade toward grey, never navy, so the chosen
      // move is the only navy arrow ("Navy = the move it makes").
      ctx.fillStyle = rgb(lerpColor(theme.neuronInactive, theme.rayFaint, clamp(activation)));
      ctx.fill();
      ctx.strokeStyle = rgb(theme.controlBorder);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = rgb(lerpColor(theme.textDim, theme.text, clamp(activation)));
    }

    ctx.font = fonts.small(700, Math.max(12, size / 2));
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(directionGlyph(direction), cx, cy + 1);
    ctx.textBaseline = "alphabetic";
  }

  private fillCircle(x: number, y: number, radius: number, color: Rgb): void {
    const { ctx } = this;
    ctx.fillStyle = rgb(color);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}
