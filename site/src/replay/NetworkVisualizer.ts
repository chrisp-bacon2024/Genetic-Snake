import { lerpColor, rgb, theme } from "../styles/theme";
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

const FEATURE_SECTIONS = [
  { key: "food" as const, label: "Food", color: theme.neuronInputFood },
  { key: "head" as const, label: "Head dir", color: theme.controlActive },
  { key: "tail" as const, label: "Tail dir", color: theme.controlActive },
  { key: "lookahead" as const, label: "Lookahead", color: theme.neuronInputFood },
  { key: "space" as const, label: "Space", color: theme.neuronInputBody },
];

const OUTPUT_DIRECTIONS: DirectionName[] = ["UP", "DOWN", "LEFT", "RIGHT"];
const INPUT_LABELS = ["Wall", "Food", "Body"];

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function inputFeatureColor(featureRow: number, value: number): Rgb {
  const bases = [theme.neuronInputWall, theme.neuronInputFood, theme.neuronInputBody];
  return lerpColor(theme.neuronInactive, bases[featureRow] ?? theme.neuronInactive, clamp(value));
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
  private panelWidth = 280;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    this.ctx = ctx;
  }

  resize(width: number): void {
    this.panelWidth = Math.max(220, width);
    this.canvas.width = this.panelWidth;
    this.canvas.height = this.estimateHeight(64);
  }

  draw(frame: LiteReplayFrame): void {
    const { ctx } = this;
    const width = this.panelWidth;
    const margin = 12;
    const usable = width - margin * 2;

    if (!frame.inputs?.length || !frame.outputs?.length) {
      ctx.clearRect(0, 0, width, this.canvas.height);
      ctx.fillStyle = rgb(theme.panel);
      ctx.fillRect(0, 0, width, this.canvas.height);
      ctx.fillStyle = rgb(theme.textDim);
      ctx.font = "13px IBM Plex Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Neural data unavailable", width / 2, this.canvas.height / 2);
      ctx.fillText("(use full replay export)", width / 2, this.canvas.height / 2 + 18);
      return;
    }

    const hidden = frame.hidden_layers?.[0] ?? [];
    this.canvas.height = this.estimateHeight(hidden.length);

    ctx.clearRect(0, 0, width, this.canvas.height);
    ctx.fillStyle = rgb(theme.panel);
    ctx.fillRect(0, 0, width, this.canvas.height);

    let y = 10;
    y = this.drawTitle(y, width);
    y = this.drawRaySection(frame.inputs, y, usable, margin);
    for (const section of FEATURE_SECTIONS) {
      const [start, count] = FEATURE_OFFSETS[section.key];
      y = this.drawFeatureSection(frame.inputs, section.label, count, start, section.color, y, usable, margin);
    }
    y = this.drawHiddenSection(hidden, y, usable, margin);
    this.drawOutputSection(frame.outputs, frame.direction, y, width);
  }

  private estimateHeight(hiddenCount: number): number {
    const hiddenRows = Math.ceil(hiddenCount / Math.max(1, Math.floor((this.panelWidth - 24) / 12)));
    return 520 + hiddenRows * 10;
  }

  private drawTitle(y: number, width: number): number {
    const { ctx } = this;
    ctx.fillStyle = rgb(theme.text);
    ctx.font = "600 14px IBM Plex Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Neural Net", width / 2, y + 12);
    return y + 24;
  }

  private drawLayerLabel(text: string, y: number, width: number, suffix?: string): number {
    const { ctx } = this;
    ctx.fillStyle = rgb(theme.textDim);
    ctx.font = "11px IBM Plex Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(text, width / 2, y + 10);
    let bottom = y + 14;
    if (suffix) {
      ctx.font = "9px IBM Plex Mono, monospace";
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

    ctx.font = "9px IBM Plex Mono, monospace";
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
      ctx.fillStyle = rgb(activationColor(activation));
      ctx.fill();
      ctx.strokeStyle = rgb(lerpColor(theme.controlBorder, theme.controlActive, activation));
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = rgb(lerpColor(theme.textDim, theme.text, activation));
    }

    ctx.font = `700 ${Math.max(12, size / 2)}px IBM Plex Mono, monospace`;
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
