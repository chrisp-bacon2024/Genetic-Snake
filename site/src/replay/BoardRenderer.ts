import { lerpColor, rgb, theme } from "../styles/theme";
import type { DirectionName, LiteReplayFrame } from "./types";
import { relativeRayDeltas } from "./types";

export interface BoardLayout {
  cols: number;
  rows: number;
  cellSize: number;
  offsetX: number;
  offsetY: number;
}

export function boardLayout(cols: number, rows: number, canvasWidth: number): BoardLayout {
  const padding = 8;
  const usable = canvasWidth - padding * 2;
  const cellSize = Math.max(8, Math.floor(usable / Math.max(cols, rows)));
  const boardWidth = cols * cellSize;
  return {
    cols,
    rows,
    cellSize,
    offsetX: Math.floor((canvasWidth - boardWidth) / 2),
    offsetY: padding,
  };
}

export class BoardRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private layout: BoardLayout;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    cols: number,
    rows: number,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    this.ctx = ctx;
    this.layout = boardLayout(cols, rows, canvas.width);
    this.resize(cols, rows);
  }

  resize(cols: number, rows: number): void {
    const cellSize = Math.max(10, Math.floor(220 / Math.max(cols, rows)));
    const padding = 8;
    this.canvas.width = cols * cellSize + padding * 2;
    this.canvas.height = rows * cellSize + padding * 2 + 24;
    this.layout = boardLayout(cols, rows, this.canvas.width);
  }

  draw(frame: LiteReplayFrame, options?: { showRays?: boolean }): void {
    const { ctx, layout } = this;
    const { cols, rows, cellSize, offsetX, offsetY } = layout;

    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = rgb(theme.background);
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.strokeStyle = rgb(theme.gridLine);
    ctx.lineWidth = 1;
    for (let x = 0; x <= cols; x += 1) {
      const px = offsetX + x * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, offsetY);
      ctx.lineTo(px, offsetY + rows * cellSize);
      ctx.stroke();
    }
    for (let y = 0; y <= rows; y += 1) {
      const py = offsetY + y * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(offsetX, py);
      ctx.lineTo(offsetX + cols * cellSize, py);
      ctx.stroke();
    }

    if (options?.showRays) {
      this.drawVisionRays(frame);
    }

    const won = frame.died && frame.score >= cols * rows - 1;
    if (!won) {
      this.drawFood(frame.food);
    }

    this.drawSnake(frame.snake, frame.direction);
    this.drawHud(frame, won);

    if (frame.died) {
      const message = won ? "You Win!" : frame.starved ? "Starved" : "Died";
      this.drawOverlay(message, won);
    }
  }

  private cellCenter(x: number, y: number): [number, number] {
    const { cellSize, offsetX, offsetY } = this.layout;
    return [offsetX + x * cellSize + cellSize / 2, offsetY + y * cellSize + cellSize / 2];
  }

  private drawFood([fx, fy]: [number, number]): void {
    const { ctx, layout } = this;
    const [cx, cy] = this.cellCenter(fx, fy);
    const radius = layout.cellSize * 0.34;

    ctx.fillStyle = rgb(theme.appleShadow);
    ctx.beginPath();
    ctx.arc(cx + 1, cy + 2, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = rgb(theme.food);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = rgb(theme.appleHighlight, 0.55);
    ctx.beginPath();
    ctx.arc(cx - radius * 0.25, cy - radius * 0.25, radius * 0.22, 0, Math.PI * 2);
    ctx.fill();
  }

  private drawSnake(snake: [number, number][], direction: DirectionName): void {
    const { ctx, layout } = this;
    const length = snake.length;
    if (length === 0) return;

    for (let index = length - 1; index >= 0; index -= 1) {
      const [x, y] = snake[index];
      const t = length <= 1 ? 0 : index / (length - 1);
      const color = lerpColor(theme.snakeHead, theme.snakeTail, t);
      const widthFrac = 0.44 - t * (0.44 - 0.16);
      const radius = layout.cellSize * widthFrac * 0.5;
      const [cx, cy] = this.cellCenter(x, y);

      ctx.fillStyle = rgb(color);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    const [hx, hy] = snake[0];
    const [hcx, hcy] = this.cellCenter(hx, hy);
    const [dx, dy] = directionDelta(direction);
    const eyeOffset = layout.cellSize * 0.12;
    const eyeRadius = layout.cellSize * 0.07;
    for (const side of [-1, 1]) {
      const ex = hcx + (-dy * side * eyeOffset) + dx * eyeOffset * 0.35;
      const ey = hcy + (dx * side * eyeOffset) + dy * eyeOffset * 0.35;
      ctx.fillStyle = rgb(theme.snakeEye);
      ctx.beginPath();
      ctx.arc(ex, ey, eyeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = rgb(theme.snakePupil);
      ctx.beginPath();
      ctx.arc(ex + dx * 1.5, ey + dy * 1.5, eyeRadius * 0.45, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private drawHud(frame: LiteReplayFrame, won: boolean): void {
    const { ctx, layout } = this;
    ctx.fillStyle = rgb(theme.text);
    ctx.font = "600 12px IBM Plex Mono, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Score ${frame.score}`, layout.offsetX, this.canvas.height - 8);
    if (won) {
      ctx.fillStyle = rgb(theme.win);
      ctx.textAlign = "right";
      ctx.fillText("Board cleared", layout.offsetX + layout.cols * layout.cellSize, this.canvas.height - 8);
    }
  }

  private drawOverlay(message: string, won: boolean): void {
    const { ctx } = this;
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = rgb(won ? theme.win : theme.gameOver);
    ctx.font = "700 16px IBM Plex Sans, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(message, this.canvas.width / 2, this.canvas.height / 2);
  }

  private drawVisionRays(frame: LiteReplayFrame): void {
    const { ctx, layout } = this;
    if (frame.snake.length === 0) return;

    const [headX, headY] = frame.snake[0];
    const body = new Set(frame.snake.slice(1).map(([x, y]) => `${x},${y}`));
    const deltas = relativeRayDeltas(frame.direction);

    for (const [dx, dy] of deltas) {
      const end = castRay(headX, headY, dx, dy, layout.cols, layout.rows, body);
      const [x0, y0] = this.cellCenter(headX, headY);
      const [x1, y1] = this.cellCenter(end.x, end.y);

      ctx.strokeStyle = rgb(end.kind === "body" ? theme.rayBody : theme.rayWall, 0.85);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
  }
}

function directionDelta(direction: DirectionName): [number, number] {
  switch (direction) {
    case "UP":
      return [0, -1];
    case "DOWN":
      return [0, 1];
    case "LEFT":
      return [-1, 0];
    case "RIGHT":
      return [1, 0];
  }
}

interface RayEnd {
  x: number;
  y: number;
  kind: "wall" | "body";
}

function castRay(
  headX: number,
  headY: number,
  dx: number,
  dy: number,
  cols: number,
  rows: number,
  body: Set<string>,
): RayEnd {
  let x = headX;
  let y = headY;
  let steps = 0;

  while (true) {
    x += dx;
    y += dy;
    steps += 1;
    if (x < 0 || y < 0 || x >= cols || y >= rows) {
      return {
        x: headX + dx * Math.max(0, steps - 1),
        y: headY + dy * Math.max(0, steps - 1),
        kind: "wall",
      };
    }
    if (body.has(`${x},${y}`)) {
      return { x, y, kind: "body" };
    }
  }
}
