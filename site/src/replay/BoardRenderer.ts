import { lerpColor, rgb, snakeLayout, theme } from "../styles/theme";
import type { LiteReplayFrame } from "./types";
import { DIRECTION_DELTA, relativeRayDeltas } from "./types";

type Rgb = readonly [number, number, number];
type Point = [number, number];
type Cell = [number, number];

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
  private boardPixels: number;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    cols: number,
    rows: number,
    boardPixels = 220,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    this.ctx = ctx;
    this.boardPixels = boardPixels;
    this.layout = boardLayout(cols, rows, canvas.width);
    this.resize(cols, rows, boardPixels);
  }

  resize(cols: number, rows: number, boardPixels = this.boardPixels): void {
    this.boardPixels = boardPixels;
    const cellSize = Math.max(12, Math.floor(boardPixels / Math.max(cols, rows)));
    const padding = 8;
    this.canvas.width = cols * cellSize + padding * 2;
    this.canvas.height = rows * cellSize + padding * 2 + 24;
    this.layout = boardLayout(cols, rows, this.canvas.width);
  }

  draw(frame: LiteReplayFrame, options?: { showRays?: boolean }): void {
    const { ctx, layout } = this;
    const { cols, rows } = layout;

    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = rgb(theme.background);
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    this.drawGrid();

    if (options?.showRays) {
      this.drawVisionRays(frame);
    }

    const won = frame.died && frame.score >= cols * rows - 1;
    if (!won) {
      this.drawApple(frame.food[0], frame.food[1]);
    }

    const moveDelta = DIRECTION_DELTA[frame.direction];
    this.drawSnake(frame.snake, moveDelta, frame.died && !won);

    if (won) {
      this.drawWinGaps(frame.snake);
    }

    this.drawHud(frame, won);

    if (frame.died) {
      const message = won ? "You Win!" : frame.starved ? "Starved" : "Died";
      this.drawOverlay(message, won);
    }
  }

  private drawGrid(): void {
    const { ctx, layout } = this;
    const { cols, rows, cellSize, offsetX, offsetY } = layout;
    const boardWidth = cols * cellSize;
    const boardHeight = rows * cellSize;

    ctx.strokeStyle = rgb(theme.gridLine);
    ctx.lineWidth = 1;
    for (let col = 0; col <= cols; col += 1) {
      const x = offsetX + col * cellSize;
      ctx.beginPath();
      ctx.moveTo(x, offsetY);
      ctx.lineTo(x, offsetY + boardHeight);
      ctx.stroke();
    }
    for (let row = 0; row <= rows; row += 1) {
      const y = offsetY + row * cellSize;
      ctx.beginPath();
      ctx.moveTo(offsetX, y);
      ctx.lineTo(offsetX + boardWidth, y);
      ctx.stroke();
    }
  }

  private cellCenter(col: number, row: number): Point {
    const { cellSize, offsetX, offsetY } = this.layout;
    return [
      offsetX + col * cellSize + cellSize / 2,
      offsetY + row * cellSize + cellSize / 2,
    ];
  }

  private snakeSegmentColor(index: number, length: number): Rgb {
    if (length <= 1) return theme.snakeHead;
    const t = index / (length - 1);
    return lerpColor(theme.snakeHead, theme.snakeTail, t);
  }

  private segmentWidthFrac(index: number, length: number): number {
    if (length <= 1) return snakeLayout.segmentWidthHeadFrac;
    const t = index / (length - 1);
    return (
      snakeLayout.segmentWidthHeadFrac +
      (snakeLayout.segmentWidthTailFrac - snakeLayout.segmentWidthHeadFrac) * t
    );
  }

  private segmentAxis(body: Cell[], index: number, moveDx: number, moveDy: number): Point | null {
    if (body.length < 1) return null;
    if (index === 0) return [moveDx, moveDy];
    if (index < body.length - 1) {
      const [cx, cy] = body[index];
      const [nx, ny] = body[index + 1];
      return [nx - cx, ny - cy];
    }
    const [px, py] = body[index - 1];
    const [cx, cy] = body[index];
    return [cx - px, cy - py];
  }

  private segmentNeighborDirs(body: Cell[], index: number): Point[] {
    const dirs: Point[] = [];
    const [cx, cy] = body[index];
    if (index > 0) {
      const [px, py] = body[index - 1];
      dirs.push([px - cx, py - cy]);
    }
    if (index < body.length - 1) {
      const [nx, ny] = body[index + 1];
      dirs.push([nx - cx, ny - cy]);
    }
    return dirs;
  }

  private drawSnake(body: Cell[], moveDelta: Point, dead: boolean): void {
    const length = body.length;
    if (length === 0) return;

    const [moveDx, moveDy] = moveDelta;

    for (let index = 0; index < length - 1; index += 1) {
      this.drawSegmentJoint(body, index, length);
    }

    for (let index = 0; index < length; index += 1) {
      const [col, row] = body[index];
      const color = this.snakeSegmentColor(index, length);
      const axis = this.segmentAxis(body, index, moveDx, moveDy);
      if (!axis) continue;
      this.drawSegmentPad(col, row, color, index, length, body, moveDx, moveDy);
      if (index === 0) {
        this.drawHeadEyes(col, row, moveDx, moveDy, dead);
      } else if (length > 1) {
        this.drawScaleChevron(col, row, axis[0], axis[1], color);
      }
    }
  }

  private drawSegmentJoint(body: Cell[], index: number, length: number): void {
    const { layout } = this;
    const cellSize = layout.cellSize;
    const [ax, ay] = this.cellCenter(body[index][0], body[index][1]);
    const [bx, by] = this.cellCenter(body[index + 1][0], body[index + 1][1]);
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    const widthA = this.segmentWidthFrac(index, length);
    const widthB = this.segmentWidthFrac(index + 1, length);
    const radius = cellSize * ((widthA + widthB) / 2) * snakeLayout.jointRadiusScale;
    const color = lerpColor(
      this.snakeSegmentColor(index, length),
      this.snakeSegmentColor(index + 1, length),
      0.5,
    );
    const ring = color.map((c) => Math.max(0, c - 42)) as [number, number, number];
    const highlight = color.map((c) => Math.min(255, c + 28)) as [number, number, number];

    this.fillCircle(mx, my, Math.max(2, radius), color);
    this.strokeCircle(mx, my, Math.max(2, radius), ring, Math.max(1, cellSize / 18));
    this.fillCircle(mx - radius * 0.22, my - radius * 0.22, Math.max(1, radius * 0.22), highlight);
  }

  private drawSegmentPad(
    col: number,
    row: number,
    color: Rgb,
    index: number,
    length: number,
    body: Cell[],
    moveDx: number,
    moveDy: number,
  ): void {
    const cellSize = this.layout.cellSize;
    const [cx, cy] = this.cellCenter(col, row);
    const halfW = cellSize * this.segmentWidthFrac(index, length);
    const stubLen = cellSize * snakeLayout.stubFrac;
    const coreR = Math.max(2, halfW * 0.78);

    for (const [dx, dy] of this.segmentNeighborDirs(body, index)) {
      this.drawSegmentStub(cx, cy, dx, dy, halfW, stubLen, color);
    }

    if (index === 0 && (moveDx !== 0 || moveDy !== 0)) {
      const noseLen = cellSize * snakeLayout.headNoseFrac;
      this.drawSegmentStub(cx, cy, moveDx, moveDy, halfW, noseLen, color);
    }

    this.fillCircle(cx, cy, coreR, color);
    const shadow = color.map((c) => Math.max(0, c - 34)) as [number, number, number];
    this.strokeCircle(cx, cy, coreR, shadow, Math.max(1, cellSize / 16));
  }

  private drawSegmentStub(
    cx: number,
    cy: number,
    dx: number,
    dy: number,
    halfW: number,
    length: number,
    color: Rgb,
  ): void {
    const span = Math.hypot(dx, dy);
    if (span === 0) return;
    const ux = dx / span;
    const uy = dy / span;
    const px = -uy;
    const py = ux;
    const tipX = cx + ux * length;
    const tipY = cy + uy * length;
    const points: Point[] = [
      [cx + px * halfW, cy + py * halfW],
      [tipX + px * halfW, tipY + py * halfW],
      [tipX, tipY],
      [tipX - px * halfW, tipY - py * halfW],
      [cx - px * halfW, cy - py * halfW],
    ];
    const shadow = color.map((c) => Math.max(0, c - 34)) as [number, number, number];
    this.fillPolygon(points, color);
    this.strokePolygon(points, shadow, Math.max(1, this.layout.cellSize / 18));
    this.fillCircle(tipX, tipY, Math.max(2, halfW * 0.55), color);
  }

  private drawHeadEyes(col: number, row: number, dx: number, dy: number, dead: boolean): void {
    if (dx === 0 && dy === 0) return;
    const cellSize = this.layout.cellSize;
    const [cx, cy] = this.cellCenter(col, row);
    const span = Math.hypot(dx, dy);
    const ux = dx / span;
    const uy = dy / span;
    const px = -uy;
    const py = ux;
    const forward = cellSize * 0.16;
    const side = cellSize * 0.13;
    const eyeR = Math.max(2, cellSize / 8);
    const pupilR = Math.max(1, eyeR / 2);
    const pupilFwd = Math.max(1, eyeR / 3);

    for (const sign of [-1, 1]) {
      const ex = cx + ux * forward + px * side * sign;
      const ey = cy + uy * forward + py * side * sign;
      this.fillCircle(ex, ey, eyeR, theme.snakeEye);
      if (dead) {
        this.drawEyeX(ex, ey, eyeR);
      } else {
        this.fillCircle(ex + ux * pupilFwd, ey + uy * pupilFwd, pupilR, theme.snakePupil);
      }
    }
  }

  private drawEyeX(cx: number, cy: number, radius: number): void {
    const { ctx } = this;
    const half = Math.max(2, radius - 1);
    const width = Math.max(2, this.layout.cellSize / 14);
    ctx.strokeStyle = rgb(theme.snakePupil);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(cx - half, cy - half);
    ctx.lineTo(cx + half, cy + half);
    ctx.moveTo(cx - half, cy + half);
    ctx.lineTo(cx + half, cy - half);
    ctx.stroke();
  }

  private drawScaleChevron(col: number, row: number, dx: number, dy: number, baseColor: Rgb): void {
    const [cx, cy] = this.cellCenter(col, row);
    const points = this.chevronPoints(cx, cy, dx, dy);
    if (!points) return;
    const fill = baseColor.map((c) => Math.min(255, c + 42)) as [number, number, number];
    const edge = baseColor.map((c) => Math.max(0, c - 50)) as [number, number, number];
    this.fillPolygon(points, fill);
    this.strokePolygon(points, edge, Math.max(1, this.layout.cellSize / 16));
  }

  private chevronPoints(cx: number, cy: number, dx: number, dy: number): Point[] | null {
    if (dx === 0 && dy === 0) return null;
    const length = Math.hypot(dx, dy);
    const ux = dx / length;
    const uy = dy / length;
    const px = -uy;
    const py = ux;
    const size = this.layout.cellSize;
    const tip: Point = [cx + ux * size * 0.3, cy + uy * size * 0.3];
    const baseX = cx - ux * size * 0.1;
    const baseY = cy - uy * size * 0.1;
    const half = size * 0.2;
    const left: Point = [baseX + px * half, baseY + py * half];
    const right: Point = [baseX - px * half, baseY - py * half];
    const notch: Point = [cx - ux * size * 0.22, cy - uy * size * 0.22];
    return [tip, left, notch, right];
  }

  private drawWinGaps(body: Cell[]): void {
    const { cols, rows } = this.layout;
    const occupied = new Set(body.map(([x, y]) => `${x},${y}`));
    if (body.length === 0) return;
    const tailColor = this.snakeSegmentColor(body.length - 1, body.length);

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        if (occupied.has(`${col},${row}`)) continue;
        for (const [dx, dy] of [
          [0, -1],
          [0, 1],
          [-1, 0],
          [1, 0],
        ] as Point[]) {
          if (!occupied.has(`${col + dx},${row + dy}`)) continue;
          this.drawWinGapHint(col, row, [-dx, -dy], tailColor);
          break;
        }
      }
    }
  }

  private drawWinGapHint(col: number, row: number, towardBody: Point, color: Rgb): void {
    const cellSize = this.layout.cellSize;
    const [cx, cy] = this.cellCenter(col, row);
    const [dx, dy] = towardBody;
    const span = Math.hypot(dx, dy);
    if (span === 0) return;
    const ux = dx / span;
    const uy = dy / span;
    const length = cellSize * snakeLayout.stubFrac * 0.55;
    const halfW = cellSize * snakeLayout.segmentWidthTailFrac * 0.85;
    const faded = color.map((c, i) =>
      Math.round(c * 0.55 + theme.background[i] * 0.45),
    ) as [number, number, number];
    const tipX = cx + ux * length;
    const tipY = cy + uy * length;
    const px = -uy;
    const py = ux;
    this.fillPolygon(
      [
        [cx + px * halfW, cy + py * halfW],
        [tipX + px * halfW, tipY + py * halfW],
        [tipX, tipY],
        [tipX - px * halfW, tipY - py * halfW],
        [cx - px * halfW, cy - py * halfW],
      ],
      faded,
    );
  }

  private drawApple(col: number, row: number): void {
    const { ctx, layout } = this;
    const cellSize = layout.cellSize;
    const [cx, cy] = this.cellCenter(col, row);
    const size = cellSize;
    const bodyW = size * 0.58;
    const bodyH = size * 0.64;
    const centerY = cy + size * 0.05;
    const top = centerY - bodyH / 2;

    this.fillEllipse(cx + size * 0.03, centerY + size * 0.04, bodyW, bodyH, theme.appleShadow);
    this.fillEllipse(cx, centerY, bodyW, bodyH, theme.food);
    this.strokeEllipse(cx, centerY, bodyW, bodyH, theme.appleOutline, Math.max(1, size / 28));
    this.fillEllipse(
      cx - bodyW * 0.16,
      centerY - bodyH * 0.04 + size * 0.05,
      bodyW * 0.36,
      bodyH * 0.3,
      theme.appleHighlight,
    );

    const dimpleR = Math.max(2, size * 0.06);
    const dimpleY = top + bodyH * 0.1;
    this.fillCircle(cx, dimpleY, dimpleR, theme.appleShadow);

    const stemW = Math.max(2, size * 0.07);
    const stemTopY = top - size * 0.1;
    const stemBaseY = top + size * 0.06;
    ctx.strokeStyle = rgb(theme.appleStem);
    ctx.lineWidth = stemW;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx, stemBaseY);
    ctx.lineTo(cx, stemTopY);
    ctx.stroke();

    const leafBase: Point = [cx - size * 0.02, top + size * 0.02];
    const leafMid: Point = [cx + size * 0.06, top - size * 0.1];
    const leafTip: Point = [cx + size * 0.18, top - size * 0.02];
    this.fillPolygon([leafBase, leafMid, leafTip], theme.appleLeaf);
    const leafVein = theme.appleLeaf.map((c) => Math.max(0, c - 30)) as [number, number, number];
    ctx.strokeStyle = rgb(leafVein);
    ctx.lineWidth = Math.max(1, size / 24);
    ctx.beginPath();
    ctx.moveTo(leafBase[0], leafBase[1]);
    ctx.lineTo(leafTip[0], leafTip[1]);
    ctx.stroke();
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
    const { ctx, layout } = this;
    const boardWidth = layout.cols * layout.cellSize;
    const boardHeight = layout.rows * layout.cellSize;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(layout.offsetX, layout.offsetY, boardWidth, boardHeight);
    ctx.fillStyle = rgb(won ? theme.win : theme.gameOver);
    ctx.font = "700 16px IBM Plex Sans, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(message, layout.offsetX + boardWidth / 2, layout.offsetY + boardHeight / 2);
  }

  private drawVisionRays(frame: LiteReplayFrame): void {
    const { ctx, layout } = this;
    if (frame.snake.length === 0) return;

    const [headX, headY] = frame.snake[0];
    const body = new Set(frame.snake.slice(1).map(([x, y]) => `${x},${y}`));
    const deltas = relativeRayDeltas(frame.direction);
    const lineWidth = Math.max(1, layout.cellSize / 22);
    const endRadius = Math.max(1, layout.cellSize / 16);

    for (const [dx, dy] of deltas) {
      const end = castRay(headX, headY, dx, dy, layout.cols, layout.rows, body);
      const [x0, y0] = this.cellCenter(headX, headY);
      const [x1, y1] = this.cellCenter(end.x, end.y);

      ctx.strokeStyle = rgb(end.kind === "body" ? theme.rayBody : theme.rayWall, 0.85);
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      this.fillCircle(x1, y1, endRadius, end.kind === "body" ? theme.rayBody : theme.rayWall);
    }

    const [foodX, foodY] = frame.food;
    if (foodX !== headX || foodY !== headY) {
      const [x0, y0] = this.cellCenter(headX, headY);
      const [x1, y1] = this.cellCenter(foodX, foodY);
      ctx.strokeStyle = rgb(theme.rayFood, 0.85);
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      this.fillCircle(x1, y1, endRadius, theme.rayFood);
    }
  }

  private fillCircle(x: number, y: number, radius: number, color: Rgb): void {
    const { ctx } = this;
    ctx.fillStyle = rgb(color);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  private strokeCircle(x: number, y: number, radius: number, color: Rgb, width: number): void {
    const { ctx } = this;
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  private fillEllipse(x: number, y: number, w: number, h: number, color: Rgb): void {
    const { ctx } = this;
    ctx.fillStyle = rgb(color);
    ctx.beginPath();
    ctx.ellipse(x, y, w / 2, h / 2, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  private strokeEllipse(x: number, y: number, w: number, h: number, color: Rgb, width: number): void {
    const { ctx } = this;
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.ellipse(x, y, w / 2, h / 2, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  private fillPolygon(points: Point[], color: Rgb): void {
    const { ctx } = this;
    ctx.fillStyle = rgb(color);
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.fill();
  }

  private strokePolygon(points: Point[], color: Rgb, width: number): void {
    const { ctx } = this;
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.stroke();
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
