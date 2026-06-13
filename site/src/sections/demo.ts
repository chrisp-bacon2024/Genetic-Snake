import { BoardRenderer } from "../replay/BoardRenderer";
import { ReplayPlayer } from "../replay/ReplayPlayer";
import type { GenerationEntry, LiteReplayFrame, SiteManifest } from "../replay/types";
import { OUTPUT_LABELS } from "../replay/types";

interface DemoTile {
  entry: GenerationEntry;
  canvas: HTMLCanvasElement;
  renderer: BoardRenderer;
  player: ReplayPlayer;
  root: HTMLElement;
}

const END_HOLD_MS = 2500;

export class DemoSection {
  private readonly gridRoot: HTMLElement;
  private readonly featuredBoard: HTMLCanvasElement;
  private readonly featuredMeta: HTMLElement;
  private readonly outputPanel: HTMLElement;
  private readonly featuredRenderer: BoardRenderer;
  private readonly featuredPlayer = new ReplayPlayer();
  private tiles: DemoTile[] = [];
  private playing = true;
  private speed = 1;
  private activeGeneration = 215;
  private finishedCount = 0;
  private restartTimeout = 0;

  constructor(
    private readonly manifest: SiteManifest,
    gridRoot: HTMLElement,
    featuredBoard: HTMLCanvasElement,
    featuredMeta: HTMLElement,
    outputPanel: HTMLElement,
    controls: {
      playButton: HTMLButtonElement;
      speedSelect: HTMLSelectElement;
    },
  ) {
    this.gridRoot = gridRoot;
    this.featuredBoard = featuredBoard;
    this.featuredMeta = featuredMeta;
    this.outputPanel = outputPanel;
    this.featuredRenderer = new BoardRenderer(featuredBoard, 20, 20);
    this.activeGeneration = manifest.default_featured_generation;

    this.featuredPlayer.setLoop(true);
    this.featuredPlayer.onFrame((frame) => {
      this.featuredRenderer.draw(frame, { showRays: true });
      this.renderOutputs(frame);
    });

    controls.playButton.addEventListener("click", () => {
      this.playing = !this.playing;
      controls.playButton.textContent = this.playing ? "Pause" : "Play";
      this.syncPlayback();
    });

    controls.speedSelect.addEventListener("change", () => {
      this.speed = Number(controls.speedSelect.value);
      this.syncSpeed();
    });
  }

  async init(): Promise<void> {
    await Promise.all(this.manifest.grid_generations.map((entry) => this.createTile(entry)));
    this.syncSpeed();
    this.syncPlayback();
    await this.loadFeatured(this.activeGeneration);
    this.featuredPlayer.startLoop();
  }

  private async createTile(entry: GenerationEntry): Promise<void> {
    const root = document.createElement("button");
    root.type = "button";
    root.className = "demo-tile";
    root.setAttribute("aria-label", `View generation ${entry.generation} in featured player`);
    root.innerHTML = `
      <div class="demo-tile-meta">
        <strong>Gen ${entry.generation}</strong>
        Score ${entry.score} · ${entry.grid_cols}×${entry.grid_rows}
        <span>${entry.narrative}</span>
      </div>
    `;

    const canvas = document.createElement("canvas");
    root.appendChild(canvas);

    const renderer = new BoardRenderer(canvas, entry.grid_cols, entry.grid_rows);
    const player = new ReplayPlayer();
    player.setLoop(false);
    await player.load(entry.path);
    player.onFrame((frame) => renderer.draw(frame));
    player.onComplete(() => this.handleTileComplete());

    root.addEventListener("click", () => {
      void this.selectFeatured(entry.generation);
    });

    this.gridRoot.appendChild(root);
    this.tiles.push({ entry, canvas, renderer, player, root });
    player.startLoop();
  }

  private handleTileComplete(): void {
    this.finishedCount += 1;
    if (this.finishedCount < this.tiles.length) {
      return;
    }
    window.clearTimeout(this.restartTimeout);
    this.restartTimeout = window.setTimeout(() => {
      if (!this.playing) {
        return;
      }
      this.restartGrid();
    }, END_HOLD_MS);
  }

  private restartGrid(): void {
    this.finishedCount = 0;
    window.clearTimeout(this.restartTimeout);
    for (const tile of this.tiles) {
      tile.player.restart();
    }
  }

  private async selectFeatured(generation: number): Promise<void> {
    this.activeGeneration = generation;
    for (const tile of this.tiles) {
      tile.root.classList.toggle("active", tile.entry.generation === generation);
    }
    await this.loadFeatured(generation);
    document.getElementById("featured")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  private async loadFeatured(generation: number): Promise<void> {
    const entry =
      this.manifest.featured_generations.find((item) => item.generation === generation) ??
      this.manifest.featured_generations.find(
        (item) => item.generation === this.manifest.default_featured_generation,
      );
    if (!entry) return;

    this.featuredRenderer.resize(entry.grid_cols, entry.grid_rows);
    this.featuredBoard.width = Math.max(this.featuredBoard.width, entry.grid_cols * 18 + 16);
    this.featuredBoard.height = Math.max(this.featuredBoard.height, entry.grid_rows * 18 + 32);
    this.featuredRenderer.resize(entry.grid_cols, entry.grid_rows);

    this.featuredMeta.textContent = `Generation ${entry.generation} · score ${entry.score} · ${entry.grid_cols}×${entry.grid_rows} · died: ${entry.death_cause}`;

    this.featuredPlayer.stopLoop();
    await this.featuredPlayer.load(entry.path);
    this.featuredPlayer.setSpeed(this.speed);
    this.featuredPlayer.setPlaying(this.playing);
    this.featuredPlayer.startLoop();
  }

  private renderOutputs(frame: LiteReplayFrame): void {
    const outputs = frame.outputs ?? [0, 0, 0, 0];
    const max = Math.max(...outputs, 0.001);
    const chosen = frame.direction;

    this.outputPanel.innerHTML = `
      <h4>Network outputs</h4>
      <div class="output-bars">
        ${OUTPUT_LABELS.map((label, index) => {
          const active = label === chosen;
          const width = Math.max(4, (outputs[index] / max) * 100);
          return `
            <div class="output-row${active ? " active" : ""}">
              <span>${label}</span>
              <div class="output-bar-track">
                <div class="output-bar-fill" style="width:${width}%"></div>
              </div>
              <span>${outputs[index].toFixed(2)}</span>
            </div>
          `;
        }).join("")}
      </div>
      <p style="margin-top:0.75rem;color:rgb(160,160,180);font-size:0.85rem">
        Chosen move: <span class="mono">${chosen}</span>
      </p>
    `;
  }

  private syncPlayback(): void {
    if (this.playing) {
      const allFinished = this.tiles.length > 0 && this.tiles.every((tile) => tile.player.isFinished);
      if (allFinished) {
        this.restartGrid();
      } else {
        for (const tile of this.tiles) {
          tile.player.setPlaying(true);
        }
      }
      window.clearTimeout(this.restartTimeout);
    } else {
      for (const tile of this.tiles) {
        tile.player.setPlaying(false);
      }
    }
    this.featuredPlayer.setPlaying(this.playing);
  }

  private syncSpeed(): void {
    for (const tile of this.tiles) {
      tile.player.setSpeed(this.speed);
    }
    this.featuredPlayer.setSpeed(this.speed);
  }
}
