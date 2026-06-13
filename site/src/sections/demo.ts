import { BoardRenderer } from "../replay/BoardRenderer";
import { NetworkVisualizer } from "../replay/NetworkVisualizer";
import { ReplayPlayer } from "../replay/ReplayPlayer";
import type { GenerationEntry, SiteManifest } from "../replay/types";

interface DemoTile {
  entry: GenerationEntry;
  canvas: HTMLCanvasElement;
  renderer: BoardRenderer;
  player: ReplayPlayer;
  root: HTMLElement;
  restartTimeout: number;
}

const END_HOLD_MS = 2500;
const GRID_TILE_SPEED = 3;

export class DemoSection {
  private readonly gridRoot: HTMLElement;
  private readonly featuredMeta: HTMLElement;
  private readonly networkPanel: HTMLCanvasElement;
  private readonly featuredRenderer: BoardRenderer;
  private readonly networkVisualizer: NetworkVisualizer;
  private readonly featuredPlayer = new ReplayPlayer();
  private readonly generationSelect: HTMLSelectElement;
  private tiles: DemoTile[] = [];
  private featuredPlaying = true;
  private featuredSpeed = 3;
  private activeGeneration = 215;
  private featuredRestartTimeout = 0;

  constructor(
    private readonly manifest: SiteManifest,
    gridRoot: HTMLElement,
    featuredBoard: HTMLCanvasElement,
    featuredMeta: HTMLElement,
    networkPanel: HTMLCanvasElement,
    controls: {
      playButton: HTMLButtonElement;
      speedSelect: HTMLSelectElement;
      generationSelect: HTMLSelectElement;
    },
  ) {
    this.gridRoot = gridRoot;
    this.featuredMeta = featuredMeta;
    this.networkPanel = networkPanel;
    this.generationSelect = controls.generationSelect;
    this.featuredRenderer = new BoardRenderer(featuredBoard, 20, 20, 380);
    this.networkVisualizer = new NetworkVisualizer(networkPanel);
    this.activeGeneration = manifest.default_featured_generation;

    this.populateGenerationSelect();

    this.featuredPlayer.setLoop(false);
    this.featuredPlayer.onFrame((frame) => {
      this.featuredRenderer.draw(frame, { showRays: this.featuredSpeed < 10 });
      this.networkVisualizer.draw(frame);
    });
    this.featuredPlayer.onComplete(() => this.scheduleFeaturedRestart());

    controls.playButton.addEventListener("click", () => {
      this.featuredPlaying = !this.featuredPlaying;
      controls.playButton.textContent = this.featuredPlaying ? "Pause" : "Play";
      this.syncFeaturedPlayback();
    });

    controls.speedSelect.addEventListener("change", () => {
      this.featuredSpeed = Number(controls.speedSelect.value);
      this.featuredPlayer.setSpeed(this.featuredSpeed);
    });

    this.generationSelect.addEventListener("change", () => {
      void this.selectFeatured(Number(this.generationSelect.value));
    });
  }

  private populateGenerationSelect(): void {
    const entries = [...this.manifest.featured_generations].sort(
      (a, b) => a.generation - b.generation,
    );

    this.generationSelect.replaceChildren(
      ...entries.map((entry) => {
        const option = document.createElement("option");
        option.value = String(entry.generation);
        option.textContent = `Gen ${entry.generation} · score ${entry.score} · ${entry.grid_cols}×${entry.grid_rows}`;
        option.selected = entry.generation === this.activeGeneration;
        return option;
      }),
    );
  }

  async init(): Promise<void> {
    this.networkVisualizer.resize(this.networkPanel.clientWidth || 280);
    await Promise.all(this.manifest.grid_generations.map((entry) => this.createTile(entry)));
    await this.loadFeatured(this.activeGeneration);
    this.featuredPlayer.setSpeed(this.featuredSpeed);
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
    player.setSpeed(GRID_TILE_SPEED);
    player.onFrame((frame) => renderer.draw(frame));

    const tile: DemoTile = {
      entry,
      canvas,
      renderer,
      player,
      root,
      restartTimeout: 0,
    };
    player.onComplete(() => this.scheduleTileRestart(tile));

    root.addEventListener("click", () => {
      void this.selectFeatured(entry.generation, { scroll: true });
    });

    this.gridRoot.appendChild(root);
    this.tiles.push(tile);

    await player.load(entry.path);
    player.startLoop();
  }

  private findFeaturedEntry(generation: number): GenerationEntry | undefined {
    return (
      this.manifest.featured_generations.find((item) => item.generation === generation) ??
      this.manifest.grid_generations.find((item) => item.generation === generation)
    );
  }

  private featuredReplayPath(entry: GenerationEntry): string {
    if (!entry.lite) {
      return entry.path;
    }
    return entry.path.replace("_lite.json", "_full.json");
  }

  private scheduleTileRestart(tile: DemoTile): void {
    window.clearTimeout(tile.restartTimeout);
    tile.restartTimeout = window.setTimeout(() => {
      tile.player.restart();
    }, END_HOLD_MS);
  }

  private scheduleFeaturedRestart(): void {
    window.clearTimeout(this.featuredRestartTimeout);
    this.featuredRestartTimeout = window.setTimeout(() => {
      if (!this.featuredPlaying) {
        return;
      }
      this.featuredPlayer.restart();
    }, END_HOLD_MS);
  }

  private async selectFeatured(
    generation: number,
    options: { scroll?: boolean } = {},
  ): Promise<void> {
    this.activeGeneration = generation;
    this.generationSelect.value = String(generation);

    for (const tile of this.tiles) {
      tile.root.classList.toggle("active", tile.entry.generation === generation);
    }

    await this.loadFeatured(generation);

    if (options.scroll) {
      document.getElementById("featured")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  private async loadFeatured(generation: number): Promise<void> {
    const entry = this.findFeaturedEntry(generation);
    if (!entry) return;

    window.clearTimeout(this.featuredRestartTimeout);
    this.featuredRenderer.resize(entry.grid_cols, entry.grid_rows, 380);
    this.networkVisualizer.resize(this.networkPanel.clientWidth || 280);

    const durationSec = Math.ceil(entry.frame_count / entry.ticks_per_second);
    this.featuredMeta.textContent = `Generation ${entry.generation} · score ${entry.score} · ${entry.grid_cols}×${entry.grid_rows} · ${entry.frame_count} frames (~${durationSec}s at 1×) · died: ${entry.death_cause}`;

    this.featuredPlayer.stopLoop();
    await this.featuredPlayer.load(this.featuredReplayPath(entry));
    this.featuredPlayer.setSpeed(this.featuredSpeed);
    this.featuredPlayer.setPlaying(this.featuredPlaying);
    this.featuredPlayer.startLoop();
  }

  private syncFeaturedPlayback(): void {
    if (this.featuredPlaying) {
      window.clearTimeout(this.featuredRestartTimeout);
      if (this.featuredPlayer.isFinished) {
        this.featuredPlayer.restart();
      } else {
        this.featuredPlayer.setPlaying(true);
      }
    } else {
      window.clearTimeout(this.featuredRestartTimeout);
      this.featuredPlayer.setPlaying(false);
    }
  }
}
