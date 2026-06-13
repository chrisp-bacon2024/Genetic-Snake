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

export class DemoSection {
  private readonly gridRoot: HTMLElement;
  private readonly featuredMeta: HTMLElement;
  private readonly networkPanel: HTMLCanvasElement;
  private readonly featuredRenderer: BoardRenderer;
  private readonly networkVisualizer: NetworkVisualizer;
  private readonly featuredPlayer = new ReplayPlayer();
  private tiles: DemoTile[] = [];
  private playing = true;
  private speed = 1;
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
    },
  ) {
    this.gridRoot = gridRoot;
    this.featuredMeta = featuredMeta;
    this.networkPanel = networkPanel;
    this.featuredRenderer = new BoardRenderer(featuredBoard, 20, 20, 380);
    this.networkVisualizer = new NetworkVisualizer(networkPanel);
    this.activeGeneration = manifest.default_featured_generation;

    this.featuredPlayer.setLoop(false);
    this.featuredPlayer.onFrame((frame) => {
      this.featuredRenderer.draw(frame, { showRays: true });
      this.networkVisualizer.draw(frame);
    });
    this.featuredPlayer.onComplete(() => this.scheduleFeaturedRestart());

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
    this.networkVisualizer.resize(this.networkPanel.clientWidth || 280);
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
      void this.selectFeatured(entry.generation);
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
      if (!this.playing) {
        return;
      }
      tile.player.restart();
    }, END_HOLD_MS);
  }

  private scheduleFeaturedRestart(): void {
    window.clearTimeout(this.featuredRestartTimeout);
    this.featuredRestartTimeout = window.setTimeout(() => {
      if (!this.playing) {
        return;
      }
      this.featuredPlayer.restart();
    }, END_HOLD_MS);
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
    const entry = this.findFeaturedEntry(generation);
    if (!entry) return;

    window.clearTimeout(this.featuredRestartTimeout);
    this.featuredRenderer.resize(entry.grid_cols, entry.grid_rows, 380);
    this.networkVisualizer.resize(this.networkPanel.clientWidth || 280);

    const durationSec = Math.ceil(entry.frame_count / entry.ticks_per_second);
    this.featuredMeta.textContent = `Generation ${entry.generation} · score ${entry.score} · ${entry.grid_cols}×${entry.grid_rows} · ${entry.frame_count} frames (~${durationSec}s at 1×) · died: ${entry.death_cause}`;

    this.featuredPlayer.stopLoop();
    await this.featuredPlayer.load(this.featuredReplayPath(entry));
    this.featuredPlayer.setSpeed(this.speed);
    this.featuredPlayer.setPlaying(this.playing);
    this.featuredPlayer.startLoop();
  }

  private syncPlayback(): void {
    if (this.playing) {
      for (const tile of this.tiles) {
        if (tile.player.isFinished) {
          tile.player.restart();
        } else {
          tile.player.setPlaying(true);
        }
        window.clearTimeout(tile.restartTimeout);
      }
      window.clearTimeout(this.featuredRestartTimeout);
      if (this.featuredPlayer.isFinished) {
        this.featuredPlayer.restart();
      } else {
        this.featuredPlayer.setPlaying(true);
      }
    } else {
      for (const tile of this.tiles) {
        tile.player.setPlaying(false);
        window.clearTimeout(tile.restartTimeout);
      }
      window.clearTimeout(this.featuredRestartTimeout);
      this.featuredPlayer.setPlaying(false);
    }
  }

  private syncSpeed(): void {
    for (const tile of this.tiles) {
      tile.player.setSpeed(this.speed);
    }
    this.featuredPlayer.setSpeed(this.speed);
  }
}
