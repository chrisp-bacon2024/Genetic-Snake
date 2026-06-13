import "./styles/main.css";
import { renderArchitecture } from "./sections/architecture";
import { DemoSection } from "./sections/demo";
import { renderResults } from "./sections/results";
import { renderTimeline } from "./sections/timeline";
import type { MetricRow, SiteManifest } from "./replay/types";

const REPO_URL = "https://github.com/chrisp-bacon2024/Genetic-Snake";

async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return (await response.json()) as T;
}

function renderShell(): void {
  const app = document.getElementById("app");
  if (!app) throw new Error("#app missing");

  app.innerHTML = `
    <header class="site-header">
      <div class="brand">Genetic Snake</div>
      <nav>
        <a href="#challenge">Challenge</a>
        <a href="#journey">Journey</a>
        <a href="#architecture">Architecture</a>
        <a href="#live">Live demo</a>
        <a href="#featured">Featured</a>
        <a href="#results">Results</a>
        <a href="${REPO_URL}" target="_blank" rel="noreferrer">GitHub</a>
      </nav>
    </header>

    <main>
      <section class="section hero">
        <p class="section-label">Portfolio demo</p>
        <h1>Snake brains evolved, not trained.</h1>
        <p class="hero-lead">
          A genetic algorithm evolves the weights of a small neural network that plays Snake —
          no backpropagation, no labeled dataset. Watch generations improve from random flailing
          to a full-board win on 20×20.
        </p>
        <div class="hero-actions">
          <a class="btn btn-primary" href="#live">Watch evolution</a>
          <a class="btn" href="${REPO_URL}" target="_blank" rel="noreferrer">View source</a>
        </div>
        <div class="stats-row">
          <div class="stat-card"><strong>44 → 64 → 4</strong><span>MLP topology</span></div>
          <div class="stat-card"><strong>~3,140</strong><span>evolved genes</span></div>
          <div class="stat-card"><strong>216</strong><span>generations logged</span></div>
          <div class="stat-card"><strong>399</strong><span>apples on 20×20 win</span></div>
        </div>
      </section>

      <section class="section" id="challenge">
        <p class="section-label">The challenge</p>
        <h2>Why this problem is hard</h2>
        <p>
          On a 20×20 board the snake must survive thousands of steps, avoid its growing body,
          and eventually fill every cell. Random policies die quickly. Hand-coded heuristics
          struggle as the body lengthens.
        </p>
        <p>
          I used a feedforward neural network to control the snake, with ray-based inputs. This
          is not a conventional supervised learning problem: there is no ground-truth
          &ldquo;correct move&rdquo; for each state, so the network cannot be trained with
          ordinary backpropagation. Instead, I evolved the weights with a genetic algorithm:
          each genome is a full set of network weights, fitness comes from game outcomes, and
          the population improves over generations.
        </p>
      </section>

      <section class="section" id="journey">
        <p class="section-label">Engineering journey</p>
        <h2>How the solution evolved</h2>
        <div id="timeline-root"></div>
      </section>

      <section class="section" id="architecture">
        <p class="section-label">System design</p>
        <h2>How it works</h2>
        <div id="architecture-root"></div>
      </section>

      <section class="section" id="live">
        <p class="section-label">Live demo</p>
        <h2>Six generations at once</h2>
        <p>
          Each tile replays the best snake saved for that generation on the grid size it was
          trained on. Use the controls below to pause or speed up all boards together. Click a
          tile to inspect it in the featured player.
        </p>
        <div class="demo-controls">
          <button class="btn" id="demo-play" type="button">Pause</button>
          <label>
            Speed
            <select class="speed-select" id="demo-speed">
              <option value="1">1×</option>
              <option value="3" selected>3×</option>
              <option value="10">10×</option>
            </select>
          </label>
        </div>
        <div class="grid-demo" id="grid-demo"></div>
      </section>

      <section class="section" id="featured">
        <p class="section-label">Featured run</p>
        <h2>Vision rays and network outputs</h2>
        <p id="featured-intro">
          The featured player shows heading-relative vision rays and the four direction logits
          chosen each tick. Full neural activations are included in the exported replay JSON.
        </p>
        <div class="featured-demo">
          <div class="featured-board-wrap">
            <div class="featured-meta" id="featured-meta"></div>
            <canvas id="featured-board"></canvas>
          </div>
          <div class="featured-side output-panel" id="output-panel"></div>
        </div>
      </section>

      <section class="section" id="results">
        <p class="section-label">Results</p>
        <h2>Training progress</h2>
        <div id="results-root"></div>
      </section>
    </main>

    <footer class="site-footer">
      <p><strong>Genetic Snake</strong> — Python, NumPy, Pygame, Matplotlib</p>
      <p>
        <a href="${REPO_URL}" target="_blank" rel="noreferrer">github.com/chrisp-bacon2024/Genetic-Snake</a>
      </p>
    </footer>
  `;
}

async function bootstrap(): Promise<void> {
  renderShell();
  renderTimeline(document.getElementById("timeline-root")!);
  renderArchitecture(document.getElementById("architecture-root")!);

  const [manifest, metrics] = await Promise.all([
    loadJson<SiteManifest>("data/manifest.json"),
    loadJson<MetricRow[]>("data/metrics.json"),
  ]);

  renderResults(document.getElementById("results-root")!, metrics);

  const demo = new DemoSection(
    manifest,
    document.getElementById("grid-demo")!,
    document.getElementById("featured-board") as HTMLCanvasElement,
    document.getElementById("featured-meta")!,
    document.getElementById("output-panel")!,
    {
      playButton: document.getElementById("demo-play") as HTMLButtonElement,
      speedSelect: document.getElementById("demo-speed") as HTMLSelectElement,
    },
  );
  await demo.init();
}

bootstrap().catch((error) => {
  console.error(error);
  const app = document.getElementById("app");
  if (app) {
    app.innerHTML = `<pre style="padding:2rem;color:#f88">Failed to load site: ${String(error)}</pre>`;
  }
});
