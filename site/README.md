# Genetic Snake — Demo Site

Static portfolio site for the Genetic Snake project. Replays exported from local training artifacts are played back in the browser with Canvas.

## Prerequisites

- Node.js 18+
- Python environment with project dependencies (for re-exporting replay data)
- Local `src/replays/` folder with `gen_*.npz` files (not committed to git)

## Export replay data

From the repository root:

```bash
python scripts/export_site_data.py --replays-dir src/replays
```

This writes JSON replays, `manifest.json`, `metrics.json`, and `training_chart.png` into `site/public/data/`.

Options:

```bash
python scripts/export_site_data.py --replays-dir src/replays --generations 0,10,50 --lite
python scripts/export_site_data.py --replays-dir src/replays --featured-generations 0,215 --full
python scripts/export_site_data.py --metrics-only --replays-dir src/replays
```

## Live site

**https://chrisp-bacon2024.github.io/Genetic-Snake/**

Pushes to `main` that touch `site/` trigger [GitHub Actions](../.github/workflows/deploy-site.yml) to rebuild and publish.

## Run locally

```bash
cd site
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

## Production build

```bash
cd site
npm run build
npm run preview
```

Built files land in `site/dist/` with relative asset paths (`base: './'`) for future GitHub Pages deployment.
