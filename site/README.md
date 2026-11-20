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

Open `http://localhost:8080` (set in `vite.config.ts`).

## Production build

```bash
cd site
npm run build
npm run preview
```

Built files land in `site/dist/` with relative asset paths (`base: './'`) for future GitHub Pages deployment.

## Styling

The site uses the shared [christophermpalmer.com](https://christophermpalmer.com/) theme, so it matches the portfolio and the Pathfinding Visualizer:

- `index.html` links `https://christophermpalmer.com/shared/theme/v1/theme.css` before the site's own CSS. It provides the brand tokens (colours, serif type, spacing) and shared components (`.btn`, `.btn-primary`, `.page-title`, `.section-title`, `.select`). The page needs an internet connection to look right, including with `npm run dev`.
- `src/styles/*.css` hold only this page's layout, written with the theme's variables (`var(--color-navy)`, `var(--space-4)`…) rather than raw colours.
- The canvases read the same variables at runtime (`src/styles/theme.ts`), so the board and network panel follow theme changes too. The snake, apple and vision-ray colours are this project's own and stay in `theme.ts`.
- **Font exception (brand guidelines §7.8):** the network panel's labels of 11px and below use the system sans-serif, because serif text that small is hard to read in a dense data display. Everything else is serif, and the flow diagram is monospace because it's a code-style diagram.

The pygame app (`src/`) keeps its original dark colours from `src/config.py`.
