import type { MetricRow } from "../replay/types";

export function renderResults(container: HTMLElement, metrics: MetricRow[]): void {
  const last = metrics[metrics.length - 1];
  const peak = metrics.reduce((best, row) => Math.max(best, row.best_score), 0);
  const wins = metrics.filter((row) => row.death_cause === "win").length;
  const firstWin = metrics.find((row) => row.death_cause === "win");

  const milestones = [
    { label: "Generations trained", value: String(metrics.length) },
    { label: "Peak best score", value: String(peak) },
    { label: "Final grid", value: `${last.grid_cols}×${last.grid_rows}` },
    { label: "Winning runs logged", value: String(wins) },
  ];

  const highlightRows = [0, 10, 50, 100, 150, 215]
    .map((generation) => metrics.find((row) => row.generation === generation))
    .filter((row): row is MetricRow => row !== undefined);

  container.innerHTML = `
    <div class="stats-row">
      ${milestones
        .map(
          (item) => `
            <div class="stat-card">
              <strong>${item.value}</strong>
              <span>${item.label}</span>
            </div>
          `,
        )
        .join("")}
    </div>
    <img class="results-chart" src="data/training_chart.png" alt="Training progress chart showing best score by generation" />
    <div class="results-table-wrap">
      <table class="results-table">
        <thead>
          <tr>
            <th>Generation</th>
            <th>Grid</th>
            <th>Best score</th>
            <th>Best ever</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          ${highlightRows
            .map(
              (row) => `
                <tr>
                  <td>${row.generation}</td>
                  <td>${row.grid_cols}×${row.grid_rows}</td>
                  <td>${row.best_score}</td>
                  <td>${row.best_ever_score}</td>
                  <td>${row.death_cause}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
    ${
      firstWin
        ? `<p style="margin-top:1rem">First logged win at generation ${firstWin.generation} on a ${firstWin.grid_cols}×${firstWin.grid_rows} board.</p>`
        : ""
    }
  `;
}
