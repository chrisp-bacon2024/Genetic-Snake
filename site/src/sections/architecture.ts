export function renderArchitecture(container: HTMLElement): void {
  container.innerHTML = `
    <div class="arch-grid">
      <article class="arch-card">
        <h3>State encoder (44 inputs)</h3>
        <p>8 vision rays × wall/food/body proximity, food bearing, head/tail direction, safe lookahead moves, and reachable-space metrics.</p>
      </article>
      <article class="arch-card">
        <h3>Policy network</h3>
        <p>MLP with one hidden layer: 44 → 64 (ReLU) → 4 logits. Roughly 3,140 evolvable weights. Illegal moves are masked before argmax.</p>
      </article>
      <article class="arch-card">
        <h3>Genetic algorithm</h3>
        <p>Population 1000, elitism, tournament selection, SBX crossover, mixed-scale mutation, hall-of-fame champion cloning.</p>
      </article>
      <article class="arch-card">
        <h3>Curriculum</h3>
        <p>Master smaller boards first (5×5, then 10×10) before training on the 20×20 win condition.</p>
      </article>
    </div>
    <pre class="flow-diagram">GameStateEncoder ──► NeuralNetwork ──► masked argmax ──► Game.tick()
        ▲                                              │
        └──────────── snake + grid state ──────────────┘

Each generation:
  evaluate population (parallel headless sim)
  rank by fitness → save best genome
  evolve next generation → checkpoint</pre>
  `;
}
