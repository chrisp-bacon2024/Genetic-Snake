export interface TimelineItem {
  step: string;
  title: string;
  body: string;
  stat?: string;
}

export const timelineItems: TimelineItem[] = [
  {
    step: "01",
    title: "Built the Snake game",
    body: "Started with pygame rules: movement, food, collisions, starvation, and win detection on configurable grids.",
  },
  {
    step: "02",
    title: "Neural network control",
    body: "Replaced keyboard input with a small feedforward network that picks UP/DOWN/LEFT/RIGHT each tick.",
  },
  {
    step: "03",
    title: "Ray-based inputs (first pass)",
    body:
      "Gave the network heading-relative vision rays — wall, food, and body proximity in eight directions — instead of raw coordinates. The snake could sense its surroundings without reading the whole board.",
  },
  {
    step: "04",
    title: "First genetic algorithm",
    body:
      "With that ray-fed network defined, evolved its weights with tournament selection and mutation. Early runs topped out around score 6.",
    stat: "Score 6 @ 200 gens",
  },
  {
    step: "05",
    title: "Fitness and GA tuning",
    body: "Rewarded eating and pathing, penalized idle wandering. Population quality improved steadily.",
    stat: "Score 9 @ gen 131",
  },
  {
    step: "06",
    title: "Rays plus memory",
    body:
      "Added a recurrent layer (LSTM/GRU) on top of the ray encoder so the policy could carry context across ticks. Scores improved, but each evaluation required a sequential forward pass per tick, which made generation times impractical at population scale.",
    stat: "Max score ~43",
  },
  {
    step: "07",
    title: "Full-board input detour",
    body:
      "Temporarily switched to one input neuron per grid cell to simplify the problem while searching for a fitness function that actually rewarded winning behavior. Trained on a fixed 5×5 grid, and that architecture eventually cleared the board once the reward shaping clicked.",
    stat: "5×5 win",
  },
  {
    step: "08",
    title: "Back to rays — refined, no memory",
    body:
      "Returned to ray vision without recurrence, but with a richer feature set than the first version: explicit food bearing, tail direction, safe-move lookahead, and reachable-space metrics (44 inputs total). Grid-size agnostic — the same encoder scales from 5×5 through 20×20.",
  },
  {
    step: "09",
    title: "Scale with multiprocessing",
    body: "Headless parallel evaluation evaluates 1000 genomes per generation across multiple CPU workers.",
  },
  {
    step: "10",
    title: "Training observability",
    body: "Added live dashboard, JSONL metrics, and per-generation checkpoints for resume and analysis.",
  },
  {
    step: "11",
    title: "Curriculum learning",
    body:
      "Added automatic stage progression for the refined ray encoder: start on 5×5, advance to 10×10 and then 20×20 once enough of the population wins the current board. Unlike the earlier fixed-grid experiments, one training run now walks up in difficulty.",
  },
  {
    step: "12",
    title: "20×20 full-board win",
    body: "The evolved ray policy eventually fills the entire 20×20 board — 399 apples, no backpropagation.",
    stat: "Score 399 @ gen 215",
  },
];

/** On phones only the first steps show until "Show all" is pressed. Keep in
 *  sync with the :nth-child(n+5) rule in styles/layout.css. */
const PHONE_VISIBLE_STEPS = 4;

export function renderTimeline(container: HTMLElement): void {
  const items = timelineItems
    .map(
      (item) => `
        <article class="timeline-item">
          <div class="timeline-step"><span class="timeline-step-word">Step </span>${item.step}</div>
          <div>
            <h3>${item.title}</h3>
            <p>${item.body}${item.stat ? ` <span class="timeline-stat">(${item.stat})</span>` : ""}</p>
          </div>
        </article>
      `,
    )
    .join("");

  const hiddenOnPhones = timelineItems.length - PHONE_VISIBLE_STEPS;
  container.innerHTML = `
    <div class="timeline" id="timeline-list">${items}</div>
    <button class="btn timeline-toggle" type="button" aria-expanded="false" aria-controls="timeline-list">
      Show all ${timelineItems.length} steps
    </button>
  `;

  const list = container.querySelector<HTMLElement>(".timeline")!;
  const toggle = container.querySelector<HTMLButtonElement>(".timeline-toggle")!;
  toggle.hidden = hiddenOnPhones <= 0;
  toggle.addEventListener("click", () => {
    const expanded = list.classList.toggle("expanded");
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.textContent = expanded ? "Show fewer steps" : `Show all ${timelineItems.length} steps`;
    if (!expanded) list.scrollIntoView({ block: "start" });
  });
}
