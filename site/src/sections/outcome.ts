/** Plain-English labels for how a run ended (death_cause in the data). */
const OUTCOMES: Record<string, string> = {
  win: "cleared the board",
  wall: "hit a wall",
  body: "hit itself",
  starved: "ran out of moves",
};

export function outcomeLabel(cause: string): string {
  return OUTCOMES[cause] ?? cause;
}
