import type { AlertEvent, SimulationRow, UiState } from "@/types/simulation";

export interface StateRun {
  state: Exclude<UiState, "NORMAL">;
  start: number; // first row index (inclusive)
  end: number; // last row index (inclusive)
}

/** "2026-11-11 17:36:30" -> ms. Safari cannot parse the space-separated form. */
export function parseMs(t: string): number {
  return Date.parse(t.replace(" ", "T"));
}

/** "2026-11-11 17:36:30" -> "17:36" (or "17:36:30" with seconds). */
export function clock(t: string, withSeconds = false): string {
  return t.length >= 19 ? t.slice(11, withSeconds ? 19 : 16) : t;
}

/** Contiguous stretches of WARNING / ATTACK / RECOVERING between two row indexes (inclusive). */
export function stateRuns(rows: SimulationRow[], from: number, to: number): StateRun[] {
  const runs: StateRun[] = [];
  let cur: StateRun | null = null;
  const stop = Math.min(to, rows.length - 1);
  for (let i = Math.max(from, 0); i <= stop; i++) {
    const s = rows[i].uiState;
    if (s === "NORMAL") {
      cur = null;
    } else if (cur && cur.state === s) {
      cur.end = i;
    } else {
      cur = { state: s, start: i, end: i };
      runs.push(cur);
    }
  }
  return runs;
}

/**
 * How many seconds before attack event `i` the first forecast alert was raised.
 * Counts a warning if it started within 60 s of the attack, or if it was still active when the
 * attack began. Returns null when there was no advance warning. `events` is chronological.
 * The value is a lower bound: attack onset is the start of the first attack window.
 */
export function warningLead(events: AlertEvent[], i: number): number | null {
  const atk = parseMs(events[i].startTime);
  let best: number | null = null;
  let latest = true;
  for (let j = i - 1; j >= 0 && events[j].type !== "ATTACK"; j--) {
    if (events[j].type !== "WARNING") continue;
    const lead = (atk - parseMs(events[j].startTime)) / 1000;
    const activeAtOnset = latest && events[i].from === "WARNING";
    latest = false;
    if (lead <= 60 || activeAtOnset) best = lead;
    else break;
  }
  return best;
}
