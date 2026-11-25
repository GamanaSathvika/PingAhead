import Papa from "papaparse";
import {
  UI_STATES,
  type RawSimulationRow,
  type SimulationRow,
  type UiState,
  type Resolution,
  type Scenario,
} from "@/types/simulation";

/**
 * ASSUMPTION: resolution is encoded in the file name. Change this one
 * function if your backend exports differently (folder, query param, etc.).
 */
export function buildSimulationUrl(scenario: Scenario, resolution: Resolution) {
  return `/data/simulated/${resolution}/sim_${scenario}.csv`;
}

const toBool = (v: string | undefined) => Number(v) === 1;
const toUiState = (v: string): UiState =>
  (UI_STATES as readonly string[]).includes(v?.trim()) ? (v.trim() as UiState) : "NORMAL";

export function parseSimulationCsv(csvText: string): SimulationRow[] {
  const { data } = Papa.parse<RawSimulationRow>(csvText, {
    header: true,
    skipEmptyLines: true,
  });

  return data
    .map((r, i): SimulationRow => ({
      index: i,
      startTime: r.StartTime,
      // "2026-03-01 10:00:05" -> ISO-ish; parsed as local time, fine for relative playback
      timestampMs: Date.parse(r.StartTime.replace(" ", "T")),
      scenario: r.scenario,
      isAttack: toBool(r.is_attack),
      attackInNext60s: toBool(r.attack_in_next_60s),
      predProb: Number.parseFloat(r.pred_prob) || 0,
      forecastValid: toBool(r.forecast_valid),
      forecastAlert: toBool(r.forecast_alert),
      uiState: toUiState(r.ui_state),
    }))
    .sort((a, b) => a.timestampMs - b.timestampMs)
    .map((r, i) => ({ ...r, index: i })); // re-index after sort
}