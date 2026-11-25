export const UI_STATES = ["NORMAL", "WARNING", "ATTACK", "RECOVERING"] as const;
export type UiState = (typeof UI_STATES)[number];

export const RESOLUTIONS = ["5s", "10s", "30s", "60s"] as const;
export type Resolution = (typeof RESOLUTIONS)[number];

export const SCENARIOS = ["scenario-5-disrupt", "scenario-8-authtest"] as const;
export type Scenario = (typeof SCENARIOS)[number];

export const PLAYBACK_SPEEDS = [1, 2, 5] as const;
export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number];

/** Exactly what PapaParse gives us: every CSV cell is a string. */
export interface RawSimulationRow {
  StartTime: string;
  scenario: string;
  is_attack: string;
  attack_in_next_60s: string;
  pred_prob: string;
  forecast_valid: string;
  forecast_alert: string;
  ui_state: string;
}

/** Parsed, typed row used everywhere in the UI. */
export interface SimulationRow {
  index: number;          // position in the file; handy as a stable key
  startTime: string;      // original string, for display
  timestampMs: number;    // for chart x-axis / sorting
  scenario: string;
  isAttack: boolean;
  attackInNext60s: boolean;
  predProb: number;
  forecastValid: boolean;
  forecastAlert: boolean;
  uiState: UiState;
}

export type AlertEventType = Exclude<UiState, "NORMAL">;

/** A state transition worth logging in the ticker. */
export interface AlertEvent {
  id: string;
  rowIndex: number;
  startTime: string;
  type: AlertEventType;
  from: UiState;
  predProb: number;
}