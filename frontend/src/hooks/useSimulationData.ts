"use client";
import { useEffect, useState } from "react";
import {
  UI_STATES,
  type Resolution,
  type Scenario,
  type SimulationRow,
  type UiState,
} from "@/types/simulation";

/** One shared empty array: a fresh [] each render would make useSimulationPlayer rewind in a loop. */
const EMPTY: SimulationRow[] = [];

/** Successfully parsed files, kept for the session so switching back is instant. */
const cache = new Map<string, SimulationRow[]>();

/** The only columns the dashboard needs (the CSV has ~135; the rest are ignored). */
const REQUIRED_COLUMNS = [
  "StartTime",
  "scenario",
  "is_attack",
  "attack_in_next_60s",
  "pred_prob",
  "forecast_valid",
  "forecast_alert",
  "ui_state",
] as const;

const toBool = (v: string) => v === "1" || v.toLowerCase() === "true";

/**
 * Minimal CSV parser for the files written by train_multires.py. Those files hold only
 * numbers, ISO-style timestamps and plain words, so no field contains a quoted comma.
 */
export function parseSimulationCsv(text: string): SimulationRow[] {
  const lines = text.split(/\r?\n/);
  const header = (lines[0] ?? "").split(",").map((h) => h.trim());

  const col: Record<string, number> = {};
  for (const name of REQUIRED_COLUMNS) col[name] = header.indexOf(name);
  const missing = REQUIRED_COLUMNS.filter((name) => col[name] === -1);
  if (missing.length > 0) {
    throw new Error(`CSV is missing columns: ${missing.join(", ")}`);
  }

  const rows: SimulationRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue; // blank / trailing line
    const c = line.split(",");

    const rawState = c[col.ui_state].trim();
    if (!(UI_STATES as readonly string[]).includes(rawState)) {
      throw new Error(`Row ${i}: unknown ui_state "${rawState}"`);
    }

    const startTime = c[col.StartTime].trim();
    // "2026-01-01 09:01:00" -> ISO form; Safari cannot parse the space-separated version
    const timestampMs = Date.parse(startTime.replace(" ", "T"));
    const prob = Number(c[col.pred_prob]);

    rows.push({
      index: rows.length,
      startTime,
      timestampMs,
      scenario: c[col.scenario].trim(),
      isAttack: toBool(c[col.is_attack]),
      attackInNext60s: toBool(c[col.attack_in_next_60s]),
      predProb: Number.isFinite(prob) ? prob : 0,
      forecastValid: toBool(c[col.forecast_valid]),
      forecastAlert: toBool(c[col.forecast_alert]),
      uiState: rawState as UiState,
    });
  }
  if (rows.length === 0) throw new Error("CSV has no data rows");
  return rows;
}

export function useSimulationData(scenario: Scenario, resolution: Resolution) {
  const key = `${resolution}/${scenario}`;
  const url = `/data/simulated/${resolution}/sim_${scenario}.csv`;

  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const [, bump] = useState(0); // re-render once a fetch finishes and fills the cache

  const rows = cache.get(key);

  useEffect(() => {
    if (cache.has(key)) return;
    const controller = new AbortController();

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
        return res.text();
      })
      .then((text) => {
        cache.set(key, parseSimulationCsv(text));
        bump((n) => n + 1);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return; // switched away
        setFailure({ key, message: err instanceof Error ? err.message : String(err) });
      });

    return () => controller.abort();
  }, [key, url]);

  const failed = !rows && failure?.key === key;
  return {
    rows: rows ?? EMPTY,
    loading: !rows && !failed,
    error: failed ? failure.message : null,
  };
}