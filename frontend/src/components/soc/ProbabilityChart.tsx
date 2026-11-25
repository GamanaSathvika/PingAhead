"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import type { SimulationRow, UiState } from "@/types/simulation";
import { clock, stateRuns } from "@/lib/simulationUtils";

const HEIGHT = 320;
const MARGIN = { top: 12, right: 14, bottom: 26, left: 44 };
const LIVE_WINDOW_ROWS = 180; // 30 minutes at 10 s bins

type Mode = "live" | "full";

const BAND: Record<Exclude<UiState, "NORMAL">, { fill: string; opacity: number; label: string }> = {
  WARNING: { fill: "#fbbf24", opacity: 0.2, label: "Warning" },
  ATTACK: { fill: "#fb7185", opacity: 0.28, label: "Attack" },
  RECOVERING: { fill: "#60a5fa", opacity: 0.16, label: "Recovering" },
};

const DOT: Record<UiState, string> = {
  NORMAL: "#34d399",
  WARNING: "#fbbf24",
  ATTACK: "#fb7185",
  RECOVERING: "#60a5fa",
};

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export function ProbabilityChart({ rows, index }: { rows: SimulationRow[]; index: number }) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>("live");

  const last = rows.length - 1;
  const to = Math.min(index, last);
  const cur = rows[to] ?? null;

  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 1);
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

  // x-domain in row indexes: the whole run, or a sliding window ending at the playhead
  const d0 = mode === "full" ? 0 : Math.max(0, to - LIVE_WINDOW_ROWS + 1);
  const d1 = mode === "full" ? Math.max(last, d0 + 1) : d0 + LIVE_WINDOW_ROWS - 1;
  const span = Math.max(d1 - d0, 1);
  const step = innerW / span;
  const xOf = (i: number) => MARGIN.left + ((i - d0) / span) * innerW;
  const yOf = (p: number) => MARGIN.top + (1 - clamp01(p)) * innerH;

  // Approximate alert threshold: the lowest score that raised an alert in this file
  const threshold = useMemo(() => {
    let t = Infinity;
    for (const r of rows) if (r.forecastAlert && r.predProb < t) t = r.predProb;
    return Number.isFinite(t) ? t : null;
  }, [rows]);

  const drawn = useMemo(() => {
    const pts: string[] = [];
    const alerts: { cx: number; cy: number; i: number }[] = [];
    for (let i = d0; i <= to; i++) {
      const r = rows[i];
      const px = MARGIN.left + ((i - d0) / span) * innerW;
      const py = MARGIN.top + (1 - clamp01(r.predProb)) * innerH;
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
      if (r.forecastAlert) alerts.push({ cx: px, cy: py, i });
    }
    return { points: pts.join(" "), alerts, runs: stateRuns(rows, d0, to) };
  }, [rows, d0, to, span, innerW, innerH]);

  const xTicks = Array.from({ length: 5 }, (_, k) => Math.round(d0 + (span * k) / 4));
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  const modeBtn = (m: Mode, label: string) => (
    <button
      type="button"
      aria-pressed={mode === m}
      onClick={() => setMode(m)}
      className={
        "px-3 py-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 " +
        (mode === m ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700")
      }
    >
      {label}
    </button>
  );

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-semibold tracking-wide text-slate-200">Attack-risk score</h3>
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
          {(Object.keys(BAND) as (keyof typeof BAND)[]).map((k) => (
            <span key={k} className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: BAND[k].fill, opacity: 0.8 }}
              />
              {BAND[k].label}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 border-t border-dashed border-slate-400" />
            Alert threshold
          </span>
          <div className="flex overflow-hidden rounded-lg border border-slate-700" role="group" aria-label="Chart range">
            {modeBtn("live", "Live (30 min)")}
            {modeBtn("full", "Full run")}
          </div>
        </div>
      </div>

      <div ref={wrapRef} style={{ height: HEIGHT }} className="w-full">
        {rows.length === 0 ? (
          <div className="grid h-full place-items-center text-slate-500">Waiting for data…</div>
        ) : (
          width > 0 && (
            <svg
              width={width}
              height={HEIGHT}
              role="img"
              aria-label={
                cur
                  ? `Attack-risk score chart. Current score ${(cur.predProb * 100).toFixed(1)} percent, state ${cur.uiState}.`
                  : "Attack-risk score chart"
              }
            >
              {/* horizontal grid + y labels */}
              {yTicks.map((t) => (
                <g key={t}>
                  <line x1={MARGIN.left} x2={width - MARGIN.right} y1={yOf(t)} y2={yOf(t)} stroke="#1e293b" />
                  <text x={MARGIN.left - 8} y={yOf(t) + 4} textAnchor="end" fontSize="11" fill="#64748b">
                    {Math.round(t * 100)}%
                  </text>
                </g>
              ))}

              {/* state bands (min 3 px wide so single-window episodes stay visible) */}
              {drawn.runs.map((r) => {
                const x0 = Math.max(xOf(r.start) - step / 2, MARGIN.left);
                const x1 = Math.min(xOf(r.end) + step / 2, MARGIN.left + innerW);
                const b = BAND[r.state];
                return (
                  <rect
                    key={`${r.state}-${r.start}`}
                    x={x0}
                    y={MARGIN.top}
                    width={Math.max(x1 - x0, 3)}
                    height={innerH}
                    fill={b.fill}
                    opacity={b.opacity}
                  />
                );
              })}

              {/* alert threshold */}
              {threshold !== null && (
                <line
                  x1={MARGIN.left}
                  x2={width - MARGIN.right}
                  y1={yOf(threshold)}
                  y2={yOf(threshold)}
                  stroke="#94a3b8"
                  strokeDasharray="5 4"
                />
              )}

              {/* score line */}
              <polyline points={drawn.points} fill="none" stroke="#38bdf8" strokeWidth={1.75} strokeLinejoin="round" />

              {/* forecast alerts */}
              {drawn.alerts.map((a) => (
                <circle key={a.i} cx={a.cx} cy={a.cy} r={3} fill="#fbbf24" stroke="#0f172a" strokeWidth={1} />
              ))}

              {/* playhead */}
              {cur && (
                <g>
                  <line
                    x1={xOf(to)}
                    x2={xOf(to)}
                    y1={MARGIN.top}
                    y2={MARGIN.top + innerH}
                    stroke="#94a3b8"
                    strokeDasharray="2 3"
                  />
                  <circle cx={xOf(to)} cy={yOf(cur.predProb)} r={5} fill={DOT[cur.uiState]} stroke="#0f172a" strokeWidth={2} />
                </g>
              )}

              {/* x labels */}
              {xTicks.map((i) => {
                const r = rows[i];
                if (!r) return null;
                return (
                  <text key={i} x={xOf(i)} y={HEIGHT - 6} textAnchor="middle" fontSize="11" fill="#64748b">
                    {clock(r.startTime)}
                  </text>
                );
              })}
            </svg>
          )
        )}
      </div>
    </div>
  );
}
