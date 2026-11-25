import { ShieldCheck, TriangleAlert, ShieldAlert, ShieldEllipsis } from "lucide-react";
import type { SimulationRow, UiState } from "@/types/simulation";

const STATE_STYLES: Record<
  UiState,
  { label: string; sub: string; wrap: string; dot: string; Icon: typeof ShieldCheck }
> = {
  NORMAL: {
    label: "NORMAL",
    sub: "Traffic within expected baseline",
    wrap: "bg-emerald-500/10 border-emerald-500/40 text-emerald-300",
    dot: "bg-emerald-400",
    Icon: ShieldCheck,
  },
  WARNING: {
    label: "WARNING",
    sub: "Pre-onset signal: attack forecast within 60s",
    wrap: "bg-amber-500/15 border-amber-500/50 text-amber-300",
    dot: "bg-amber-400",
    Icon: TriangleAlert,
  },
  ATTACK: {
    label: "ATTACK",
    sub: "Active intrusion detected",
    wrap: "bg-rose-500/15 border-rose-500/60 text-rose-300",
    dot: "bg-rose-400",
    Icon: ShieldAlert,
  },
  RECOVERING: {
    label: "RECOVERING",
    sub: "Attack subsiding, monitoring for recurrence",
    wrap: "bg-blue-500/10 border-blue-500/40 text-blue-300",
    dot: "bg-blue-400",
    Icon: ShieldEllipsis,
  },
};

export function StatusBanner({ row }: { row: SimulationRow | null }) {
  const state: UiState = row?.uiState ?? "NORMAL";
  const s = STATE_STYLES[state];
  const pulse = state === "ATTACK" || state === "WARNING";

  return (
    <section
      role="status"
      aria-live="polite"
      className={`flex flex-wrap items-center justify-between gap-4 rounded-xl border px-6 py-5 transition-colors duration-300 ${s.wrap}`}
    >
      <div className="flex items-center gap-4">
        <span className="relative flex h-3.5 w-3.5">
          {pulse && (
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${s.dot}`} />
          )}
          <span className={`relative inline-flex h-3.5 w-3.5 rounded-full ${s.dot}`} />
        </span>
        <s.Icon className="h-7 w-7" aria-hidden />
        <div>
          <h2 className="text-2xl font-bold tracking-widest">{s.label}</h2>
          <p className="text-sm opacity-80">{s.sub}</p>
        </div>
      </div>

      <dl className="flex items-center gap-8 font-mono text-sm">
        <div>
          <dt className="text-xs uppercase opacity-60">Time</dt>
          <dd>{row?.startTime ?? "--"}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase opacity-60">P(attack ≤60s)</dt>
          <dd className="text-lg font-semibold tabular-nums">
            {row ? (row.predProb * 100).toFixed(1) + "%" : "--"}
          </dd>
        </div>
        <div className="flex gap-2">
          {row && !row.forecastValid && (
            <span className="rounded bg-slate-700/60 px-2 py-0.5 text-xs text-slate-300">warming up</span>
          )}
          {row?.forecastAlert && (
            <span className="rounded bg-amber-400/20 px-2 py-0.5 text-xs text-amber-200">forecast alert</span>
          )}
        </div>
      </dl>
    </section>
  );
}