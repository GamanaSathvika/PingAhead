import { ShieldAlert, ShieldEllipsis, TriangleAlert } from "lucide-react";
import type { AlertEvent, AlertEventType } from "@/types/simulation";
import { clock, warningLead } from "@/lib/simulationUtils";

const STYLE: Record<AlertEventType, { label: string; wrap: string; text: string; Icon: typeof ShieldAlert }> = {
  WARNING: {
    label: "Forecast alert",
    wrap: "border-amber-500/40 bg-amber-500/10",
    text: "text-amber-300",
    Icon: TriangleAlert,
  },
  ATTACK: {
    label: "Attack started",
    wrap: "border-rose-500/50 bg-rose-500/10",
    text: "text-rose-300",
    Icon: ShieldAlert,
  },
  RECOVERING: {
    label: "Attack subsiding",
    wrap: "border-blue-500/40 bg-blue-500/10",
    text: "text-blue-300",
    Icon: ShieldEllipsis,
  },
};

export function AlertTicker({ events }: { events: AlertEvent[] }) {
  // Lead times need the chronological list; show newest first.
  const items = events
    .map((e, i) => ({ e, lead: e.type === "ATTACK" ? warningLead(events, i) : undefined }))
    .reverse();

  return (
    <div>
      <div className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold tracking-wide text-slate-200">Alert ticker</h3>
        <span className="text-xs text-slate-400">
          {events.length} {events.length === 1 ? "event" : "events"}
        </span>
      </div>

      {items.length === 0 ? (
        <div className="grid h-80 place-items-center text-sm text-slate-500">
          No alerts yet. Press Play to start the replay.
        </div>
      ) : (
        <ul role="log" aria-live="polite" className="flex h-80 flex-col gap-2 overflow-y-auto pr-1">
          {items.map(({ e, lead }) => {
            const s = STYLE[e.type];
            return (
              <li key={e.id} className={`rounded-lg border px-3 py-2 ${s.wrap}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className={`flex items-center gap-2 text-sm font-semibold ${s.text}`}>
                    <s.Icon className="h-4 w-4" aria-hidden />
                    {s.label}
                  </span>
                  <span className="font-mono text-xs text-slate-400">{clock(e.startTime, true)}</span>
                </div>
                <div className="mt-1 flex items-center justify-between gap-2 text-xs text-slate-400">
                  <span>
                    {e.from === "NORMAL" ? "" : `from ${e.from.toLowerCase()} · `}score{" "}
                    <span className="tabular-nums">{(e.predProb * 100).toFixed(1)}%</span>
                  </span>
                  {e.type === "ATTACK" &&
                    (lead != null ? (
                      <span className="rounded bg-emerald-500/15 px-2 py-0.5 text-emerald-300">
                        warned {Math.round(lead)} s ahead
                      </span>
                    ) : (
                      <span className="rounded bg-rose-500/15 px-2 py-0.5 text-rose-300">no advance warning</span>
                    ))}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
