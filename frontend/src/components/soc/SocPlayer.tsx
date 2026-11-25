"use client";
import { useState } from "react";
import { useSimulationData } from "@/hooks/useSimulationData";
import { useSimulationPlayer } from "@/hooks/useSimulationPlayer";
import { PlaybackControls } from "./PlaybackControls";
import { StatusBanner } from "./StatusBanner";
import { ProbabilityChart } from "./ProbabilityChart";
import { AlertTicker } from "./AlertTicker";
import { RESOLUTIONS, SCENARIOS, type Resolution, type Scenario } from "@/types/simulation";

const select =
  "rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500";

export default function SocPlayer() {
  const [scenario, setScenario] = useState<Scenario>(SCENARIOS[0]);
  const [resolution, setResolution] = useState<Resolution>("10s");

  const { rows, loading, error } = useSimulationData(scenario, resolution);
  const player = useSimulationPlayer({ rows });

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-wide">
              PingAhead <span className="text-slate-500">/ SOC Simulation</span>
            </h1>
          </div>
          <div className="flex gap-3">
            <select className={select} value={scenario} onChange={(e) => setScenario(e.target.value as Scenario)} aria-label="Scenario">
              {SCENARIOS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select className={select} value={resolution} onChange={(e) => setResolution(e.target.value as Resolution)} aria-label="Resolution">
              {RESOLUTIONS.map((r) => <option key={r} value={r}>{r} bins</option>)}
            </select>
          </div>
        </header>

        {error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            Couldn't load simulation: {error}
          </div>
        )}

        <StatusBanner row={player.currentRow} />
        <PlaybackControls player={player} rows={rows} disabled={loading || rows.length === 0} />

        <div className="grid gap-6 lg:grid-cols-3">
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-4 lg:col-span-2">
            <ProbabilityChart rows={rows} index={player.index} />
          </section>
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <AlertTicker events={player.events} />
          </section>
        </div>
      </div>
    </main>
  );
}