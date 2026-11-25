"use client";
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { PLAYBACK_SPEEDS } from "@/types/simulation";
import type { SimulationPlayer } from "@/hooks/useSimulationPlayer";
import type { SimulationRow } from "@/types/simulation";

const iconBtn =
  "inline-flex h-10 w-10 items-center justify-center rounded-lg border border-slate-700 bg-slate-800 text-slate-200 " +
  "transition hover:bg-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 " +
  "disabled:cursor-not-allowed disabled:opacity-40";

interface Props {
  player: SimulationPlayer;
  rows: SimulationRow[];
  disabled?: boolean;
}

export function PlaybackControls({ player, rows, disabled }: Props) {
  const { index, lastIndex, isPlaying, speed } = player;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <button className={iconBtn} onClick={player.reset} disabled={disabled} aria-label="Reset">
          <RotateCcw className="h-4 w-4" />
        </button>
        <button className={iconBtn} onClick={player.stepBackward} disabled={disabled || index === 0} aria-label="Step back">
          <SkipBack className="h-4 w-4" />
        </button>

        <button
          onClick={player.toggle}
          disabled={disabled}
          aria-label={isPlaying ? "Pause" : "Play"}
          className="inline-flex h-10 min-w-24 items-center justify-center gap-2 rounded-lg bg-sky-600 px-4 font-medium text-white
                     transition hover:bg-sky-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300
                     disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {isPlaying ? "Pause" : "Play"}
        </button>

        <button className={iconBtn} onClick={player.stepForward} disabled={disabled || player.isAtEnd} aria-label="Step forward">
          <SkipForward className="h-4 w-4" />
        </button>

        {/* Speed segmented control */}
        <div className="ml-auto inline-flex overflow-hidden rounded-lg border border-slate-700" role="group" aria-label="Playback speed">
          {PLAYBACK_SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => player.setSpeed(s)}
              aria-pressed={speed === s}
              className={`px-3 py-2 text-sm font-medium tabular-nums transition ${
                speed === s ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Scrubber */}
      <div className="mt-4">
        <input
          type="range"
          min={0}
          max={lastIndex}
          step={1}
          value={index}
          disabled={disabled}
          onChange={(e) => player.seek(Number(e.target.value))}
          aria-label="Timeline scrubber"
          className="h-2 w-full cursor-pointer accent-sky-500 disabled:opacity-40"
        />
        <div className="mt-1 flex justify-between font-mono text-xs text-slate-400">
          <span>{rows[0]?.startTime ?? "--"}</span>
          <span className="text-slate-200">
            {index + 1} / {rows.length || 0}
          </span>
          <span>{rows[lastIndex]?.startTime ?? "--"}</span>
        </div>
      </div>
    </div>
  );
}