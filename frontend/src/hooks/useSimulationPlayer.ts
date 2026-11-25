"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  AlertEvent,
  PlaybackSpeed,
  SimulationRow,
} from "@/types/simulation";

interface Options {
  rows: SimulationRow[];
  /** Wall-clock ms per row at 1x. */
  baseTickMs?: number;
  loop?: boolean;
}

/** Build the full event log once; the playhead just slices it. */
function deriveEvents(rows: SimulationRow[]): AlertEvent[] {
  const events: AlertEvent[] = [];
  for (let i = 1; i < rows.length; i++) {
    const prev = rows[i - 1].uiState;
    const cur = rows[i];
    if (cur.uiState !== prev && cur.uiState !== "NORMAL") {
      events.push({
        id: `${i}-${cur.uiState}`,
        rowIndex: i,
        startTime: cur.startTime,
        type: cur.uiState,
        from: prev,
        predProb: cur.predProb,
      });
    }
  }
  // If the file starts mid-incident, log the first row too
  if (rows[0] && rows[0].uiState !== "NORMAL") {
    events.unshift({
      id: `0-${rows[0].uiState}`,
      rowIndex: 0,
      startTime: rows[0].startTime,
      type: rows[0].uiState,
      from: "NORMAL",
      predProb: rows[0].predProb,
    });
  }
  return events;
}

export function useSimulationPlayer({ rows, baseTickMs = 1000, loop = false }: Options) {
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);

  // Refs mirror state so the rAF loop never reads stale closures.
  const indexRef = useRef(0);
  const speedRef = useRef<PlaybackSpeed>(1);
  const lastIndexRef = useRef(0);

  const lastIndex = Math.max(rows.length - 1, 0);
  lastIndexRef.current = lastIndex;
  speedRef.current = speed;

  const commit = useCallback((next: number) => {
    const clamped = Math.min(Math.max(next, 0), lastIndexRef.current);
    indexRef.current = clamped;
    setIndex(clamped);
    return clamped;
  }, []);

  // New dataset (scenario / resolution change) -> rewind and pause.
  useEffect(() => {
    setIsPlaying(false);
    commit(0);
  }, [rows, commit]);

  // ---- Playback loop -------------------------------------------------------
  useEffect(() => {
    if (!isPlaying || rows.length === 0) return;

    let raf = 0;
    let last = performance.now();
    let acc = 0;

    const frame = (now: number) => {
      // Clamp so a backgrounded tab doesn't fast-forward on return.
      acc += Math.min(now - last, 1000);
      last = now;

      const interval = baseTickMs / speedRef.current;
      if (acc >= interval) {
        const steps = Math.floor(acc / interval);
        acc -= steps * interval;

        const target = indexRef.current + steps;
        if (target >= lastIndexRef.current) {
          if (loop) {
            commit(0);
          } else {
            commit(lastIndexRef.current);
            setIsPlaying(false);
            return; // stop the loop
          }
        } else {
          commit(target);
        }
      }
      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, rows.length, baseTickMs, loop, commit]);

  // ---- Controls ------------------------------------------------------------
  const play = useCallback(() => {
    if (rows.length === 0) return;
    if (indexRef.current >= lastIndexRef.current) commit(0); // replay from end
    setIsPlaying(true);
  }, [rows.length, commit]);

  const pause = useCallback(() => setIsPlaying(false), []);
  const toggle = useCallback(() => (isPlaying ? pause() : play()), [isPlaying, play, pause]);

  const reset = useCallback(() => {
    setIsPlaying(false);
    commit(0);
  }, [commit]);

  const stepForward = useCallback(() => {
    setIsPlaying(false);
    commit(indexRef.current + 1);
  }, [commit]);

  const stepBackward = useCallback(() => {
    setIsPlaying(false);
    commit(indexRef.current - 1);
  }, [commit]);

  /** Scrub: keeps playing state so dragging while playing resumes naturally. */
  const seek = useCallback((i: number) => void commit(i), [commit]);

  // ---- Derived data --------------------------------------------------------
  const currentRow = rows[index] ?? null;

  /** Rows revealed so far; feeds the chart's progressive reveal. */
  const visibleRows = useMemo(() => rows.slice(0, index + 1), [rows, index]);

  const allEvents = useMemo(() => deriveEvents(rows), [rows]);
  const events = useMemo(
    () => allEvents.filter((e) => e.rowIndex <= index),
    [allEvents, index]
  );

  return {
    // state
    index,
    lastIndex,
    isPlaying,
    speed,
    progress: lastIndex > 0 ? index / lastIndex : 0,
    isAtEnd: index >= lastIndex,
    // data
    currentRow,
    visibleRows,
    events,
    // controls
    play,
    pause,
    toggle,
    reset,
    stepForward,
    stepBackward,
    seek,
    setSpeed,
  };
}

export type SimulationPlayer = ReturnType<typeof useSimulationPlayer>;