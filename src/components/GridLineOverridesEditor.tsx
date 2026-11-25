import React, { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, MapPin, Plus, X } from 'lucide-react';
import type { GridLineColorOverrides } from '../utils/wormholeMultiverse';

interface GridLineOverridesEditorProps {
  cols: number;
  rows: number;
  lonOverrides?: GridLineColorOverrides;
  latOverrides?: GridLineColorOverrides;
  onChangeLonOverrides: (next: GridLineColorOverrides | undefined) => void;
  onChangeLatOverrides: (next: GridLineColorOverrides | undefined) => void;
  defaultOpen?: boolean;
}

const WAYPOINT_COLORS = [
  '#facc15', // Gold
  '#22c55e', // Emerald
  '#ffffff', // White
  '#fb7185', // Coral
  '#38bdf8', // Sky
  '#f97316', // Orange
  '#c084fc', // Violet
] as const;

function formatDeg(deg: number): string {
  const rounded = Math.abs(deg - Math.round(deg)) < 0.05 ? Math.round(deg) : Number(deg.toFixed(1));
  if (rounded === 0) return '0°';
  return rounded > 0 ? `+${rounded}°` : `${rounded}°`;
}

export function formatLongitudeLineLabel(index: number, cols: number): string {
  const safeCols = Math.max(1, cols);
  const deg = (index / safeCols - 0.5) * 360;
  let tag = '';
  if (Math.abs(deg) < 0.05) tag = ' · Front';
  else if (Math.abs(Math.abs(deg) - 180) < 0.05) tag = ' · Back';
  else if (Math.abs(deg + 90) < 0.05) tag = ' · Left';
  else if (Math.abs(deg - 90) < 0.05) tag = ' · Right';
  const degStr = Math.abs(Math.abs(deg) - 180) < 0.05 ? '±180°' : formatDeg(deg);
  return `#${index} (${degStr}${tag})`;
}

export function formatLatitudeLineLabel(index: number, rows: number): string {
  const safeRows = Math.max(1, rows);
  const deg = (index / safeRows - 0.5) * 180;
  if (Math.abs(deg) < 0.05) return `#${index} (0° · Equator)`;
  const hemi = deg > 0 ? 'N' : 'S';
  return `#${index} (${formatDeg(deg)} ${hemi})`;
}

function cleanRecord(rec: GridLineColorOverrides): GridLineColorOverrides | undefined {
  return Object.keys(rec).length > 0 ? rec : undefined;
}

export const GridLineOverridesEditor: React.FC<GridLineOverridesEditorProps> = ({
  cols,
  rows,
  lonOverrides,
  latOverrides,
  onChangeLonOverrides,
  onChangeLatOverrides,
  defaultOpen = false,
}) => {
  const safeCols = Math.max(2, Math.round(cols));
  const safeRows = Math.max(2, Math.round(rows));

  const activeEntries = useMemo(() => {
    const items: Array<{ axis: 'lon' | 'lat'; index: number; color: string; inRange: boolean }> = [];
    if (lonOverrides) {
      for (const [k, color] of Object.entries(lonOverrides)) {
        const index = Number(k);
        if (Number.isInteger(index)) {
          items.push({ axis: 'lon', index, color, inRange: index >= 0 && index < safeCols });
        }
      }
    }
    if (latOverrides) {
      for (const [k, color] of Object.entries(latOverrides)) {
        const index = Number(k);
        if (Number.isInteger(index)) {
          items.push({ axis: 'lat', index, color, inRange: index >= 1 && index < safeRows });
        }
      }
    }
    items.sort((a, b) => (a.axis !== b.axis ? (a.axis === 'lon' ? -1 : 1) : a.index - b.index));
    return items;
  }, [lonOverrides, latOverrides, safeCols, safeRows]);

  const [expanded, setExpanded] = useState(defaultOpen || activeEntries.length > 0);
  const [draftAxis, setDraftAxis] = useState<'lon' | 'lat'>('lon');
  const [draftIndex, setDraftIndex] = useState<number>(() => Math.floor(safeCols / 2));
  const [draftColor, setDraftColor] = useState<string>(WAYPOINT_COLORS[0]);

  // Keep draftIndex valid when switching axis or resizing grid density
  useEffect(() => {
    if (draftAxis === 'lon') {
      if (draftIndex < 0 || draftIndex >= safeCols) {
        setDraftIndex(Math.floor(safeCols / 2));
      }
    } else {
      if (draftIndex < 1 || draftIndex >= safeRows) {
        setDraftIndex(Math.max(1, Math.floor(safeRows / 2)));
      }
    }
  }, [draftAxis, draftIndex, safeCols, safeRows]);

  const setLineColor = (axis: 'lon' | 'lat', index: number, color: string) => {
    if (axis === 'lon') {
      onChangeLonOverrides({ ...(lonOverrides ?? {}), [index]: color });
    } else {
      onChangeLatOverrides({ ...(latOverrides ?? {}), [index]: color });
    }
  };

  const removeLineColor = (axis: 'lon' | 'lat', index: number) => {
    if (axis === 'lon') {
      if (!lonOverrides || !(index in lonOverrides)) return;
      const next = { ...lonOverrides };
      delete next[index];
      onChangeLonOverrides(cleanRecord(next));
    } else {
      if (!latOverrides || !(index in latOverrides)) return;
      const next = { ...latOverrides };
      delete next[index];
      onChangeLatOverrides(cleanRecord(next));
    }
  };

  const toggleLandmark = (axis: 'lon' | 'lat', index: number, color: string) => {
    const existing = axis === 'lon' ? lonOverrides?.[index] : latOverrides?.[index];
    if (existing) {
      removeLineColor(axis, index);
    } else {
      setLineColor(axis, index, color);
    }
  };

  const handleAddDraft = () => {
    setLineColor(draftAxis, draftIndex, draftColor);
    // Cycle next suggested color so multi-waypoint authoring is immediately distinct
    const nextColorIdx = (activeEntries.length + 1) % WAYPOINT_COLORS.length;
    setDraftColor(WAYPOINT_COLORS[nextColorIdx]);
  };

  const frontIdx = Math.floor(safeCols / 2);
  const backIdx = 0;
  const leftIdx = Math.round(safeCols / 4) % safeCols;
  const rightIdx = Math.round((3 * safeCols) / 4) % safeCols;
  const equatorIdx = Math.max(1, Math.min(safeRows - 1, Math.round(safeRows / 2)));

  const quickLandmarks = [
    { label: 'Front', axis: 'lon' as const, index: frontIdx, color: '#facc15' },
    { label: 'Back', axis: 'lon' as const, index: backIdx, color: '#ffffff' },
    ...(safeCols >= 4 && leftIdx !== frontIdx && leftIdx !== backIdx
      ? [{ label: 'Left', axis: 'lon' as const, index: leftIdx, color: '#22c55e' }]
      : []),
    ...(safeCols >= 4 && rightIdx !== frontIdx && rightIdx !== backIdx
      ? [{ label: 'Right', axis: 'lon' as const, index: rightIdx, color: '#fb7185' }]
      : []),
    {
      label: safeRows % 2 === 0 ? 'Equator' : `Lat #${equatorIdx}`,
      axis: 'lat' as const,
      index: equatorIdx,
      color: '#a3e635',
    },
  ];

  const lineOptions = useMemo(() => {
    if (draftAxis === 'lon') {
      return Array.from({ length: safeCols }, (_, idx) => ({
        index: idx,
        label: formatLongitudeLineLabel(idx, safeCols),
      }));
    }
    return Array.from({ length: Math.max(1, safeRows - 1) }, (_, i) => {
      const idx = i + 1;
      return {
        index: idx,
        label: formatLatitudeLineLabel(idx, safeRows),
      };
    });
  }, [draftAxis, safeCols, safeRows]);

  return (
    <div className="flex flex-col gap-1.5 border-t border-slate-800/80 pt-1.5">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between text-left group"
      >
        <span className="flex items-center gap-1 text-[9px] font-bold uppercase tracking-wider text-slate-300 group-hover:text-fuchsia-200 transition">
          {expanded ? (
            <ChevronDown className="h-3 w-3 text-slate-400" />
          ) : (
            <ChevronRight className="h-3 w-3 text-slate-400" />
          )}
          <MapPin className="h-2.5 w-2.5 text-amber-300" />
          <span>Reference Lines</span>
        </span>
        <span className="font-mono text-[9px] text-slate-400 tabular-nums">
          {activeEntries.length > 0 ? `${activeEntries.length} custom` : 'None'}
        </span>
      </button>

      {expanded && (
        <div className="flex flex-col gap-2 rounded border border-slate-800/90 bg-slate-950/80 p-2">
          {/* One-click cardinal landmark toggles */}
          <div className="flex flex-wrap items-center gap-1">
            <span className="text-[9px] text-slate-500 mr-0.5">Waypoints:</span>
            {quickLandmarks.map((lm) => {
              const activeColor =
                lm.axis === 'lon' ? lonOverrides?.[lm.index] : latOverrides?.[lm.index];
              const isOn = Boolean(activeColor);
              return (
                <button
                  key={`${lm.axis}-${lm.label}`}
                  type="button"
                  onClick={() => toggleLandmark(lm.axis, lm.index, lm.color)}
                  title={`${isOn ? 'Remove' : 'Highlight'} ${lm.axis === 'lon' ? 'Longitude' : 'Latitude'} ${
                    lm.axis === 'lon'
                      ? formatLongitudeLineLabel(lm.index, safeCols)
                      : formatLatitudeLineLabel(lm.index, safeRows)
                  }`}
                  className={`flex items-center gap-1 rounded border px-1.5 py-0.5 text-[9px] font-medium transition ${
                    isOn
                      ? 'border-amber-400/70 bg-amber-950/60 text-amber-100'
                      : 'border-slate-800 bg-slate-900 text-slate-400 hover:border-slate-600 hover:text-slate-200'
                  }`}
                >
                  <span
                    className="h-2 w-2 rounded-full border border-white/30"
                    style={{ backgroundColor: activeColor ?? lm.color }}
                  />
                  <span>{lm.label}</span>
                </button>
              );
            })}
          </div>

          {/* Arbitrary line selector & color picker */}
          <div className="flex flex-col gap-1.5 border-t border-slate-800/80 pt-1.5">
            <div className="grid grid-cols-2 gap-1">
              <button
                type="button"
                onClick={() => {
                  setDraftAxis('lon');
                  setDraftIndex(frontIdx);
                }}
                className={`rounded border py-0.5 text-[9px] font-semibold transition ${
                  draftAxis === 'lon'
                    ? 'border-fuchsia-400/80 bg-fuchsia-950/80 text-fuchsia-100'
                    : 'border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200'
                }`}
              >
                Longitude (0–{safeCols - 1})
              </button>
              <button
                type="button"
                onClick={() => {
                  setDraftAxis('lat');
                  setDraftIndex(equatorIdx);
                }}
                className={`rounded border py-0.5 text-[9px] font-semibold transition ${
                  draftAxis === 'lat'
                    ? 'border-fuchsia-400/80 bg-fuchsia-950/80 text-fuchsia-100'
                    : 'border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200'
                }`}
              >
                Latitude (1–{safeRows - 1})
              </button>
            </div>

            <div className="flex items-center gap-1">
              <select
                value={draftIndex}
                aria-label={`Select ${draftAxis === 'lon' ? 'longitude' : 'latitude'} line`}
                onChange={(e) => setDraftIndex(Number(e.target.value))}
                className="min-w-0 flex-1 rounded border border-slate-700 bg-slate-900 px-1.5 py-1 font-mono text-[10px] text-slate-100 outline-none focus:border-fuchsia-400"
              >
                {lineOptions.map((opt) => (
                  <option key={opt.index} value={opt.index}>
                    {opt.label}
                  </option>
                ))}
              </select>

              <label
                className="relative flex h-6 w-7 shrink-0 cursor-pointer items-center justify-center rounded border border-slate-600 overflow-hidden"
                title={`Override color: ${draftColor}`}
              >
                <span className="h-full w-full" style={{ backgroundColor: draftColor }} />
                <input
                  type="color"
                  value={draftColor}
                  aria-label="New line override color"
                  onChange={(e) => setDraftColor(e.target.value)}
                  className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                />
              </label>

              <button
                type="button"
                onClick={handleAddDraft}
                className="flex shrink-0 items-center gap-0.5 rounded border border-fuchsia-500/60 bg-fuchsia-950/70 px-2 py-1 text-[10px] font-semibold text-fuchsia-100 transition hover:bg-fuchsia-900"
              >
                <Plus className="h-3 w-3" />
                <span>Set</span>
              </button>
            </div>
          </div>

          {/* Active line overrides list */}
          {activeEntries.length > 0 && (
            <div className="flex flex-col gap-1 border-t border-slate-800/80 pt-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-semibold text-slate-400">Active Overrides</span>
                {activeEntries.length > 1 && (
                  <button
                    type="button"
                    onClick={() => {
                      onChangeLonOverrides(undefined);
                      onChangeLatOverrides(undefined);
                    }}
                    className="text-[9px] text-slate-400 hover:text-rose-300 transition"
                  >
                    Clear all
                  </button>
                )}
              </div>

              <div className="flex max-h-32 flex-col gap-1 overflow-y-auto pr-0.5">
                {activeEntries.map((entry) => (
                  <div
                    key={`${entry.axis}-${entry.index}`}
                    className="flex items-center justify-between gap-1.5 rounded border border-slate-800 bg-slate-900/90 px-1.5 py-1"
                  >
                    <div className="flex min-w-0 items-center gap-1.5">
                      <label
                        className="relative h-3.5 w-3.5 shrink-0 cursor-pointer overflow-hidden rounded border border-white/30"
                        title={`Change color (${entry.color})`}
                      >
                        <span
                          className="block h-full w-full"
                          style={{ backgroundColor: entry.color }}
                        />
                        <input
                          type="color"
                          value={entry.color}
                          aria-label={`Color for ${entry.axis} line ${entry.index}`}
                          onChange={(e) => setLineColor(entry.axis, entry.index, e.target.value)}
                          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                        />
                      </label>
                      <span
                        className={`shrink-0 rounded px-1 py-0.2 font-mono text-[8px] font-bold uppercase ${
                          entry.axis === 'lon'
                            ? 'bg-sky-950/90 text-sky-300 border border-sky-500/30'
                            : 'bg-emerald-950/90 text-emerald-300 border border-emerald-500/30'
                        }`}
                      >
                        {entry.axis}
                      </span>
                      <span className="truncate font-mono text-[9.5px] text-slate-200 tabular-nums">
                        {entry.axis === 'lon'
                          ? formatLongitudeLineLabel(entry.index, safeCols)
                          : formatLatitudeLineLabel(entry.index, safeRows)}
                      </span>
                      {!entry.inRange && (
                        <span
                          className="shrink-0 font-mono text-[8px] text-amber-400"
                          title="Line index exceeds current grid density"
                        >
                          (hidden)
                        </span>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={() => removeLineColor(entry.axis, entry.index)}
                      className="shrink-0 text-slate-500 hover:text-rose-300 transition"
                      title="Remove line override"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
