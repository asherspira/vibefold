import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Box,
  Compass,
  Download,
  GitBranch,
  Link2,
  Maximize2,
  Minimize2,
  Plus,
  RotateCcw,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import {
  createDefaultUniverseNode,
  createDefaultWormholeEdge,
  eulerDegFromQuaternion,
  MULTIVERSE_PRESETS,
  parseMultiverseGraphJson,
  quaternionFromEulerDeg,
  serializeMultiverseGraph,
  type MouthOverlapWarning,
  type MultiverseGraph,
  type MultiversePresetId,
  type SerializedCameraKeyframe,
  type UniverseNode,
  type UniverseSkyMode,
  type WormholeEdge,
  type WormholeMouthPlacement,
} from '../utils/wormholeMultiverse';
import { DEFAULT_SHAPE_SCALE, WORMHOLE_SHAPES, type WormholeShapeId } from '../utils/wormholeSdf';
import { GridLineOverridesEditor } from './GridLineOverridesEditor';

export interface UniverseSkyMeta {
  fileLabel: string | null;
  isLoading: boolean;
  error: string | null;
}

export type WorkspaceSelection =
  | { kind: 'universe'; id: string }
  | { kind: 'wormhole'; id: string; end: 'A' | 'B' }
  | null;

interface MultiverseWorkspaceDrawerProps {
  open: boolean;
  onClose: () => void;
  graph: MultiverseGraph;
  onChangeGraph: (updater: (prev: MultiverseGraph) => MultiverseGraph) => void;
  activePreset: MultiversePresetId | 'custom';
  onLoadPreset: (preset: MultiversePresetId) => void;
  activeUniverseIndex: number;
  onTeleportToUniverse: (universeIndex: number) => void;
  onFrameMouth: (universeIndex: number, position: [number, number, number], radius: number) => void;
  selection: WorkspaceSelection;
  onSelect: (selection: WorkspaceSelection) => void;
  overlaps: readonly MouthOverlapWarning[];
  showOverlay3d: boolean;
  onToggleOverlay3d: () => void;
  skyMetaById: Record<string, UniverseSkyMeta>;
  onUploadUniverseSky: (universeId: string, file: File) => void;
  onClearUniverseSky: (universeId: string) => void;
  onDeleteUniverseCleanup: (universeId: string) => void;
  keyframes?: readonly SerializedCameraKeyframe[];
  onImportGraph: (
    graph: MultiverseGraph,
    keyframes: SerializedCameraKeyframe[],
    warnings: string[]
  ) => void;
}

const GRID_PALETTES = [
  { v: '#ff0000', h: '#0000ff', name: 'Red/Blue' },
  { v: '#00ffff', h: '#ff00ff', name: 'Cyan/Magenta' },
  { v: '#f59e0b', h: '#10b981', name: 'Amber/Emerald' },
  { v: '#a855f7', h: '#38bdf8', name: 'Violet/Sky' },
  { v: '#fb7185', h: '#facc15', name: 'Coral/Gold' },
  { v: '#ffffff', h: '#555555', name: 'Mono' },
] as const;

const GRAPH_VIEW_W = 480;
const GRAPH_VIEW_H = 240;
const NODE_RADIUS = 24;

function computeDefaultNodePosition(index: number, total: number): { x: number; y: number } {
  const cx = GRAPH_VIEW_W / 2;
  const cy = GRAPH_VIEW_H / 2;
  if (total <= 1) return { x: cx, y: cy };
  if (total === 2) {
    return { x: index === 0 ? cx - 120 : cx + 120, y: cy };
  }
  const rx = Math.min(165, 105 + total * 10);
  const ry = Math.min(82, 62 + total * 4);
  const angle = -Math.PI / 2 + (index / total) * Math.PI * 2;
  return {
    x: cx + Math.cos(angle) * rx,
    y: cy + Math.sin(angle) * ry,
  };
}

interface NumericSliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  display: string;
  onChange: (value: number) => void;
  accent?: 'fuchsia' | 'amber' | 'sky';
}

const NumericSlider: React.FC<NumericSliderProps> = ({
  label,
  min,
  max,
  step,
  value,
  display,
  onChange,
  accent = 'fuchsia',
}) => {
  const accentClass =
    accent === 'amber'
      ? 'accent-amber-400'
      : accent === 'sky'
        ? 'accent-sky-400'
        : 'accent-fuchsia-400';
  const textClass =
    accent === 'amber'
      ? 'text-amber-200'
      : accent === 'sky'
        ? 'text-sky-200'
        : 'text-fuchsia-200';

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between text-[10px]">
        <span className="font-medium text-slate-300">{label}</span>
        <span className={`font-mono tabular-nums ${textClass}`}>{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`h-1.5 w-full cursor-pointer ${accentClass}`}
      />
    </div>
  );
};

export const MultiverseWorkspaceDrawer: React.FC<MultiverseWorkspaceDrawerProps> = ({
  open,
  onClose,
  graph,
  onChangeGraph,
  activePreset,
  onLoadPreset,
  activeUniverseIndex,
  onTeleportToUniverse,
  onFrameMouth,
  selection,
  onSelect,
  overlaps,
  showOverlay3d,
  onToggleOverlay3d,
  skyMetaById,
  onUploadUniverseSky,
  onClearUniverseSky,
  onDeleteUniverseCleanup,
  keyframes,
  onImportGraph,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [linkSourceId, setLinkSourceId] = useState<string | null>(null);
  const [importNotice, setImportNotice] = useState<{ kind: 'info' | 'error'; text: string } | null>(null);
  const [nodePositions, setNodePositions] = useState<Record<string, { x: number; y: number }>>({});
  const draggingRef = useRef<{
    id: string;
    moved: boolean;
    startClientX: number;
    startClientY: number;
  } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const jsonInputRef = useRef<HTMLInputElement | null>(null);
  const skyInputRef = useRef<HTMLInputElement | null>(null);

  // Ensure every universe has a valid position inside the SVG viewport
  useEffect(() => {
    setNodePositions((prev) => {
      const next: Record<string, { x: number; y: number }> = {};
      let changed = false;
      const total = graph.universes.length;
      graph.universes.forEach((u, idx) => {
        if (prev[u.id]) {
          next[u.id] = prev[u.id];
        } else {
          next[u.id] = computeDefaultNodePosition(idx, total);
          changed = true;
        }
      });
      if (Object.keys(prev).length !== Object.keys(next).length) changed = true;
      return changed ? next : prev;
    });
  }, [graph.universes]);

  // Reset layout cleanly when switching presets or importing
  const resetGraphLayout = () => {
    const next: Record<string, { x: number; y: number }> = {};
    graph.universes.forEach((u, idx) => {
      next[u.id] = computeDefaultNodePosition(idx, graph.universes.length);
    });
    setNodePositions(next);
  };

  // Validate selection still exists
  const selectedUniverse: UniverseNode | null = useMemo(() => {
    if (selection?.kind === 'universe') {
      return graph.universes.find((u) => u.id === selection.id) ?? null;
    }
    return null;
  }, [graph.universes, selection]);

  const selectedWormhole: WormholeEdge | null = useMemo(() => {
    if (selection?.kind === 'wormhole') {
      return graph.wormholes.find((w) => w.id === selection.id) ?? null;
    }
    return null;
  }, [graph.wormholes, selection]);

  const selectedEnd: 'A' | 'B' = selection?.kind === 'wormhole' ? selection.end : 'A';

  // Ensure something is selected when open
  useEffect(() => {
    if (!open) return;
    if (selection?.kind === 'universe' && selectedUniverse) return;
    if (selection?.kind === 'wormhole' && selectedWormhole) return;
    const fallbackU = graph.universes[activeUniverseIndex] ?? graph.universes[0];
    if (fallbackU) {
      onSelect({ kind: 'universe', id: fallbackU.id });
    }
  }, [open, selection, selectedUniverse, selectedWormhole, graph.universes, activeUniverseIndex, onSelect]);

  // Count mouths per universe for badges
  const mouthCountsByUniverse = useMemo(() => {
    const counts = new Map<string, number>();
    for (const u of graph.universes) counts.set(u.id, 0);
    for (const wh of graph.wormholes) {
      counts.set(wh.mouthA.universeId, (counts.get(wh.mouthA.universeId) ?? 0) + 1);
      counts.set(wh.mouthB.universeId, (counts.get(wh.mouthB.universeId) ?? 0) + 1);
    }
    return counts;
  }, [graph]);

  // Track overlapping wormhole IDs for red edge highlighting
  const overlappingWormholeIds = useMemo(() => {
    const set = new Set<string>();
    for (const w of overlaps) {
      set.add(w.wormholeAId);
      set.add(w.wormholeBId);
    }
    return set;
  }, [overlaps]);

  // Group edges between the same unordered pair of universes so parallel bridges fan out cleanly
  const edgeLayoutMeta = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const wh of graph.wormholes) {
      const a = wh.mouthA.universeId;
      const b = wh.mouthB.universeId;
      const key = a <= b ? `${a}::${b}` : `${b}::${a}`;
      const list = groups.get(key) ?? [];
      list.push(wh.id);
      groups.set(key, list);
    }
    const meta = new Map<string, { index: number; total: number }>();
    for (const list of groups.values()) {
      list.forEach((id, idx) => {
        meta.set(id, { index: idx, total: list.length });
      });
    }
    return meta;
  }, [graph.wormholes]);

  if (!open) return null;

  const activeUniverseId = graph.universes[activeUniverseIndex]?.id ?? graph.universes[0]?.id ?? '';

  const handleAddUniverse = () => {
    const newNode = createDefaultUniverseNode(graph.universes);
    onChangeGraph((prev) => ({
      ...prev,
      universes: [...prev.universes, newNode],
    }));
    setLinkSourceId(null);
    onSelect({ kind: 'universe', id: newNode.id });
  };

  const handleAddWormhole = (fromId?: string, toId?: string) => {
    if (graph.universes.length === 0) return;
    const sourceId =
      fromId ??
      (selection?.kind === 'universe' ? selection.id : activeUniverseId) ??
      graph.universes[0].id;
    let targetId = toId;
    if (!targetId) {
      const other = graph.universes.find((u) => u.id !== sourceId);
      targetId = other ? other.id : sourceId;
    }
    const newWh = createDefaultWormholeEdge(graph, sourceId, targetId);
    onChangeGraph((prev) => ({
      ...prev,
      wormholes: [...prev.wormholes, newWh],
    }));
    setLinkSourceId(null);
    onSelect({ kind: 'wormhole', id: newWh.id, end: 'A' });
  };

  const handleDeleteUniverse = (universeId: string) => {
    if (graph.universes.length <= 1) return;
    onDeleteUniverseCleanup(universeId);
    onChangeGraph((prev) => {
      const nextUniverses = prev.universes.filter((u) => u.id !== universeId);
      const nextWormholes = prev.wormholes.filter(
        (w) => w.mouthA.universeId !== universeId && w.mouthB.universeId !== universeId
      );
      return { universes: nextUniverses, wormholes: nextWormholes };
    });
    const fallback = graph.universes.find((u) => u.id !== universeId);
    onSelect(fallback ? { kind: 'universe', id: fallback.id } : null);
  };

  const handleDeleteWormhole = (wormholeId: string) => {
    onChangeGraph((prev) => ({
      ...prev,
      wormholes: prev.wormholes.filter((w) => w.id !== wormholeId),
    }));
    const fallbackU = graph.universes[activeUniverseIndex] ?? graph.universes[0];
    onSelect(fallbackU ? { kind: 'universe', id: fallbackU.id } : null);
  };

  const updateUniverse = (universeId: string, patch: Partial<UniverseNode>) => {
    onChangeGraph((prev) => ({
      ...prev,
      universes: prev.universes.map((u) => (u.id === universeId ? { ...u, ...patch } : u)),
    }));
  };

  const updateWormhole = (wormholeId: string, updater: (wh: WormholeEdge) => WormholeEdge) => {
    onChangeGraph((prev) => ({
      ...prev,
      wormholes: prev.wormholes.map((w) => (w.id === wormholeId ? updater(w) : w)),
    }));
  };

  const updateSelectedMouth = (
    wormholeId: string,
    end: 'A' | 'B',
    patch: Partial<WormholeMouthPlacement>
  ) => {
    updateWormhole(wormholeId, (wh) =>
      end === 'A'
        ? { ...wh, mouthA: { ...wh.mouthA, ...patch } }
        : { ...wh, mouthB: { ...wh.mouthB, ...patch } }
    );
  };

  const handleExportJson = () => {
    const json = serializeMultiverseGraph(graph, keyframes);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.download = `wormhole-multiverse-${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    const kfNote = keyframes && keyframes.length > 0 ? ` and ${keyframes.length} keyframe(s)` : '';
    setImportNotice({
      kind: 'info',
      text: `Exported ${graph.universes.length} universe(s), ${graph.wormholes.length} wormhole(s)${kfNote}.`,
    });
  };

  const handleImportFile = async (file: File) => {
    try {
      const text = await file.text();
      const parsed = parseMultiverseGraphJson(text);
      onImportGraph(parsed.graph, parsed.keyframes, parsed.warnings);
      const nextPositions: Record<string, { x: number; y: number }> = {};
      parsed.graph.universes.forEach((u, idx) => {
        nextPositions[u.id] = computeDefaultNodePosition(idx, parsed.graph.universes.length);
      });
      setNodePositions(nextPositions);
      const kfNote = parsed.keyframes.length > 0 ? ` · ${parsed.keyframes.length} keyframes` : '';
      setImportNotice({
        kind: 'info',
        text:
          parsed.warnings.length > 0
            ? `Imported ${parsed.graph.universes.length}U / ${parsed.graph.wormholes.length}W${kfNote}. ${parsed.warnings[0]}`
            : `Imported ${parsed.graph.universes.length} universe(s), ${parsed.graph.wormholes.length} wormhole(s)${kfNote}.`,
      });
    } catch (err) {
      setImportNotice({
        kind: 'error',
        text: err instanceof Error ? err.message : 'Failed to parse multiverse JSON.',
      });
    }
  };

  // SVG Pointer Dragging for Universe Nodes
  const clientToSvgPoint = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: GRAPH_VIEW_W / 2, y: GRAPH_VIEW_H / 2 };
    const rect = svg.getBoundingClientRect();
    const x = ((clientX - rect.left) / Math.max(rect.width, 1)) * GRAPH_VIEW_W;
    const y = ((clientY - rect.top) / Math.max(rect.height, 1)) * GRAPH_VIEW_H;
    return {
      x: Math.max(NODE_RADIUS + 14, Math.min(GRAPH_VIEW_W - NODE_RADIUS - 14, x)),
      y: Math.max(NODE_RADIUS + 14, Math.min(GRAPH_VIEW_H - NODE_RADIUS - 18, y)),
    };
  };

  const handleNodePointerDown = (e: React.PointerEvent<SVGGElement>, universeId: string) => {
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    draggingRef.current = {
      id: universeId,
      moved: false,
      startClientX: e.clientX,
      startClientY: e.clientY,
    };
  };

  const handleSvgPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const drag = draggingRef.current;
    if (!drag) return;
    if (!drag.moved) {
      const dist = Math.hypot(e.clientX - drag.startClientX, e.clientY - drag.startClientY);
      if (dist < 4) return;
      drag.moved = true;
    }
    const pt = clientToSvgPoint(e.clientX, e.clientY);
    setNodePositions((prev) => ({ ...prev, [drag.id]: pt }));
  };

  const handleSvgPointerUp = () => {
    const drag = draggingRef.current;
    draggingRef.current = null;
    if (!drag) return;
    if (!drag.moved) {
      if (linkSourceId) {
        handleAddWormhole(linkSourceId, drag.id);
      } else {
        onSelect({ kind: 'universe', id: drag.id });
      }
    }
  };

  const selectedUniverseMeta: UniverseSkyMeta = selectedUniverse
    ? skyMetaById[selectedUniverse.id] ?? { fileLabel: null, isLoading: false, error: null }
    : { fileLabel: null, isLoading: false, error: null };

  const activeMouth: WormholeMouthPlacement | null = selectedWormhole
    ? selectedEnd === 'A'
      ? selectedWormhole.mouthA
      : selectedWormhole.mouthB
    : null;
  const partnerMouth: WormholeMouthPlacement | null = selectedWormhole
    ? selectedEnd === 'A'
      ? selectedWormhole.mouthB
      : selectedWormhole.mouthA
    : null;

  const activeMouthEuler = activeMouth ? eulerDegFromQuaternion(activeMouth.quaternion) : [0, 0, 0];

  const universeNameById = new Map(graph.universes.map((u) => [u.id, u.name]));
  const universeIndexById = new Map(graph.universes.map((u, i) => [u.id, i]));

  return (
    <aside
      aria-label="Multiverse Architect Workspace"
      className={`absolute right-4 top-4 bottom-4 z-20 flex flex-col overflow-hidden rounded-xl border border-slate-700/80 bg-[#070c18]/96 text-slate-100 shadow-2xl transition-[width] duration-200 ${
        expanded ? 'w-[min(780px,calc(100%-2rem))]' : 'w-[min(450px,calc(100%-2rem))]'
      }`}
    >
      {/* Top Studio Header */}
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 bg-[#0a1020] px-3.5 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-fuchsia-500/40 bg-fuchsia-950/70 text-fuchsia-300">
            <GitBranch className="h-3.5 w-3.5" />
          </div>
          <div className="flex min-w-0 flex-col">
            <span className="truncate text-xs font-bold tracking-wide text-slate-100">
              Multiverse Architect
            </span>
            <span className="font-mono text-[10px] text-slate-400 tabular-nums">
              {graph.universes.length} {graph.universes.length === 1 ? 'Universe' : 'Universes'} ·{' '}
              {graph.wormholes.length} {graph.wormholes.length === 1 ? 'Wormhole' : 'Wormholes'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onToggleOverlay3d}
            title="Toggle 3D wireframe placement cages in viewport"
            className={`flex items-center gap-1 rounded border px-2 py-1 text-[10px] font-semibold transition ${
              showOverlay3d
                ? 'border-sky-400/80 bg-sky-950/80 text-sky-100'
                : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
            }`}
          >
            <Box className="h-3 w-3" />
            <span>3D Guides</span>
          </button>

          <button
            type="button"
            onClick={() => jsonInputRef.current?.click()}
            title="Import Multiverse JSON"
            className="flex items-center gap-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-[10px] font-semibold text-slate-300 transition hover:border-slate-500 hover:text-slate-100"
          >
            <Upload className="h-3 w-3" />
            <span className="hidden sm:inline">Import</span>
          </button>
          <input
            ref={jsonInputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleImportFile(file);
              e.target.value = '';
            }}
          />

          <button
            type="button"
            onClick={handleExportJson}
            title="Export Multiverse Topology as JSON"
            className="flex items-center gap-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-[10px] font-semibold text-slate-300 transition hover:border-slate-500 hover:text-slate-100"
          >
            <Download className="h-3 w-3" />
            <span className="hidden sm:inline">Export</span>
          </button>

          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            title={expanded ? 'Compact panel' : 'Expand panel'}
            className="hidden sm:flex h-6 w-6 items-center justify-center rounded border border-slate-700 bg-slate-900 text-slate-400 transition hover:border-slate-500 hover:text-slate-100"
          >
            {expanded ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
          </button>

          <button
            type="button"
            onClick={onClose}
            title="Close Multiverse Architect"
            className="flex h-6 w-6 items-center justify-center rounded border border-slate-700 bg-slate-900 text-slate-400 transition hover:border-slate-500 hover:text-slate-100"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Preset Bar & Import Feedback */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/90 bg-[#080d1a] px-3.5 py-2">
        <div className="flex items-center gap-1">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Templates
          </span>
          {MULTIVERSE_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => {
                setImportNotice(null);
                setLinkSourceId(null);
                onLoadPreset(p.id);
              }}
              className={`rounded border px-2 py-0.5 text-[10px] font-semibold transition ${
                activePreset === p.id
                  ? 'border-fuchsia-400 bg-fuchsia-950 text-fuchsia-100'
                  : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-fuchsia-500/50 hover:text-slate-100'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleAddUniverse}
            className="flex items-center gap-1 rounded border border-emerald-500/50 bg-emerald-950/60 px-2 py-0.5 text-[10px] font-semibold text-emerald-100 transition hover:bg-emerald-900/70"
          >
            <Plus className="h-3 w-3" />
            <span>Universe</span>
          </button>
          <button
            type="button"
            onClick={() => handleAddWormhole()}
            disabled={graph.universes.length === 0}
            className="flex items-center gap-1 rounded border border-fuchsia-500/50 bg-fuchsia-950/60 px-2 py-0.5 text-[10px] font-semibold text-fuchsia-100 transition hover:bg-fuchsia-900/70 disabled:opacity-40"
          >
            <Plus className="h-3 w-3" />
            <span>Wormhole</span>
          </button>
          <button
            type="button"
            onClick={() => {
              if (linkSourceId) {
                setLinkSourceId(null);
              } else {
                const startId =
                  selection?.kind === 'universe' ? selection.id : activeUniverseId;
                setLinkSourceId(startId || graph.universes[0]?.id || null);
              }
            }}
            disabled={graph.universes.length === 0}
            title="Click a second universe node (or the same node) in the graph to bridge a new wormhole"
            className={`flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-semibold transition disabled:opacity-40 ${
              linkSourceId
                ? 'border-amber-400 bg-amber-950 text-amber-100'
                : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
            }`}
          >
            <Link2 className="h-3 w-3" />
            <span>{linkSourceId ? 'Click Target…' : 'Connect'}</span>
          </button>
        </div>
      </div>

      {importNotice && (
        <div
          className={`flex items-center justify-between gap-2 border-b px-3.5 py-1.5 text-[10px] ${
            importNotice.kind === 'error'
              ? 'border-red-500/40 bg-red-950/80 text-red-100'
              : 'border-sky-500/30 bg-sky-950/60 text-sky-100'
          }`}
        >
          <span className="leading-snug">{importNotice.text}</span>
          <button
            type="button"
            onClick={() => setImportNotice(null)}
            className="shrink-0 text-slate-400 hover:text-slate-100"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {/* Overlap Validation Warnings */}
      {overlaps.length > 0 && (
        <div className="flex flex-col gap-1 border-b border-rose-500/40 bg-rose-950/75 px-3.5 py-2">
          <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-rose-200">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-300" />
            <span>
              Mouth Overlap Detected ({overlaps.length}) — Separate Mouths in Same Universe
            </span>
          </div>
          <div className="flex flex-wrap gap-1">
            {overlaps.slice(0, 4).map((w, idx) => (
              <button
                key={`${w.wormholeAId}-${w.endA}-${w.wormholeBId}-${w.endB}-${idx}`}
                type="button"
                onClick={() => {
                  onSelect({ kind: 'wormhole', id: w.wormholeAId, end: w.endA });
                  const uIdx = universeIndexById.get(w.universeId);
                  if (uIdx !== undefined) onTeleportToUniverse(uIdx);
                }}
                className="rounded border border-rose-400/50 bg-rose-950 px-2 py-0.5 text-[10px] text-rose-100 transition hover:bg-rose-900"
              >
                {w.universeName}: {w.wormholeAName} ({w.endA}) × {w.wormholeBName} ({w.endB})
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Content Area: Responsive Split when Expanded, Stacked when Compact */}
      <div
        className={`flex min-h-0 flex-1 ${
          expanded ? 'flex-col md:flex-row' : 'flex-col'
        }`}
      >
        {/* Interactive 2D Topology Graph Canvas */}
        <div
          className={`flex flex-col border-slate-800 ${
            expanded
              ? 'h-56 shrink-0 border-b md:h-auto md:w-[46%] md:border-b-0 md:border-r'
              : 'h-52 shrink-0 border-b'
          }`}
        >
          <div className="flex items-center justify-between bg-[#060a14] px-3 py-1.5 border-b border-slate-800/80">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              {linkSourceId
                ? `Select target universe to bridge from ${universeNameById.get(linkSourceId) ?? linkSourceId}`
                : 'Topology Graph — Drag Nodes · Click to Inspect'}
            </span>
            <button
              type="button"
              onClick={resetGraphLayout}
              className="text-[10px] text-slate-400 hover:text-slate-200 transition"
              title="Arrange universe nodes evenly"
            >
              Auto-layout
            </button>
          </div>

          <div className="relative flex-1 bg-[#050811]">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${GRAPH_VIEW_W} ${GRAPH_VIEW_H}`}
              className="h-full w-full select-none touch-none"
              onPointerMove={handleSvgPointerMove}
              onPointerUp={handleSvgPointerUp}
              onPointerCancel={handleSvgPointerUp}
            >
              <defs>
                <pattern
                  id="multiverse-grid-dots"
                  width="24"
                  height="24"
                  patternUnits="userSpaceOnUse"
                >
                  <circle cx="2" cy="2" r="0.9" fill="#1e293b" />
                </pattern>
              </defs>
              <rect width={GRAPH_VIEW_W} height={GRAPH_VIEW_H} fill="url(#multiverse-grid-dots)" />

              {/* Wormhole Edges */}
              {graph.wormholes.map((wh) => {
                const pA = nodePositions[wh.mouthA.universeId];
                const pB = nodePositions[wh.mouthB.universeId];
                if (!pA || !pB) return null;

                const isSelected = selection?.kind === 'wormhole' && selection.id === wh.id;
                const isOverlapping = overlappingWormholeIds.has(wh.id);
                const layout = edgeLayoutMeta.get(wh.id) ?? { index: 0, total: 1 };
                const shapeLabel = WORMHOLE_SHAPES[wh.shape]?.label ?? 'Sphere';

                const strokeColor = isOverlapping
                  ? '#fb7185'
                  : isSelected
                    ? '#f0abfc'
                    : '#64748b';

                if (wh.mouthA.universeId === wh.mouthB.universeId) {
                  // Teardrop self-loop curving outward from node center relative to canvas center
                  const vx = pA.x - GRAPH_VIEW_W / 2;
                  const vy = pA.y - GRAPH_VIEW_H / 2;
                  const vLen = Math.hypot(vx, vy);
                  const dirX = vLen > 5 ? vx / vLen : 0;
                  const dirY = vLen > 5 ? vy / vLen : -1;
                  const perpX = -dirY;
                  const perpY = dirX;
                  const loopReach = 54 + layout.index * 22;
                  const spread = 32 + layout.index * 10;

                  const c1x = pA.x + dirX * loopReach + perpX * spread;
                  const c1y = pA.y + dirY * loopReach + perpY * spread;
                  const c2x = pA.x + dirX * loopReach - perpX * spread;
                  const c2y = pA.y + dirY * loopReach - perpY * spread;
                  const d = `M ${pA.x} ${pA.y} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${pA.x} ${pA.y}`;
                  const labelX = pA.x + dirX * (loopReach * 0.78);
                  const labelY = pA.y + dirY * (loopReach * 0.78);

                  return (
                    <g
                      key={wh.id}
                      className="cursor-pointer"
                      onClick={(e) => {
                        e.stopPropagation();
                        setLinkSourceId(null);
                        onSelect({ kind: 'wormhole', id: wh.id, end: 'A' });
                      }}
                    >
                      <path d={d} fill="none" stroke="transparent" strokeWidth={14} />
                      <path
                        d={d}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth={isSelected || isOverlapping ? 2.6 : 1.8}
                        strokeDasharray="5 3"
                      />
                      <rect
                        x={labelX - 28}
                        y={labelY - 8}
                        width={56}
                        height={16}
                        rx={4}
                        fill={isSelected ? '#4a044e' : '#0f172a'}
                        stroke={strokeColor}
                        strokeWidth={1}
                      />
                      <text
                        x={labelX}
                        y={labelY + 3}
                        textAnchor="middle"
                        className="fill-slate-100 font-mono text-[8.5px] font-semibold"
                      >
                        {shapeLabel}
                      </text>
                    </g>
                  );
                }

                // Curved edge between two distinct universes
                const mx = (pA.x + pB.x) * 0.5;
                const my = (pA.y + pB.y) * 0.5;
                const dx = pB.x - pA.x;
                const dy = pB.y - pA.y;
                const len = Math.hypot(dx, dy) || 1;
                const nx = -dy / len;
                const ny = dx / len;
                const offsetIndex = layout.index - (layout.total - 1) / 2;
                const bend = offsetIndex * 34;
                const cx = mx + nx * bend;
                const cy = my + ny * bend;
                const d = `M ${pA.x} ${pA.y} Q ${cx} ${cy} ${pB.x} ${pB.y}`;

                // Quadratic Bezier midpoint at t = 0.5
                const lx = 0.25 * pA.x + 0.5 * cx + 0.25 * pB.x;
                const ly = 0.25 * pA.y + 0.5 * cy + 0.25 * pB.y;

                return (
                  <g
                    key={wh.id}
                    className="cursor-pointer"
                    onClick={(e) => {
                      e.stopPropagation();
                      setLinkSourceId(null);
                      onSelect({ kind: 'wormhole', id: wh.id, end: 'A' });
                    }}
                  >
                    <path d={d} fill="none" stroke="transparent" strokeWidth={14} />
                    <path
                      d={d}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={isSelected || isOverlapping ? 2.8 : 1.8}
                    />
                    <rect
                      x={lx - 28}
                      y={ly - 8}
                      width={56}
                      height={16}
                      rx={4}
                      fill={isSelected ? '#4a044e' : '#0f172a'}
                      stroke={strokeColor}
                      strokeWidth={1}
                    />
                    <text
                      x={lx}
                      y={ly + 3}
                      textAnchor="middle"
                      className="fill-slate-100 font-mono text-[8.5px] font-semibold"
                    >
                      {shapeLabel}
                    </text>
                  </g>
                );
              })}

              {/* Universe Nodes */}
              {graph.universes.map((u) => {
                const pos = nodePositions[u.id];
                if (!pos) return null;
                const isSelected = selection?.kind === 'universe' && selection.id === u.id;
                const isObserver = u.id === activeUniverseId;
                const isLinkOrigin = linkSourceId === u.id;
                const mouthCount = mouthCountsByUniverse.get(u.id) ?? 0;

                return (
                  <g
                    key={u.id}
                    transform={`translate(${pos.x}, ${pos.y})`}
                    className="cursor-grab active:cursor-grabbing"
                    onPointerDown={(e) => handleNodePointerDown(e, u.id)}
                  >
                    {/* Active observer outer ring */}
                    {isObserver && (
                      <circle
                        r={NODE_RADIUS + 5}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth={1.6}
                        strokeDasharray="4 2"
                      />
                    )}
                    {/* Link source highlight */}
                    {isLinkOrigin && (
                      <circle
                        r={NODE_RADIUS + 8}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth={2}
                      />
                    )}

                    {/* Node body */}
                    <circle
                      r={NODE_RADIUS}
                      fill="#0b1222"
                      stroke={isSelected ? '#f0abfc' : isObserver ? '#38bdf8' : '#475569'}
                      strokeWidth={isSelected ? 2.4 : 1.5}
                    />

                    {/* Sky dual-color swatch inside node */}
                    {u.skyMode === 'texture' && u.textureLayer >= 0 ? (
                      <circle
                        r={10}
                        cy={-4}
                        fill={u.isHdr ? '#f59e0b' : '#38bdf8'}
                        opacity={0.85}
                      />
                    ) : (
                      <g transform="translate(0, -4)">
                        <path d="M -8 0 A 8 8 0 0 1 8 0 Z" fill={u.gridVerticalColor} />
                        <path d="M -8 0 A 8 8 0 0 0 8 0 Z" fill={u.gridHorizontalColor} />
                        <circle r={8} fill="none" stroke="#e2e8f0" strokeOpacity={0.3} strokeWidth={0.8} />
                      </g>
                    )}

                    {/* Mouth count mini badge */}
                    <text
                      y={14}
                      textAnchor="middle"
                      className="fill-slate-300 font-mono text-[8.5px] font-semibold"
                    >
                      {mouthCount}m
                    </text>

                    {/* Universe name underneath */}
                    <text
                      y={NODE_RADIUS + 13}
                      textAnchor="middle"
                      className="fill-slate-100 text-[10px] font-semibold"
                    >
                      {u.name.length > 14 ? `${u.name.slice(0, 13)}…` : u.name}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Context-Sensitive Inspector Pane */}
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3.5">
          {/* UNIVERSE INSPECTOR */}
          {selectedUniverse && (
            <>
              <div className="flex items-start justify-between gap-2 border-b border-slate-800 pb-2.5">
                <div className="flex flex-1 flex-col gap-1">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-sky-300">
                    Universe Inspector
                  </span>
                  <input
                    type="text"
                    value={selectedUniverse.name}
                    maxLength={48}
                    aria-label="Universe name"
                    onChange={(e) => updateUniverse(selectedUniverse.id, { name: e.target.value })}
                    className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-semibold text-slate-100 outline-none focus:border-fuchsia-400"
                  />
                </div>
                <div className="flex items-center gap-1 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      const idx = universeIndexById.get(selectedUniverse.id);
                      if (idx !== undefined) onTeleportToUniverse(idx);
                    }}
                    className="flex items-center gap-1 rounded border border-sky-500/50 bg-sky-950/70 px-2 py-1 text-[10px] font-semibold text-sky-100 transition hover:bg-sky-900"
                    title="Move camera observer into this universe"
                  >
                    <Compass className="h-3 w-3" />
                    <span>Visit</span>
                  </button>
                  <button
                    type="button"
                    disabled={graph.universes.length <= 1}
                    onClick={() => handleDeleteUniverse(selectedUniverse.id)}
                    className="flex items-center justify-center rounded border border-red-500/40 bg-red-950/50 p-1.5 text-red-200 transition hover:bg-red-900/70 disabled:cursor-not-allowed disabled:opacity-35"
                    title={
                      graph.universes.length <= 1
                        ? 'At least one universe is required'
                        : 'Delete this universe and its connected wormholes'
                    }
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              </div>

              {/* Sky Source Selector */}
              <div className="flex flex-col gap-2 rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-300">
                    Sky Environment
                  </span>
                  {selectedUniverse.isHdr && selectedUniverse.textureLayer >= 0 && (
                    <span className="rounded border border-amber-400/50 bg-amber-950 px-1.5 py-0.5 font-mono text-[9px] font-bold text-amber-200">
                      HDRI
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                  {(['grid', 'texture'] as UniverseSkyMode[]).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => {
                        updateUniverse(selectedUniverse.id, { skyMode: mode });
                        if (
                          mode === 'texture' &&
                          !selectedUniverseMeta.fileLabel &&
                          selectedUniverse.textureLayer < 0 &&
                          !selectedUniverseMeta.isLoading
                        ) {
                          skyInputRef.current?.click();
                        }
                      }}
                      className={`rounded border py-1 text-[10px] font-semibold transition ${
                        selectedUniverse.skyMode === mode
                          ? 'border-fuchsia-400 bg-fuchsia-950 text-fuchsia-100'
                          : 'border-slate-700 bg-slate-900 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {mode === 'grid' ? 'Grid Pattern' : 'Image / HDRI'}
                    </button>
                  ))}
                </div>

                {/* Upload row (always accessible so switching to texture is 1 click) */}
                <div className="flex items-center gap-1.5 pt-0.5">
                  <button
                    type="button"
                    disabled={selectedUniverseMeta.isLoading}
                    onClick={() => skyInputRef.current?.click()}
                    className="whitespace-nowrap rounded border border-fuchsia-500/50 bg-fuchsia-950/60 px-2 py-1 text-[10px] font-semibold text-fuchsia-100 transition hover:bg-fuchsia-900/70 disabled:opacity-50"
                  >
                    {selectedUniverseMeta.isLoading ? 'Loading…' : 'Upload .hdr / .exr / Image'}
                  </button>
                  <input
                    ref={skyInputRef}
                    type="file"
                    accept="image/*,.hdr,.exr,.pic,.rgbe"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) onUploadUniverseSky(selectedUniverse.id, file);
                      e.target.value = '';
                    }}
                  />
                  <span
                    className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-400"
                    title={selectedUniverseMeta.fileLabel ?? undefined}
                  >
                    {selectedUniverseMeta.isLoading
                      ? 'Decoding sky map…'
                      : (selectedUniverseMeta.fileLabel ?? 'No file loaded')}
                  </span>
                  {(selectedUniverseMeta.fileLabel || selectedUniverse.textureLayer >= 0) &&
                    !selectedUniverseMeta.isLoading && (
                      <button
                        type="button"
                        onClick={() => onClearUniverseSky(selectedUniverse.id)}
                        className="text-[10px] text-slate-400 hover:text-slate-100"
                      >
                        Clear
                      </button>
                    )}
                </div>

                {selectedUniverseMeta.error && (
                  <p className="rounded border border-red-500/40 bg-red-950/60 px-2 py-1 text-[10px] text-red-200">
                    {selectedUniverseMeta.error}
                  </p>
                )}

                {selectedUniverse.skyMode === 'texture' &&
                  selectedUniverse.isHdr &&
                  selectedUniverse.textureLayer >= 0 && (
                    <div className="rounded border border-amber-500/30 bg-amber-950/25 p-2">
                      <NumericSlider
                        label="HDRI Exposure (ACES Filmic)"
                        min={-4}
                        max={4}
                        step={0.1}
                        value={selectedUniverse.exposureEv}
                        display={`${selectedUniverse.exposureEv >= 0 ? '+' : ''}${selectedUniverse.exposureEv.toFixed(1)} EV`}
                        onChange={(v) => updateUniverse(selectedUniverse.id, { exposureEv: v })}
                        accent="amber"
                      />
                    </div>
                  )}

                {selectedUniverse.skyMode === 'grid' && (
                  <div className="flex flex-col gap-2 border-t border-slate-800/90 pt-2">
                    <div className="grid grid-cols-2 gap-1.5">
                      <label className="flex cursor-pointer items-center justify-between rounded border border-slate-800 bg-slate-900 px-2 py-1">
                        <span className="flex items-center gap-1.5 text-[10px] text-slate-300">
                          <span
                            className="h-3.5 w-3.5 rounded border border-white/30"
                            style={{ backgroundColor: selectedUniverse.gridVerticalColor }}
                          />
                          Longitude
                        </span>
                        <input
                          type="color"
                          value={selectedUniverse.gridVerticalColor}
                          onChange={(e) =>
                            updateUniverse(selectedUniverse.id, {
                              gridVerticalColor: e.target.value,
                            })
                          }
                          className="h-0 w-0 opacity-0"
                        />
                        <span className="font-mono text-[9px] uppercase text-slate-400">
                          {selectedUniverse.gridVerticalColor}
                        </span>
                      </label>

                      <label className="flex cursor-pointer items-center justify-between rounded border border-slate-800 bg-slate-900 px-2 py-1">
                        <span className="flex items-center gap-1.5 text-[10px] text-slate-300">
                          <span
                            className="h-3.5 w-3.5 rounded border border-white/30"
                            style={{ backgroundColor: selectedUniverse.gridHorizontalColor }}
                          />
                          Latitude
                        </span>
                        <input
                          type="color"
                          value={selectedUniverse.gridHorizontalColor}
                          onChange={(e) =>
                            updateUniverse(selectedUniverse.id, {
                              gridHorizontalColor: e.target.value,
                            })
                          }
                          className="h-0 w-0 opacity-0"
                        />
                        <span className="font-mono text-[9px] uppercase text-slate-400">
                          {selectedUniverse.gridHorizontalColor}
                        </span>
                      </label>
                    </div>

                    <div className="flex flex-wrap items-center gap-1">
                      <span className="text-[9px] text-slate-500">Palettes:</span>
                      {GRID_PALETTES.map((pal) => (
                        <button
                          key={pal.name}
                          type="button"
                          title={pal.name}
                          onClick={() =>
                            updateUniverse(selectedUniverse.id, {
                              gridVerticalColor: pal.v,
                              gridHorizontalColor: pal.h,
                            })
                          }
                          className="flex h-3.5 w-6 overflow-hidden rounded border border-slate-700 hover:scale-110 transition"
                        >
                          <span className="h-full w-1/2" style={{ backgroundColor: pal.v }} />
                          <span className="h-full w-1/2" style={{ backgroundColor: pal.h }} />
                        </button>
                      ))}
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <NumericSlider
                        label="Longitude Lines"
                        min={4}
                        max={72}
                        step={2}
                        value={selectedUniverse.gridCols}
                        display={String(selectedUniverse.gridCols)}
                        onChange={(v) => updateUniverse(selectedUniverse.id, { gridCols: Math.round(v) })}
                      />
                      <NumericSlider
                        label="Latitude Lines"
                        min={2}
                        max={36}
                        step={1}
                        value={selectedUniverse.gridRows}
                        display={String(selectedUniverse.gridRows)}
                        onChange={(v) => updateUniverse(selectedUniverse.id, { gridRows: Math.round(v) })}
                      />
                    </div>

                    <GridLineOverridesEditor
                      cols={selectedUniverse.gridCols}
                      rows={selectedUniverse.gridRows}
                      lonOverrides={selectedUniverse.gridLonOverrides}
                      latOverrides={selectedUniverse.gridLatOverrides}
                      onChangeLonOverrides={(next) =>
                        updateUniverse(selectedUniverse.id, { gridLonOverrides: next })
                      }
                      onChangeLatOverrides={(next) =>
                        updateUniverse(selectedUniverse.id, { gridLatOverrides: next })
                      }
                    />
                  </div>
                )}

                {/* Sky Rotation */}
                <div className="flex flex-col gap-1.5 border-t border-slate-800/90 pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-slate-400">Sky Orientation</span>
                    <button
                      type="button"
                      onClick={() =>
                        updateUniverse(selectedUniverse.id, { rotationY: 0, rotationX: 0 })
                      }
                      className="text-[9px] font-mono text-fuchsia-300 hover:text-fuchsia-100"
                    >
                      Reset 0°
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <NumericSlider
                      label="Sky Yaw"
                      min={-180}
                      max={180}
                      step={1}
                      value={selectedUniverse.rotationY}
                      display={`${selectedUniverse.rotationY > 0 ? '+' : ''}${Math.round(selectedUniverse.rotationY)}°`}
                      onChange={(v) => updateUniverse(selectedUniverse.id, { rotationY: v })}
                    />
                    <NumericSlider
                      label="Sky Pitch"
                      min={-180}
                      max={180}
                      step={1}
                      value={selectedUniverse.rotationX}
                      display={`${selectedUniverse.rotationX > 0 ? '+' : ''}${Math.round(selectedUniverse.rotationX)}°`}
                      onChange={(v) => updateUniverse(selectedUniverse.id, { rotationX: v })}
                    />
                  </div>
                </div>
              </div>

              {/* Wormhole Mouths inside this Universe */}
              <div className="flex flex-col gap-1.5 rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-300">
                    Mouths in {selectedUniverse.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleAddWormhole(selectedUniverse.id)}
                    className="flex items-center gap-1 rounded border border-fuchsia-500/40 bg-fuchsia-950/60 px-1.5 py-0.5 text-[9px] font-semibold text-fuchsia-200 hover:bg-fuchsia-900"
                  >
                    <Plus className="h-2.5 w-2.5" />
                    <span>Add Bridge</span>
                  </button>
                </div>

                {(() => {
                  const localMouths: Array<{ wh: WormholeEdge; end: 'A' | 'B'; mouth: WormholeMouthPlacement; destId: string }> = [];
                  for (const wh of graph.wormholes) {
                    if (wh.mouthA.universeId === selectedUniverse.id) {
                      localMouths.push({ wh, end: 'A', mouth: wh.mouthA, destId: wh.mouthB.universeId });
                    }
                    if (wh.mouthB.universeId === selectedUniverse.id) {
                      localMouths.push({ wh, end: 'B', mouth: wh.mouthB, destId: wh.mouthA.universeId });
                    }
                  }
                  if (localMouths.length === 0) {
                    return (
                      <p className="text-[10px] text-slate-500">
                        No wormhole mouths in this universe yet. Click Add Bridge above to connect it.
                      </p>
                    );
                  }
                  return (
                    <div className="flex flex-col gap-1">
                      {localMouths.map((entry) => (
                        <button
                          key={`${entry.wh.id}-${entry.end}`}
                          type="button"
                          onClick={() =>
                            onSelect({ kind: 'wormhole', id: entry.wh.id, end: entry.end })
                          }
                          className="flex items-center justify-between rounded border border-slate-800 bg-slate-900/90 px-2 py-1.5 text-left transition hover:border-fuchsia-500/60"
                        >
                          <div className="flex min-w-0 flex-col">
                            <span className="truncate text-[11px] font-semibold text-slate-100">
                              {entry.wh.name} · Mouth {entry.end}
                            </span>
                            <span className="font-mono text-[9px] text-slate-400 tabular-nums">
                              {WORMHOLE_SHAPES[entry.wh.shape]?.label} →{' '}
                              {universeNameById.get(entry.destId) ?? entry.destId} · (
                              {entry.mouth.position[0].toFixed(1)}, {entry.mouth.position[1].toFixed(1)},{' '}
                              {entry.mouth.position[2].toFixed(1)})
                            </span>
                          </div>
                          <span className="ml-2 shrink-0 text-[10px] font-semibold text-fuchsia-300">
                            Edit
                          </span>
                        </button>
                      ))}
                    </div>
                  );
                })()}
              </div>
            </>
          )}

          {/* WORMHOLE INSPECTOR */}
          {selectedWormhole && activeMouth && partnerMouth && (
            <>
              <div className="flex items-start justify-between gap-2 border-b border-slate-800 pb-2.5">
                <div className="flex flex-1 flex-col gap-1">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-300">
                    Wormhole Inspector
                  </span>
                  <input
                    type="text"
                    value={selectedWormhole.name}
                    maxLength={48}
                    aria-label="Wormhole name"
                    onChange={(e) =>
                      updateWormhole(selectedWormhole.id, (wh) => ({ ...wh, name: e.target.value }))
                    }
                    className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-semibold text-slate-100 outline-none focus:border-fuchsia-400"
                  />
                </div>
                <div className="flex items-center gap-1 pt-4">
                  <button
                    type="button"
                    onClick={() => handleDeleteWormhole(selectedWormhole.id)}
                    className="flex items-center gap-1 rounded border border-red-500/40 bg-red-950/60 px-2 py-1 text-[10px] font-semibold text-red-200 transition hover:bg-red-900/70"
                    title="Delete this wormhole bridge"
                  >
                    <Trash2 className="h-3 w-3" />
                    <span>Delete</span>
                  </button>
                </div>
              </div>

              {/* Shared Throat Geometry (Enforced Identical Across Bridged Universes) */}
              <div className="flex flex-col gap-2 rounded-lg border border-fuchsia-500/25 bg-slate-950/80 p-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-200">
                    Shared Throat Geometry
                  </span>
                  <span className="font-mono text-[9px] text-slate-400">Matched on A & B</span>
                </div>

                <div className="grid grid-cols-3 gap-1">
                  {WORMHOLE_SHAPES.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      title={opt.title}
                      onClick={() =>
                        updateWormhole(selectedWormhole.id, (wh) => ({
                          ...wh,
                          shape: opt.id as WormholeShapeId,
                        }))
                      }
                      className={`rounded border px-1.5 py-1 text-[10px] font-semibold transition ${
                        selectedWormhole.shape === opt.id
                          ? 'border-fuchsia-400 bg-fuchsia-950 text-fuchsia-100'
                          : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-fuchsia-500/40'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-2.5 pt-1">
                  <NumericSlider
                    label="Throat Radius"
                    min={0.15}
                    max={4.5}
                    step={0.05}
                    value={selectedWormhole.radius}
                    display={selectedWormhole.radius.toFixed(2)}
                    onChange={(v) =>
                      updateWormhole(selectedWormhole.id, (wh) => ({ ...wh, radius: v }))
                    }
                  />
                  <NumericSlider
                    label="Gravity Ratio"
                    min={0}
                    max={2}
                    step={0.02}
                    value={selectedWormhole.gravityRatio}
                    display={selectedWormhole.gravityRatio.toFixed(2)}
                    onChange={(v) =>
                      updateWormhole(selectedWormhole.id, (wh) => ({ ...wh, gravityRatio: v }))
                    }
                  />
                </div>

                <div className="flex flex-col gap-1.5 border-t border-slate-800/80 pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-slate-400">
                      Shape Scale (X / Y / Z)
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        updateWormhole(selectedWormhole.id, (wh) => ({
                          ...wh,
                          scale: [...DEFAULT_SHAPE_SCALE],
                        }))
                      }
                      className="font-mono text-[9px] text-fuchsia-300 hover:text-fuchsia-100"
                    >
                      Reset 1:1:1
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {(['Width X', 'Height Y', 'Length Z'] as const).map((label, axis) => (
                      <NumericSlider
                        key={label}
                        label={label}
                        min={0.25}
                        max={3}
                        step={0.05}
                        value={selectedWormhole.scale[axis]}
                        display={`${selectedWormhole.scale[axis].toFixed(2)}×`}
                        onChange={(v) =>
                          updateWormhole(selectedWormhole.id, (wh) => {
                            const next: [number, number, number] = [
                              wh.scale[0],
                              wh.scale[1],
                              wh.scale[2],
                            ];
                            next[axis] = v;
                            return { ...wh, scale: next };
                          })
                        }
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Mouth A / Mouth B Placement Tabs */}
              <div className="flex flex-col gap-2.5 rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="grid grid-cols-2 gap-1.5">
                  {(['A', 'B'] as const).map((end) => {
                    const m = end === 'A' ? selectedWormhole.mouthA : selectedWormhole.mouthB;
                    const uLabel = universeNameById.get(m.universeId) ?? m.universeId;
                    return (
                      <button
                        key={end}
                        type="button"
                        onClick={() =>
                          onSelect({ kind: 'wormhole', id: selectedWormhole.id, end })
                        }
                        className={`flex flex-col items-start rounded border px-2 py-1.5 text-left transition ${
                          selectedEnd === end
                            ? 'border-sky-400 bg-sky-950/70 text-sky-100'
                            : 'border-slate-700 bg-slate-900 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <span className="text-[10px] font-bold uppercase tracking-wider">
                          Mouth {end}
                        </span>
                        <span className="w-full truncate font-mono text-[9px] text-slate-300">
                          in {uLabel}
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Target Universe Dropdown + Quick Camera Frame */}
                <div className="flex items-end gap-1.5">
                  <label className="flex flex-1 flex-col gap-1">
                    <span className="text-[10px] font-semibold text-slate-400">
                      Residing Universe (Mouth {selectedEnd})
                    </span>
                    <select
                      value={activeMouth.universeId}
                      onChange={(e) =>
                        updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                          universeId: e.target.value,
                        })
                      }
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-medium text-slate-100 outline-none focus:border-sky-400"
                    >
                      {graph.universes.map((u) => (
                        <option key={u.id} value={u.id}>
                          {u.name}
                          {u.id === partnerMouth.universeId ? ' (Same-universe shortcut)' : ''}
                        </option>
                      ))}
                    </select>
                  </label>

                  <button
                    type="button"
                    onClick={() => {
                      const uIdx = universeIndexById.get(activeMouth.universeId);
                      if (uIdx !== undefined) {
                        onFrameMouth(uIdx, activeMouth.position, selectedWormhole.radius);
                      }
                    }}
                    className="flex items-center gap-1 rounded border border-sky-500/50 bg-sky-950/70 px-2.5 py-1.5 text-[10px] font-semibold text-sky-100 transition hover:bg-sky-900"
                    title="Jump observer to this universe and frame this mouth"
                  >
                    <Compass className="h-3 w-3" />
                    <span>Frame</span>
                  </button>
                </div>

                {/* Position X / Y / Z */}
                <div className="flex flex-col gap-1.5 border-t border-slate-800/80 pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-slate-400">
                      Position (Mouth {selectedEnd})
                    </span>
                    {activeMouth.universeId !== partnerMouth.universeId && (
                      <button
                        type="button"
                        onClick={() =>
                          updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                            position: [
                              partnerMouth.position[0],
                              partnerMouth.position[1],
                              partnerMouth.position[2],
                            ],
                            quaternion: [
                              partnerMouth.quaternion[0],
                              partnerMouth.quaternion[1],
                              partnerMouth.quaternion[2],
                              partnerMouth.quaternion[3],
                            ],
                          })
                        }
                        className="font-mono text-[9px] text-sky-300 hover:text-sky-100"
                        title="Copy XYZ position and orientation from the partner mouth"
                      >
                        Match Mouth {selectedEnd === 'A' ? 'B' : 'A'}
                      </button>
                    )}
                  </div>

                  {(['Pos X', 'Pos Y', 'Pos Z'] as const).map((label, axis) => (
                    <NumericSlider
                      key={label}
                      label={label}
                      min={-10}
                      max={10}
                      step={0.05}
                      value={activeMouth.position[axis]}
                      display={activeMouth.position[axis].toFixed(2)}
                      accent="sky"
                      onChange={(v) => {
                        const nextPos: [number, number, number] = [
                          activeMouth.position[0],
                          activeMouth.position[1],
                          activeMouth.position[2],
                        ];
                        nextPos[axis] = v;
                        updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                          position: nextPos,
                        });
                      }}
                    />
                  ))}
                </div>

                {/* Orientation Yaw / Pitch / Roll */}
                <div className="flex flex-col gap-1.5 border-t border-slate-800/80 pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-slate-400">
                      Orientation (Mouth {selectedEnd})
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                          quaternion: [0, 0, 0, 1],
                        })
                      }
                      className="flex items-center gap-1 font-mono text-[9px] text-sky-300 hover:text-sky-100"
                    >
                      <RotateCcw className="h-2.5 w-2.5" />
                      <span>Reset 0°</span>
                    </button>
                  </div>

                  <NumericSlider
                    label="Yaw (Y)"
                    min={-180}
                    max={180}
                    step={1}
                    value={Math.round(activeMouthEuler[0])}
                    display={`${Math.round(activeMouthEuler[0])}°`}
                    accent="sky"
                    onChange={(yaw) =>
                      updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                        quaternion: quaternionFromEulerDeg(
                          yaw,
                          activeMouthEuler[1],
                          activeMouthEuler[2]
                        ),
                      })
                    }
                  />
                  <NumericSlider
                    label="Pitch (X)"
                    min={-89}
                    max={89}
                    step={1}
                    value={Math.round(activeMouthEuler[1])}
                    display={`${Math.round(activeMouthEuler[1])}°`}
                    accent="sky"
                    onChange={(pitch) =>
                      updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                        quaternion: quaternionFromEulerDeg(
                          activeMouthEuler[0],
                          pitch,
                          activeMouthEuler[2]
                        ),
                      })
                    }
                  />
                  <NumericSlider
                    label="Roll (Z)"
                    min={-180}
                    max={180}
                    step={1}
                    value={Math.round(activeMouthEuler[2])}
                    display={`${Math.round(activeMouthEuler[2])}°`}
                    accent="sky"
                    onChange={(roll) =>
                      updateSelectedMouth(selectedWormhole.id, selectedEnd, {
                        quaternion: quaternionFromEulerDeg(
                          activeMouthEuler[0],
                          activeMouthEuler[1],
                          roll
                        ),
                      })
                    }
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </aside>
  );
};
