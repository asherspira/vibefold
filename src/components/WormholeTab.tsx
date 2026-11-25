import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Box,
  Clapperboard,
  Compass,
  Crosshair,
  Download,
  Eye,
  EyeOff,
  GitBranch,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Repeat,
  Square,
  Trash2,
} from 'lucide-react';
import { WORMHOLE_FRAGMENT_SHADER, WORMHOLE_VERTEX_SHADER } from './wormholeShader';
import {
  MultiverseWorkspaceDrawer,
  type UniverseSkyMeta,
  type WorkspaceSelection,
} from './MultiverseWorkspaceDrawer';
import { GridLineOverridesEditor } from './GridLineOverridesEditor';
import { WORMHOLE_SHAPES, type WormholeShapeId } from '../utils/wormholeSdf';
import {
  buildMultiverseTimeline,
  computeRenderDimensions,
  generateMultiverseTourKeyframes,
  INTERPOLATION_OPTIONS,
  resolveKeyframeUniverseIndex,
  sampleMultiverseTimeline,
  sampleTimelinePathInUniverse,
  type CameraKeyframe,
  type InterpolationMode,
  type MovieFormat,
  type MultiverseTimeline,
} from '../utils/wormholeAnimation';
import {
  downloadVideoBlob,
  renderWormholeMovie,
  type RenderedMovieResult,
} from '../utils/wormholeMovieRenderer';
import { detectSkyFileKind, loadHdriTexture } from '../utils/wormholeHdri';
import {
  createPresetMultiverseGraph,
  crossMouthWorld,
  detectMouthOverlaps,
  findNearestMouthInUniverse,
  MULTIVERSE_PRESETS,
  packMultiverseToFloatBuffers,
  type GridLineColorOverrides,
  type MultiverseGraph,
  type MultiversePresetId,
  type SerializedCameraKeyframe,
  type UniverseNode,
  type UniverseSkyMode,
} from '../utils/wormholeMultiverse';
import {
  packDataTextureToHalfFloatLayer,
  packImageToHalfFloatLayer,
  SkyArrayManager,
} from '../utils/wormholeSkyArray';
import {
  disposeOverlayGroup,
  syncWormholeOverlayGroup,
} from '../utils/wormholeOverlay3d';

const DEFAULT_POSITION: [number, number, number] = [0, 0, 3];
const DEFAULT_RADIUS = 1;
const DEFAULT_SHAPE: WormholeShapeId = 0;
const DEFAULT_SHAPE_SCALE: [number, number, number] = [1, 1, 1];
const DEFAULT_GRAVITY = 0.5;
const DEFAULT_STEPS = 80;
const DEFAULT_MAX_CROSSINGS = 8;
const DEFAULT_QUALITY = 0.85;

const ACCENT = {
  track: 'accent-fuchsia-400',
  text: 'text-fuchsia-200',
  border: 'border-fuchsia-500/40',
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function detectSoftwareGl(renderer: THREE.WebGLRenderer): boolean {
  try {
    const gl = renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) ?? '');
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return false;
  }
}

function createFloatTableTexture(
  data: Float32Array,
  width: number,
  height: number
): THREE.DataTexture {
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.colorSpace = THREE.NoColorSpace;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

function createByteTableTexture(
  data: Uint8Array,
  width: number,
  height: number
): THREE.DataTexture {
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.NoColorSpace;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

interface GridConfig {
  verticalColor: string;
  horizontalColor: string;
  cols: number;
  rows: number;
  rotationY: number;
  rotationX: number;
}

const DEFAULT_GRID_A: GridConfig = {
  verticalColor: '#ff0000',
  horizontalColor: '#0000ff',
  cols: 18,
  rows: 9,
  rotationY: 0,
  rotationX: 0,
};

const DEFAULT_GRID_B: GridConfig = {
  verticalColor: '#00ffff',
  horizontalColor: '#ff00ff',
  cols: 18,
  rows: 9,
  rotationY: 0,
  rotationX: 0,
};

function createInitialGraph(
  preset: MultiversePresetId = 'classic-binary',
  preserve?: MultiverseGraph
): MultiverseGraph {
  const prevA = preserve?.universes.find((u) => u.id === 'A');
  const prevB = preserve?.universes.find((u) => u.id === 'B');
  const prevWh = preserve?.wormholes[0];

  return createPresetMultiverseGraph(preset, {
    position: DEFAULT_POSITION,
    radius: DEFAULT_RADIUS,
    shape: DEFAULT_SHAPE,
    scale: DEFAULT_SHAPE_SCALE,
    gravityRatio: prevWh ? prevWh.gravityRatio : DEFAULT_GRAVITY,
    universeA: {
      mode: prevA ? prevA.skyMode : 'grid',
      grid: prevA
        ? {
            verticalColor: prevA.gridVerticalColor,
            horizontalColor: prevA.gridHorizontalColor,
            cols: prevA.gridCols,
            rows: prevA.gridRows,
            rotationY: prevA.rotationY,
            rotationX: prevA.rotationX,
            lonOverrides: prevA.gridLonOverrides,
            latOverrides: prevA.gridLatOverrides,
          }
        : DEFAULT_GRID_A,
      isHdr: prevA ? prevA.isHdr : false,
      exposureEv: prevA ? prevA.exposureEv : 0,
      textureLayer: prevA ? prevA.textureLayer : -1,
    },
    universeB: {
      mode: prevB ? prevB.skyMode : 'grid',
      grid: prevB
        ? {
            verticalColor: prevB.gridVerticalColor,
            horizontalColor: prevB.gridHorizontalColor,
            cols: prevB.gridCols,
            rows: prevB.gridRows,
            rotationY: prevB.rotationY,
            rotationX: prevB.rotationX,
            lonOverrides: prevB.gridLonOverrides,
            latOverrides: prevB.gridLatOverrides,
          }
        : DEFAULT_GRID_B,
      isHdr: prevB ? prevB.isHdr : false,
      exposureEv: prevB ? prevB.exposureEv : 0,
      textureLayer: prevB ? prevB.textureLayer : -1,
    },
  });
}

interface SliderRowProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  display: string;
  onChange: (value: number) => void;
  presets?: { label: string; value: number }[];
}

const SliderRow: React.FC<SliderRowProps> = ({
  label,
  min,
  max,
  step,
  value,
  display,
  onChange,
  presets,
}) => (
  <div className="flex flex-col gap-1">
    <div className="flex items-center justify-between text-[11px]">
      <span className="font-semibold text-slate-200">{label}</span>
      <span className="font-mono text-fuchsia-200 tabular-nums">{display}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={label}
      onChange={(event) => onChange(parseFloat(event.target.value))}
      className={`w-full cursor-pointer ${ACCENT.track}`}
    />
    {presets && (
      <div className="flex flex-wrap gap-1">
        {presets.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => onChange(preset.value)}
            className={`rounded border px-1.5 py-0.5 font-mono text-[10px] transition ${
              Math.abs(value - preset.value) < step / 2
                ? 'border-fuchsia-400/80 bg-fuchsia-950/80 text-fuchsia-100'
                : 'border-slate-700 bg-slate-950/70 text-slate-400 hover:text-slate-200'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>
    )}
  </div>
);

interface WormholeTabProps {
  uiVisible?: boolean;
  onToggleUi?: () => void;
}

export const WormholeTab: React.FC<WormholeTabProps> = ({
  uiVisible: uiVisibleProp,
  onToggleUi: onToggleUiProp,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<() => void>(() => {});
  const frameMouthRef = useRef<
    (universeIndex: number, position: [number, number, number], radius: number) => void
  >(() => {});
  const enterThroatRef = useRef<() => void>(() => {});
  const centerDirectRef = useRef<() => void>(() => {});
  const applySizeRef = useRef<() => void>(() => {});
  const uploadUniverseSkyRef = useRef<(universeId: string, file: File) => void>(() => {});
  const clearUniverseSkyRef = useRef<(universeId: string) => void>(() => {});
  const clearAllSkiesRef = useRef<(keepIds?: Set<string>) => void>(() => {});
  const captureKeyframeRef = useRef<() => CameraKeyframe | null>(() => null);
  const applyKeyframeRef = useRef<(keyframe: CameraKeyframe) => void>(() => {});
  const renderMovieRef = useRef<
    | ((params: {
        timeline: MultiverseTimeline;
        durationSeconds: number;
        format: MovieFormat;
        onProgress: (frame: number, total: number, width: number, height: number) => void;
        signal: AbortSignal;
      }) => Promise<RenderedMovieResult>)
    | null
  >(null);
  const renderAbortRef = useRef<AbortController | null>(null);

  const [internalUiVisible, setInternalUiVisible] = useState(true);
  const isUiVisible = uiVisibleProp !== undefined ? uiVisibleProp : internalUiVisible;
  const toggleUi = () => {
    if (onToggleUiProp) {
      onToggleUiProp();
    } else {
      setInternalUiVisible((prev) => !prev);
    }
  };
  const toggleUiRef = useRef(toggleUi);
  toggleUiRef.current = toggleUi;

  // Single source of truth for the entire multiverse topology
  const [graph, setGraph] = useState<MultiverseGraph>(() => createInitialGraph('classic-binary'));
  const [activePreset, setActivePreset] = useState<MultiversePresetId | 'custom'>('classic-binary');
  const [skyMetaById, setSkyMetaById] = useState<Record<string, UniverseSkyMeta>>({});

  // Stage 3 Multiverse Workspace & 3D Wireframe Overlay state
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspaceSelection, setWorkspaceSelection] = useState<WorkspaceSelection>({
    kind: 'universe',
    id: 'A',
  });
  const [showOverlay3d, setShowOverlay3d] = useState(false);

  // Global raymarcher & observer state
  const [maxSteps, setMaxSteps] = useState(DEFAULT_STEPS);
  const [maxCrossings, setMaxCrossings] = useState(DEFAULT_MAX_CROSSINGS);
  const [diagnosticMode, setDiagnosticMode] = useState<0 | 1 | 2>(0);
  const [quality, setQuality] = useState(DEFAULT_QUALITY);
  const [room, setRoom] = useState<number>(0);
  const [hud, setHud] = useState({ x: 0, y: 0, z: -0.15, distance: 3.15, gap: 2.15 });
  const [shaderError, setShaderError] = useState<string | null>(null);

  // Stage 4: Multi-keyframe multiverse camera choreography & offline movie rendering state
  const [keyframes, setKeyframes] = useState<CameraKeyframe[]>([]);
  const [interpMode, setInterpMode] = useState<InterpolationMode>('smoothstep');
  const [durationSec, setDurationSec] = useState<number>(6);
  const [durationInput, setDurationInput] = useState<string>('6');
  const [loopAnim, setLoopAnim] = useState<boolean>(false);
  const [movieFormat, setMovieFormat] = useState<MovieFormat>('mp4');
  const [isPlayingAnim, setIsPlayingAnim] = useState(false);
  const [animProgress, setAnimProgress] = useState(0);
  const [renderStatus, setRenderStatus] = useState<{
    active: boolean;
    frame: number;
    total: number;
    width: number;
    height: number;
    format: MovieFormat;
  } | null>(null);
  const [renderNotice, setRenderNotice] = useState<{
    kind: 'success' | 'error';
    text: string;
  } | null>(null);

  // Live overlap validation across all universes
  const overlaps = useMemo(() => detectMouthOverlaps(graph), [graph]);

  // Live compiled Stage 4 multiverse camera timeline
  const timeline = useMemo(
    () => buildMultiverseTimeline(keyframes, graph, interpMode),
    [keyframes, graph, interpMode]
  );

  // Keep active universe index clamped when universes are deleted
  useEffect(() => {
    if (graph.universes.length > 0 && room >= graph.universes.length) {
      const clamped = graph.universes.length - 1;
      roomRef.current = clamped;
      setRoom(clamped);
    }
  }, [graph.universes.length, room]);

  const graphRef = useRef(graph);
  const overlapsRef = useRef(overlaps);
  const workspaceSelectionRef = useRef(workspaceSelection);
  const showOverlay3dRef = useRef(showOverlay3d);
  const stepsRef = useRef(maxSteps);
  const maxCrossingsRef = useRef(maxCrossings);
  const diagnosticModeRef = useRef<0 | 1 | 2>(diagnosticMode);
  const qualityRef = useRef(quality);
  const roomRef = useRef(room);
  const timelineRef = useRef<MultiverseTimeline>(timeline);
  const durationSecRef = useRef<number>(durationSec);
  const loopAnimRef = useRef<boolean>(loopAnim);
  const isPlayingAnimRef = useRef(false);
  const animProgressRef = useRef(0);
  const isRenderingMovieRef = useRef(false);
  const dirtyRef = useRef(true);
  const overlayDirtyRef = useRef(true);

  graphRef.current = graph;
  overlapsRef.current = overlaps;
  workspaceSelectionRef.current = workspaceSelection;
  showOverlay3dRef.current = showOverlay3d;
  stepsRef.current = maxSteps;
  maxCrossingsRef.current = maxCrossings;
  diagnosticModeRef.current = diagnosticMode;
  qualityRef.current = quality;
  roomRef.current = room;
  timelineRef.current = timeline;
  durationSecRef.current = durationSec;
  loopAnimRef.current = loopAnim;
  dirtyRef.current = true;
  overlayDirtyRef.current = true;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
        depth: false,
        stencil: false,
        powerPreference: 'high-performance',
      });
    } catch {
      setShaderError('WebGL is unavailable in this browser.');
      return;
    }

    if (!renderer.capabilities.isWebGL2) {
      renderer.dispose();
      setShaderError('WebGL2 is required for the multiverse wormhole renderer.');
      return;
    }

    renderer.autoClear = false;
    renderer.setClearColor(0x000000, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.debug.onShaderError = (gl, program) => {
      const log = gl.getProgramInfoLog(program) || 'Wormhole shader failed to link.';
      console.error(log);
      setShaderError(log);
    };
    container.appendChild(renderer.domElement);
    renderer.domElement.className = 'block h-full w-full touch-none select-none';
    renderer.domElement.style.touchAction = 'none';

    const onContextLost = (event: Event) => {
      event.preventDefault();
      setShaderError('WebGL context lost. Waiting for GPU recovery…');
    };
    const onContextRestored = () => {
      setShaderError(null);
      dirtyRef.current = true;
      overlayDirtyRef.current = true;
    };
    renderer.domElement.addEventListener('webglcontextlost', onContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', onContextRestored);

    const isSoftwareGl = detectSoftwareGl(renderer);
    const maxGpuTexSize = renderer.capabilities.maxTextureSize || 2048;
    const skyLayerWidth = isSoftwareGl ? 1024 : Math.min(2048, maxGpuTexSize);
    const skyLayerHeight = Math.max(512, Math.floor(skyLayerWidth / 2));
    const skyArrayManager = new SkyArrayManager(skyLayerWidth, skyLayerHeight);
    const objectUrls = new Map<string, string>();
    const loadSeqById = new Map<string, number>();

    // Always resolve fresh textureLayer indices from SkyArrayManager right before packing
    const buildRenderGraph = (): MultiverseGraph => {
      const current = graphRef.current;
      return {
        universes: current.universes.map((u) => ({
          ...u,
          textureLayer: skyArrayManager.getLayerIndex(u.id),
        })),
        wormholes: current.wormholes,
      };
    };

    let packedGraph = packMultiverseToFloatBuffers(buildRenderGraph());
    let universeDataTex = createFloatTableTexture(
      packedGraph.universeData,
      packedGraph.universeWidth,
      packedGraph.universeHeight
    );
    let mouthDataTex = createFloatTableTexture(
      packedGraph.mouthData,
      packedGraph.mouthWidth,
      packedGraph.mouthHeight
    );
    let gridOverrideTex = createByteTableTexture(
      packedGraph.gridOverrideData,
      packedGraph.gridOverrideWidth,
      packedGraph.gridOverrideHeight
    );

    const uniforms = {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uCameraPos: { value: new THREE.Vector3() },
      uCameraQuat: { value: new THREE.Vector4(0, 0, 0, 1) },
      uMaxSteps: { value: DEFAULT_STEPS },
      uMaxCrossings: { value: DEFAULT_MAX_CROSSINGS },
      uDiagnosticMode: { value: 0 },
      uStartUniverse: { value: 0 },
      uUniverseCount: { value: packedGraph.universeCount },
      uMouthCount: { value: packedGraph.mouthCount },
      uUniverseData: { value: universeDataTex },
      uMouthData: { value: mouthDataTex },
      uGridLineOverrides: { value: gridOverrideTex },
      uSkyArray: { value: skyArrayManager.getTexture() },
    };

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: WORMHOLE_VERTEX_SHADER,
      fragmentShader: WORMHOLE_FRAGMENT_SHADER,
      glslVersion: THREE.GLSL3,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });

    const quadScene = new THREE.Scene();
    const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    quad.frustumCulled = false;
    quadScene.add(quad);

    // Optional translucent 3D wireframe placement overlay rendered over the quad
    const overlayScene = new THREE.Scene();
    const overlayGroup = new THREE.Group();
    overlayScene.add(overlayGroup);

    const viewCamera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);
    viewCamera.position.set(0, 0, -0.15);
    viewCamera.lookAt(0, 0, 3);

    const controls = new OrbitControls(viewCamera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.75;
    controls.zoomSpeed = 0.85;
    controls.panSpeed = 0.6;
    controls.minDistance = 0.02;
    controls.maxDistance = 80;
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };
    controls.target.set(0, 0, 3);
    controls.update();

    const onControlsStart = () => {
      if (isPlayingAnimRef.current) {
        isPlayingAnimRef.current = false;
        setIsPlayingAnim(false);
      }
      dirtyRef.current = true;
    };
    const onControlsChange = () => {
      dirtyRef.current = true;
    };
    const onCanvasTouchMove = (event: TouchEvent) => {
      event.preventDefault();
    };
    controls.addEventListener('start', onControlsStart);
    controls.addEventListener('change', onControlsChange);
    renderer.domElement.addEventListener('touchmove', onCanvasTouchMove, { passive: false });

    const wormholeCenter = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const camUp = new THREE.Vector3();
    const moveDir = new THREE.Vector3();
    const offset = new THREE.Vector3();
    const outward = new THREE.Vector3();
    let transitCooldown = 0;
    let frameId = 0;
    let lastHudTime = 0;
    let lastTime = performance.now();
    let disposed = false;

    const keysDown = new Set<string>();

    const isEditableTarget = (target: EventTarget | null): boolean => {
      const el = target as HTMLElement | null;
      if (!el) return false;
      if (el.isContentEditable) return true;
      return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT';
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (isRenderingMovieRef.current) {
        if (e.key === 'Escape') {
          renderAbortRef.current?.abort();
        }
        return;
      }

      if (e.key === 'Shift') {
        if (!e.repeat && !isEditableTarget(e.target)) {
          toggleUiRef.current();
        }
        return;
      }

      if (isEditableTarget(e.target)) {
        return;
      }

      const k = e.key.toLowerCase();

      if ((k === 'h' || k === 'z') && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (isPlayingAnimRef.current) {
          isPlayingAnimRef.current = false;
          setIsPlayingAnim(false);
        }
        centerDirectRef.current();
        return;
      }

      if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' ', 'space', 'pageup', 'pagedown'].includes(k)) {
        e.preventDefault();
      }

      if (
        [
          'w', 'a', 's', 'd', 'q', 'e',
          'arrowup', 'arrowdown', 'arrowleft', 'arrowright',
          ' ', 'space', 'r', 'pageup',
          'f', 'c', 'pagedown',
        ].includes(k)
      ) {
        if (isRenderingMovieRef.current) return;
        if (isPlayingAnimRef.current) {
          isPlayingAnimRef.current = false;
          setIsPlayingAnim(false);
        }
        keysDown.add(k === ' ' ? 'space' : k);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      keysDown.delete(k === ' ' ? 'space' : k);
    };

    const onWindowBlur = () => {
      keysDown.clear();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onWindowBlur);

    const applySize = () => {
      const width = container.clientWidth || 1;
      const height = container.clientHeight || 1;
      const scale = clamp(qualityRef.current, 0.35, 1);
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5) * scale;
      renderer.setPixelRatio(Math.max(pixelRatio, 0.35));
      renderer.setSize(width, height, false);
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
      viewCamera.aspect = width / Math.max(height, 1);
      viewCamera.updateProjectionMatrix();
      renderer.getDrawingBufferSize(uniforms.uResolution.value);
      dirtyRef.current = true;
    };
    applySize();
    applySizeRef.current = applySize;

    const getActiveClusterTarget = (universeIdxOverride?: number) => {
      const g = buildRenderGraph();
      const packed = packMultiverseToFloatBuffers(g);
      const rawU = universeIdxOverride !== undefined ? universeIdxOverride : roomRef.current;
      const safeU = clamp(rawU, 0, Math.max(0, packed.universeCount - 1));
      const slice = packed.universeMouthSlices[safeU];
      if (!slice || slice.count === 0) {
        return {
          center: [0, 0, 3] as [number, number, number],
          extent: 1,
          mouthCount: 0,
        };
      }
      return {
        center: slice.clusterCenter,
        extent: Math.max(slice.clusterRadius, 0.25),
        mouthCount: slice.count,
      };
    };

    const centerWormholeDirectly = () => {
      const { center, extent, mouthCount } = getActiveClusterTarget();
      const [wx, wy, wz] = center;
      const distance = mouthCount > 1 ? Math.max(5.2, extent * 1.95) : Math.max(2.8, extent * 3.15);
      const side = viewCamera.position.z > wz ? 1 : -1;
      const cameraZ = wz + side * distance;

      viewCamera.position.set(wx, wy, cameraZ);
      controls.target.set(wx, wy, wz);
      viewCamera.up.set(0, 1, 0);
      viewCamera.lookAt(wx, wy, wz);
      controls.update();
      dirtyRef.current = true;
    };
    centerDirectRef.current = centerWormholeDirectly;

    const frameWormhole = (universeIdxOverride?: number) => {
      const { center, extent, mouthCount } = getActiveClusterTarget(universeIdxOverride);
      const [x, y, z] = center;
      const distance = mouthCount > 1 ? Math.max(5.6, extent * 2.15) : Math.max(2.6, extent * 3.4);
      viewCamera.up.set(0, 1, 0);
      viewCamera.position.set(x, y + (mouthCount > 1 ? 0.35 : extent * 0.22), z - distance);
      controls.target.set(x, y, z);
      viewCamera.lookAt(x, y, z);
      controls.update();
      dirtyRef.current = true;
    };
    frameRef.current = () => frameWormhole();

    const frameSpecificMouth = (
      universeIndex: number,
      pos: [number, number, number],
      rad: number
    ) => {
      const clampedU = clamp(universeIndex, 0, Math.max(0, graphRef.current.universes.length - 1));
      roomRef.current = clampedU;
      setRoom(clampedU);
      const distance = Math.max(2.6, Math.max(rad, 0.4) * 3.4);
      viewCamera.up.set(0, 1, 0);
      viewCamera.position.set(pos[0], pos[1] + rad * 0.25, pos[2] - distance);
      controls.target.set(pos[0], pos[1], pos[2]);
      viewCamera.lookAt(pos[0], pos[1], pos[2]);
      controls.update();
      overlayDirtyRef.current = true;
      dirtyRef.current = true;
    };
    frameMouthRef.current = frameSpecificMouth;

    const crossThroat = (forceAim: boolean) => {
      const g = buildRenderGraph();
      const safeU = clamp(roomRef.current, 0, Math.max(0, g.universes.length - 1));
      const nearest = findNearestMouthInUniverse(g, safeU, [
        viewCamera.position.x,
        viewCamera.position.y,
        viewCamera.position.z,
      ]);
      if (!nearest) return;

      viewCamera.getWorldDirection(forward);
      const crossed = crossMouthWorld(
        nearest,
        [viewCamera.position.x, viewCamera.position.y, viewCamera.position.z],
        [forward.x, forward.y, forward.z],
        forceAim
      );

      const exit = wormholeCenter.set(
        crossed.exitPosition[0],
        crossed.exitPosition[1],
        crossed.exitPosition[2]
      );
      outward.set(crossed.exitForward[0], crossed.exitForward[1], crossed.exitForward[2]);
      const lookDist = Math.max(nearest.wormhole.radius * 3.2, 2.6);
      const ahead = offset.copy(exit).addScaledVector(outward, lookDist);

      viewCamera.up.set(0, 1, 0);
      viewCamera.position.copy(exit);
      viewCamera.lookAt(ahead);
      controls.target.copy(ahead);
      controls.update();

      roomRef.current = crossed.destUniverseIndex;
      setRoom(crossed.destUniverseIndex);
      overlayDirtyRef.current = true;
      dirtyRef.current = true;
    };

    const enterThroat = () => {
      crossThroat(true);
      transitCooldown = 10;
    };
    enterThroatRef.current = enterThroat;

    const transitIfInside = () => {
      if (transitCooldown > 0) {
        transitCooldown -= 1;
        return;
      }
      const g = buildRenderGraph();
      const safeU = clamp(roomRef.current, 0, Math.max(0, g.universes.length - 1));
      const nearest = findNearestMouthInUniverse(g, safeU, [
        viewCamera.position.x,
        viewCamera.position.y,
        viewCamera.position.z,
      ]);
      if (!nearest || nearest.gap >= nearest.effectiveR * 0.02) return;
      crossThroat(false);
      transitCooldown = 6;
    };

    const handleKeyboardInput = (deltaSeconds: number) => {
      if (keysDown.size === 0) return false;

      viewCamera.updateMatrixWorld();
      const basis = viewCamera.matrixWorld.elements;
      right.set(basis[0], basis[1], basis[2]);
      camUp.set(basis[4], basis[5], basis[6]);
      forward.set(-basis[8], -basis[9], -basis[10]);

      const moveSpeed = 3.5 * deltaSeconds;
      const rotateSpeed = 1.6 * deltaSeconds;

      moveDir.set(0, 0, 0);

      if (keysDown.has('w') || keysDown.has('arrowup')) moveDir.add(forward);
      if (keysDown.has('s') || keysDown.has('arrowdown')) moveDir.sub(forward);
      if (keysDown.has('d') || keysDown.has('arrowright')) moveDir.add(right);
      if (keysDown.has('a') || keysDown.has('arrowleft')) moveDir.sub(right);
      if (keysDown.has('space') || keysDown.has('r') || keysDown.has('pageup')) {
        moveDir.add(up);
      }
      if (keysDown.has('c') || keysDown.has('f') || keysDown.has('pagedown')) {
        moveDir.sub(up);
      }

      let changed = false;

      if (moveDir.lengthSq() > 1e-6) {
        moveDir.normalize().multiplyScalar(moveSpeed);
        viewCamera.position.add(moveDir);
        controls.target.add(moveDir);
        changed = true;
      }

      let yaw = 0;
      if (keysDown.has('q')) yaw += rotateSpeed;
      if (keysDown.has('e')) yaw -= rotateSpeed;

      if (Math.abs(yaw) > 1e-6) {
        const ahead = Math.max(controls.target.distanceTo(viewCamera.position), 2.5);
        viewCamera.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(camUp, yaw));
        viewCamera.updateMatrixWorld();
        const turned = viewCamera.matrixWorld.elements;
        forward.set(-turned[8], -turned[9], -turned[10]);
        controls.target.copy(viewCamera.position).addScaledVector(forward, ahead);
        changed = true;
      }

      if (changed) {
        controls.update();
        dirtyRef.current = true;
      }

      return changed;
    };

    const syncShaderUniforms = () => {
      uniforms.uCameraPos.value.copy(viewCamera.position);
      uniforms.uCameraQuat.value.set(
        viewCamera.quaternion.x,
        viewCamera.quaternion.y,
        viewCamera.quaternion.z,
        viewCamera.quaternion.w
      );
      uniforms.uMaxSteps.value = clamp(Math.round(stepsRef.current), 20, 150);
      uniforms.uMaxCrossings.value = clamp(Math.round(maxCrossingsRef.current), 1, 16);
      uniforms.uDiagnosticMode.value = diagnosticModeRef.current;

      const nextPacked = packMultiverseToFloatBuffers(buildRenderGraph(), {
        universeData: packedGraph.universeData,
        mouthData: packedGraph.mouthData,
        gridOverrideData: packedGraph.gridOverrideData,
      });

      const clampedStart = clamp(roomRef.current, 0, Math.max(0, nextPacked.universeCount - 1));
      uniforms.uStartUniverse.value = clampedStart;

      if (nextPacked.universeData !== packedGraph.universeData) {
        universeDataTex.dispose();
        universeDataTex = createFloatTableTexture(
          nextPacked.universeData,
          nextPacked.universeWidth,
          nextPacked.universeHeight
        );
        uniforms.uUniverseData.value = universeDataTex;
      } else {
        universeDataTex.needsUpdate = true;
      }

      if (nextPacked.mouthData !== packedGraph.mouthData) {
        mouthDataTex.dispose();
        mouthDataTex = createFloatTableTexture(
          nextPacked.mouthData,
          nextPacked.mouthWidth,
          nextPacked.mouthHeight
        );
        uniforms.uMouthData.value = mouthDataTex;
      } else {
        mouthDataTex.needsUpdate = true;
      }

      if (nextPacked.gridOverrideData !== packedGraph.gridOverrideData) {
        gridOverrideTex.dispose();
        gridOverrideTex = createByteTableTexture(
          nextPacked.gridOverrideData,
          nextPacked.gridOverrideWidth,
          nextPacked.gridOverrideHeight
        );
        uniforms.uGridLineOverrides.value = gridOverrideTex;
      } else {
        gridOverrideTex.needsUpdate = true;
      }

      packedGraph = nextPacked;
      uniforms.uUniverseCount.value = packedGraph.universeCount;
      uniforms.uMouthCount.value = packedGraph.mouthCount;
      uniforms.uSkyArray.value = skyArrayManager.getTexture();
    };

    const captureCurrentKeyframe = (): CameraKeyframe => {
      const g = graphRef.current;
      const safeU = clamp(roomRef.current, 0, Math.max(0, g.universes.length - 1));
      const uNode = g.universes[safeU];
      return {
        id: `kf-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        universeId: uNode?.id,
        position: [viewCamera.position.x, viewCamera.position.y, viewCamera.position.z],
        quaternion: [
          viewCamera.quaternion.x,
          viewCamera.quaternion.y,
          viewCamera.quaternion.z,
          viewCamera.quaternion.w,
        ],
        target: [controls.target.x, controls.target.y, controls.target.z],
        room: safeU,
      };
    };
    captureKeyframeRef.current = captureCurrentKeyframe;

    const applyKeyframePose = (keyframe: CameraKeyframe, syncRoomState = true) => {
      const resolvedRoom = resolveKeyframeUniverseIndex(keyframe, graphRef.current);
      viewCamera.position.set(keyframe.position[0], keyframe.position[1], keyframe.position[2]);
      viewCamera.quaternion.set(
        keyframe.quaternion[0],
        keyframe.quaternion[1],
        keyframe.quaternion[2],
        keyframe.quaternion[3]
      );
      viewCamera.quaternion.normalize();
      viewCamera.updateMatrixWorld();
      controls.target.set(keyframe.target[0], keyframe.target[1], keyframe.target[2]);
      if (roomRef.current !== resolvedRoom) {
        roomRef.current = resolvedRoom;
        if (syncRoomState) setRoom(resolvedRoom);
      }
      overlayDirtyRef.current = true;
      dirtyRef.current = true;
    };
    applyKeyframeRef.current = (kf) => applyKeyframePose(kf, true);

    const renderViewportPasses = (includeOverlay: boolean) => {
      renderer.clear();
      renderer.render(quadScene, quadCamera);
      if (includeOverlay && showOverlay3dRef.current) {
        if (overlayDirtyRef.current) {
          const sel = workspaceSelectionRef.current;
          const g = graphRef.current;
          const safeU = clamp(roomRef.current, 0, Math.max(0, g.universes.length - 1));
          const tl = timelineRef.current;
          const cameraPathPolylines =
            tl.segments.length > 0 ? sampleTimelinePathInUniverse(tl, safeU, 24) : undefined;
          const keyframeMarkers =
            tl.keyframes.length > 0
              ? tl.keyframes
                  .map((kf, idx) => ({
                    idx,
                    uIdx: resolveKeyframeUniverseIndex(kf, g),
                    position: kf.position,
                  }))
                  .filter((item) => item.uIdx === safeU)
                  .map((item) => ({
                    position: item.position,
                    isStart: item.idx === 0,
                    isEnd: item.idx === tl.keyframes.length - 1 && tl.keyframes.length > 1,
                  }))
              : undefined;

          syncWormholeOverlayGroup(overlayGroup, {
            graph: g,
            activeUniverseIndex: safeU,
            selectedWormholeId: sel?.kind === 'wormhole' ? sel.id : null,
            selectedEnd: sel?.kind === 'wormhole' ? sel.end : null,
            overlaps: overlapsRef.current,
            cameraPathPolylines,
            keyframeMarkers,
          });
          overlayDirtyRef.current = false;
        }
        renderer.render(overlayScene, viewCamera);
      }
    };

    renderMovieRef.current = async ({
      timeline: renderTimeline,
      durationSeconds,
      format,
      onProgress,
      signal,
    }) => {
      const savedPose = captureCurrentKeyframe();
      const dprScale = Math.min(window.devicePixelRatio || 1, 1.5);
      const { width, height } = computeRenderDimensions(
        (container.clientWidth || 1280) * dprScale,
        (container.clientHeight || 720) * dprScale,
        1920
      );

      isPlayingAnimRef.current = false;
      setIsPlayingAnim(false);
      isRenderingMovieRef.current = true;
      controls.enabled = false;
      keysDown.clear();

      try {
        renderer.setPixelRatio(1);
        renderer.setSize(width, height, false);
        renderer.domElement.style.width = '100%';
        renderer.domElement.style.height = '100%';
        viewCamera.aspect = width / Math.max(height, 1);
        viewCamera.updateProjectionMatrix();
        uniforms.uResolution.value.set(width, height);

        const result = await renderWormholeMovie({
          sourceCanvas: renderer.domElement,
          width,
          height,
          durationSeconds,
          fps: 30,
          format,
          signal,
          renderFrameAt: (tNorm) => {
            const pose = sampleMultiverseTimeline(renderTimeline, tNorm);
            applyKeyframePose(pose, false);
            syncShaderUniforms();
            // Always exclude wireframe placement guides & path curves from rendered movies
            renderViewportPasses(false);
          },
          onProgress: (frame, total) => {
            onProgress(frame, total, width, height);
          },
        });
        return result;
      } finally {
        isRenderingMovieRef.current = false;
        controls.enabled = true;
        applySize();
        applyKeyframePose(savedPose, true);
        syncShaderUniforms();
        renderViewportPasses(true);
      }
    };

    let lastAnimUiSync = 0;

    const renderFrame = (time: number) => {
      if (disposed) return;
      frameId = requestAnimationFrame(renderFrame);

      if (isRenderingMovieRef.current) return;
      if (document.hidden) return;

      const deltaSeconds = clamp((time - lastTime) / 1000, 0.001, 0.1);
      lastTime = time;

      if (isPlayingAnimRef.current && timelineRef.current.segments.length > 0) {
        const dur = Math.max(0.5, durationSecRef.current);
        let nextT = animProgressRef.current + deltaSeconds / dur;
        let finished = false;

        if (nextT >= 1) {
          if (loopAnimRef.current) {
            nextT = nextT % 1;
          } else {
            nextT = 1;
            finished = true;
          }
        }
        animProgressRef.current = nextT;

        const pose = sampleMultiverseTimeline(timelineRef.current, nextT);
        applyKeyframePose(pose, true);

        if (finished) {
          isPlayingAnimRef.current = false;
          setIsPlayingAnim(false);
          setAnimProgress(1);
        } else if (time - lastAnimUiSync > 50) {
          lastAnimUiSync = time;
          setAnimProgress(nextT);
        }
      } else {
        const kbMoved = handleKeyboardInput(deltaSeconds);
        const moved = controls.update();
        transitIfInside();
        if (!moved && !kbMoved && !dirtyRef.current) return;
      }

      dirtyRef.current = false;

      syncShaderUniforms();
      renderViewportPasses(true);

      if (time - lastHudTime > 120) {
        lastHudTime = time;
        const g = buildRenderGraph();
        const safeU = clamp(roomRef.current, 0, Math.max(0, g.universes.length - 1));
        const nearest = findNearestMouthInUniverse(g, safeU, [
          viewCamera.position.x,
          viewCamera.position.y,
          viewCamera.position.z,
        ]);
        const distance = nearest ? nearest.distance : 0;
        const gap = nearest ? nearest.gap : 0;
        const next = {
          x: viewCamera.position.x,
          y: viewCamera.position.y,
          z: viewCamera.position.z,
          distance,
          gap,
        };
        setHud((prev) => {
          if (
            Math.abs(prev.x - next.x) < 0.005 &&
            Math.abs(prev.y - next.y) < 0.005 &&
            Math.abs(prev.z - next.z) < 0.005 &&
            Math.abs(prev.distance - next.distance) < 0.005 &&
            Math.abs(prev.gap - next.gap) < 0.005
          ) {
            return prev;
          }
          return next;
        });
      }
    };
    frameId = requestAnimationFrame(renderFrame);

    const resizeObserver = new ResizeObserver(() => {
      if (!isRenderingMovieRef.current) applySize();
    });
    resizeObserver.observe(container);

    // Synchronize layer indices back into React graph state whenever SkyArrayManager changes
    const syncGraphLayersToState = (
      updatedUniverseId?: string,
      patch?: Partial<UniverseNode>
    ) => {
      setGraph((prev) => ({
        ...prev,
        universes: prev.universes.map((u) => {
          const base = { ...u, textureLayer: skyArrayManager.getLayerIndex(u.id) };
          if (updatedUniverseId && u.id === updatedUniverseId && patch) {
            return { ...base, ...patch };
          }
          return base;
        }),
      }));
      dirtyRef.current = true;
    };

    const setSkyMeta = (universeId: string, patch: Partial<UniverseSkyMeta>) => {
      setSkyMetaById((prev) => ({
        ...prev,
        [universeId]: {
          fileLabel: prev[universeId]?.fileLabel ?? null,
          isLoading: prev[universeId]?.isLoading ?? false,
          error: prev[universeId]?.error ?? null,
          ...patch,
        },
      }));
    };

    const onUploadUniverseSky = async (universeId: string, file: File) => {
      const seq = (loadSeqById.get(universeId) ?? 0) + 1;
      loadSeqById.set(universeId, seq);
      setSkyMeta(universeId, { isLoading: true, error: null });

      const prevUrl = objectUrls.get(universeId);
      if (prevUrl) {
        URL.revokeObjectURL(prevUrl);
        objectUrls.delete(universeId);
      }

      const kind = await detectSkyFileKind(file);
      if (disposed || loadSeqById.get(universeId) !== seq) return;

      if (kind === 'hdr' || kind === 'exr') {
        try {
          const maxTexSize = renderer.capabilities.maxTextureSize || 4096;
          const texture = await loadHdriTexture(file, kind, maxTexSize);
          if (disposed || loadSeqById.get(universeId) !== seq) {
            texture.dispose();
            return;
          }
          const layerPixels = packDataTextureToHalfFloatLayer(
            texture,
            skyArrayManager.getLayerWidth(),
            skyArrayManager.getLayerHeight()
          );
          texture.dispose();
          skyArrayManager.setLayer(universeId, layerPixels);
          setSkyMeta(universeId, { fileLabel: file.name, isLoading: false, error: null });
          syncGraphLayersToState(universeId, { isHdr: true, skyMode: 'texture' });
        } catch (err) {
          if (disposed || loadSeqById.get(universeId) !== seq) return;
          const msg = err instanceof Error ? err.message : 'Failed to parse HDRI file.';
          setSkyMeta(universeId, { isLoading: false, error: msg });
        }
        return;
      }

      const url = URL.createObjectURL(file);
      objectUrls.set(universeId, url);
      const image = new Image();
      image.onload = () => {
        URL.revokeObjectURL(url);
        if (objectUrls.get(universeId) === url) objectUrls.delete(universeId);
        if (disposed || loadSeqById.get(universeId) !== seq) return;
        const layerPixels = packImageToHalfFloatLayer(
          image,
          skyArrayManager.getLayerWidth(),
          skyArrayManager.getLayerHeight()
        );
        skyArrayManager.setLayer(universeId, layerPixels);
        setSkyMeta(universeId, { fileLabel: file.name, isLoading: false, error: null });
        syncGraphLayersToState(universeId, { isHdr: false, skyMode: 'texture' });
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        if (objectUrls.get(universeId) === url) objectUrls.delete(universeId);
        if (disposed || loadSeqById.get(universeId) !== seq) return;
        setSkyMeta(universeId, {
          isLoading: false,
          error: 'Could not decode image. Use .hdr, .exr, .png, .jpg, or .webp.',
        });
      };
      image.src = url;
    };
    uploadUniverseSkyRef.current = (id, file) => {
      void onUploadUniverseSky(id, file);
    };

    const clearUniverseSky = (universeId: string) => {
      loadSeqById.set(universeId, (loadSeqById.get(universeId) ?? 0) + 1);
      skyArrayManager.removeLayer(universeId);
      const prevUrl = objectUrls.get(universeId);
      if (prevUrl) {
        URL.revokeObjectURL(prevUrl);
        objectUrls.delete(universeId);
      }
      setSkyMeta(universeId, { fileLabel: null, isLoading: false, error: null });
      syncGraphLayersToState(universeId, { isHdr: false, skyMode: 'grid' });
    };
    clearUniverseSkyRef.current = clearUniverseSky;

    const clearAllSkies = (keepIds?: Set<string>) => {
      for (const id of Array.from(loadSeqById.keys())) {
        if (keepIds?.has(id)) continue;
        loadSeqById.set(id, (loadSeqById.get(id) ?? 0) + 1);
        skyArrayManager.removeLayer(id);
        const url = objectUrls.get(id);
        if (url) {
          URL.revokeObjectURL(url);
          objectUrls.delete(id);
        }
      }
      // Also check any universe ID currently in graph
      for (const u of graphRef.current.universes) {
        if (keepIds?.has(u.id)) continue;
        skyArrayManager.removeLayer(u.id);
      }
      setSkyMetaById((prev) => {
        if (!keepIds || keepIds.size === 0) return {};
        const next: Record<string, UniverseSkyMeta> = {};
        for (const [k, v] of Object.entries(prev)) {
          if (keepIds.has(k)) next[k] = v;
        }
        return next;
      });
      dirtyRef.current = true;
    };
    clearAllSkiesRef.current = clearAllSkies;

    return () => {
      disposed = true;
      renderAbortRef.current?.abort();
      renderAbortRef.current = null;
      isPlayingAnimRef.current = false;
      isRenderingMovieRef.current = false;
      captureKeyframeRef.current = () => null;
      applyKeyframeRef.current = () => {};
      renderMovieRef.current = null;
      uploadUniverseSkyRef.current = () => {};
      clearUniverseSkyRef.current = () => {};
      clearAllSkiesRef.current = () => {};
      applySizeRef.current = () => {};
      centerDirectRef.current = () => {};
      frameRef.current = () => {};
      frameMouthRef.current = () => {};
      enterThroatRef.current = () => {};
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.removeEventListener('start', onControlsStart);
      controls.removeEventListener('change', onControlsChange);
      renderer.domElement.removeEventListener('touchmove', onCanvasTouchMove);
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', onContextRestored);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onWindowBlur);
      controls.dispose();
      material.dispose();
      quad.geometry.dispose();
      disposeOverlayGroup(overlayGroup);
      universeDataTex.dispose();
      mouthDataTex.dispose();
      gridOverrideTex.dispose();
      skyArrayManager.dispose();
      for (const url of objectUrls.values()) {
        URL.revokeObjectURL(url);
      }
      objectUrls.clear();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    applySizeRef.current();
  }, [quality]);

  const handleMutateGraph = (updater: (prev: MultiverseGraph) => MultiverseGraph) => {
    setActivePreset('custom');
    setGraph(updater);
  };

  const handleSelectPreset = (nextPreset: MultiversePresetId) => {
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes([]);
    animProgressRef.current = 0;
    setAnimProgress(0);
    setRenderNotice(null);
    clearAllSkiesRef.current(new Set(['A', 'B']));
    const nextGraph = createInitialGraph(nextPreset, graphRef.current);
    graphRef.current = nextGraph;
    setGraph(nextGraph);
    setActivePreset(nextPreset);
    setWorkspaceSelection({ kind: 'universe', id: nextGraph.universes[0]?.id ?? 'A' });
    roomRef.current = 0;
    setRoom(0);
    overlayDirtyRef.current = true;
    frameRef.current();
  };

  const handleImportGraph = (
    nextGraph: MultiverseGraph,
    importedKeyframes: SerializedCameraKeyframe[]
  ) => {
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    clearAllSkiesRef.current();
    graphRef.current = nextGraph;
    setGraph(nextGraph);
    setKeyframes(importedKeyframes);
    animProgressRef.current = 0;
    setAnimProgress(0);
    setRenderNotice(null);
    setActivePreset('custom');
    setWorkspaceSelection(
      nextGraph.universes[0] ? { kind: 'universe', id: nextGraph.universes[0].id } : null
    );
    roomRef.current = 0;
    setRoom(0);
    overlayDirtyRef.current = true;
    if (importedKeyframes.length > 0) {
      applyKeyframeRef.current(importedKeyframes[0]);
    } else {
      frameRef.current();
    }
  };

  const handleTeleportToUniverse = (universeIndex: number) => {
    const clamped = clamp(universeIndex, 0, Math.max(0, graph.universes.length - 1));
    roomRef.current = clamped;
    setRoom(clamped);
    overlayDirtyRef.current = true;
    frameRef.current();
  };

  const resetAll = () => {
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes([]);
    animProgressRef.current = 0;
    setAnimProgress(0);
    setRenderNotice(null);
    clearAllSkiesRef.current();
    const fresh = createInitialGraph('classic-binary');
    graphRef.current = fresh;
    setGraph(fresh);
    setActivePreset('classic-binary');
    setWorkspaceSelection({ kind: 'universe', id: 'A' });
    setMaxSteps(DEFAULT_STEPS);
    setMaxCrossings(DEFAULT_MAX_CROSSINGS);
    setQuality(DEFAULT_QUALITY);
    roomRef.current = 0;
    setRoom(0);
    overlayDirtyRef.current = true;
    centerDirectRef.current();
  };

  // Quick sidebar helpers for single-wormhole classic editing
  const singleWormhole = graph.wormholes.length === 1 ? graph.wormholes[0] : null;
  const updateSingleWormhole = (updater: (wh: MultiverseGraph['wormholes'][0]) => MultiverseGraph['wormholes'][0]) => {
    setGraph((prev) => {
      if (prev.wormholes.length !== 1) return prev;
      return { ...prev, wormholes: [updater(prev.wormholes[0])] };
    });
  };

  const updateUniverseField = (universeId: string, patch: Partial<UniverseNode>) => {
    setGraph((prev) => ({
      ...prev,
      universes: prev.universes.map((u) => (u.id === universeId ? { ...u, ...patch } : u)),
    }));
  };

  const handleAddKeyframe = () => {
    const pose = captureKeyframeRef.current();
    if (!pose) return;
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes((prev) => [...prev, pose]);
    animProgressRef.current = 1;
    setAnimProgress(1);
    setRenderNotice(null);
  };

  const handleGenerateAutoTour = () => {
    const tour = generateMultiverseTourKeyframes(graphRef.current, roomRef.current);
    if (tour.length === 0) return;
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes(tour);
    const recommendedDuration = clamp(Math.max(4, tour.length * 2.5), 4, 30);
    setDurationSec(recommendedDuration);
    setDurationInput(String(recommendedDuration));
    animProgressRef.current = 0;
    setAnimProgress(0);
    setRenderNotice(null);
    applyKeyframeRef.current(tour[0]);
  };

  const handleUpdateKeyframe = (index: number) => {
    const pose = captureKeyframeRef.current();
    if (!pose) return;
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes((prev) =>
      prev.map((kf, i) => (i === index ? { ...pose, id: kf.id ?? pose.id } : kf))
    );
    setRenderNotice(null);
  };

  const handleMoveKeyframe = (index: number, delta: -1 | 1) => {
    const targetIdx = index + delta;
    if (targetIdx < 0 || targetIdx >= keyframes.length) return;
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes((prev) => {
      const next = [...prev];
      const [item] = next.splice(index, 1);
      next.splice(targetIdx, 0, item);
      return next;
    });
    setRenderNotice(null);
  };

  const handleDeleteKeyframe = (index: number) => {
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes((prev) => prev.filter((_, i) => i !== index));
    animProgressRef.current = 0;
    setAnimProgress(0);
    setRenderNotice(null);
  };

  const handleJumpToKeyframe = (index: number) => {
    const kf = keyframes[index];
    if (!kf) return;
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    const tPos =
      index === 0
        ? 0
        : index === keyframes.length - 1
          ? 1
          : timeline.segments[index]?.tStart ?? index / Math.max(1, keyframes.length - 1);
    animProgressRef.current = tPos;
    setAnimProgress(tPos);
    applyKeyframeRef.current(kf);
  };

  const handleClearKeyframes = () => {
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    setKeyframes([]);
    animProgressRef.current = 0;
    setAnimProgress(0);
    setRenderNotice(null);
  };

  const handleTogglePlay = () => {
    if (timeline.segments.length === 0 || renderStatus?.active) return;
    if (isPlayingAnimRef.current) {
      isPlayingAnimRef.current = false;
      setIsPlayingAnim(false);
      return;
    }
    if (animProgressRef.current >= 0.999) {
      animProgressRef.current = 0;
      setAnimProgress(0);
    }
    const pose = sampleMultiverseTimeline(timeline, animProgressRef.current);
    applyKeyframeRef.current(pose);
    isPlayingAnimRef.current = true;
    setIsPlayingAnim(true);
  };

  const handleScrub = (nextT: number) => {
    if (timeline.segments.length === 0 || renderStatus?.active) return;
    isPlayingAnimRef.current = false;
    setIsPlayingAnim(false);
    const clamped = clamp(nextT, 0, 1);
    animProgressRef.current = clamped;
    setAnimProgress(clamped);
    const pose = sampleMultiverseTimeline(timeline, clamped);
    applyKeyframeRef.current(pose);
  };

  const handleCancelRender = () => {
    renderAbortRef.current?.abort();
  };

  const handleRenderMovie = async () => {
    if (timeline.segments.length === 0 || !renderMovieRef.current || renderStatus?.active) return;

    const safeDuration = clamp(Number.isFinite(durationSec) ? durationSec : 6, 0.5, 60);
    setDurationSec(safeDuration);
    setRenderNotice(null);

    const abortController = new AbortController();
    renderAbortRef.current = abortController;
    const estimatedTotal = Math.max(2, Math.round(safeDuration * 30));

    setRenderStatus({
      active: true,
      frame: 0,
      total: estimatedTotal,
      width: 0,
      height: 0,
      format: movieFormat,
    });

    try {
      const result = await renderMovieRef.current({
        timeline,
        durationSeconds: safeDuration,
        format: movieFormat,
        signal: abortController.signal,
        onProgress: (frame, total, width, height) => {
          setRenderStatus({
            active: true,
            frame,
            total,
            width,
            height,
            format: movieFormat,
          });
        },
      });

      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const filename = `wormhole-${interpMode}-${safeDuration}s-${stamp}.${result.extension}`;
      downloadVideoBlob(result.blob, filename);

      const fallbackNote =
        result.extension !== movieFormat
          ? ` (exported as .${result.extension.toUpperCase()} for browser compatibility)`
          : '';
      setRenderNotice({
        kind: 'success',
        text: `Downloaded ${result.totalFrames} frames (${result.width}×${result.height} @ ${result.fps}fps)${fallbackNote}`,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setRenderNotice({ kind: 'error', text: 'Movie render cancelled.' });
      } else {
        const message = err instanceof Error ? err.message : 'Failed to render movie.';
        setRenderNotice({ kind: 'error', text: message });
      }
    } finally {
      renderAbortRef.current = null;
      setRenderStatus(null);
    }
  };

  const safeRoomIndex = clamp(room, 0, Math.max(0, graph.universes.length - 1));
  const activeUniverse = graph.universes[safeRoomIndex] ?? graph.universes[0];
  const quickUniverses =
    graph.universes.length <= 2 ? graph.universes : activeUniverse ? [activeUniverse] : [];

  const clearance = hud.gap;
  const hasReadyTimeline = keyframes.length >= 2 && timeline.segments.length > 0;
  const routeChainLabel = useMemo(() => {
    if (timeline.universeSequence.length === 0) return '';
    return timeline.universeSequence
      .map((uIdx) => graph.universes[uIdx]?.id ?? `U${uIdx + 1}`)
      .join(' → ');
  }, [timeline.universeSequence, graph.universes]);
  const isRenderingMovie = Boolean(renderStatus?.active);
  const showControlsOverlay = isUiVisible && !isRenderingMovie;
  const activeInterpOption =
    INTERPOLATION_OPTIONS.find((opt) => opt.id === interpMode) ?? INTERPOLATION_OPTIONS[1];
  const renderPct =
    renderStatus && renderStatus.total > 0
      ? Math.min(100, Math.round((renderStatus.frame / renderStatus.total) * 100))
      : 0;

  const avgGravity =
    graph.wormholes.length > 0
      ? graph.wormholes.reduce((sum, w) => sum + w.gravityRatio, 0) / graph.wormholes.length
      : DEFAULT_GRAVITY;

  return (
    <div className="relative h-full w-full overflow-hidden bg-black text-slate-100">
      <div ref={containerRef} className="absolute inset-0 z-0 touch-none select-none" />

      {/* Left Control Sidebar */}
      <div
        className={`absolute left-4 top-4 z-10 flex max-h-[calc(100%-5.5rem)] w-80 max-w-[calc(100%-2rem)] flex-col gap-3 overflow-y-auto rounded-xl border border-fuchsia-500/35 bg-slate-950/92 p-3.5 shadow-2xl transition-all duration-300 ${
          showControlsOverlay ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-6 pointer-events-none'
        }`}
      >
        <div className="flex items-center justify-between border-b border-fuchsia-500/25 pb-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-fuchsia-100">Wormhole</span>
            <span className="rounded border border-fuchsia-500/40 bg-fuchsia-950/70 px-1.5 py-0.5 font-mono text-[10px] text-fuchsia-200 tabular-nums">
              {graph.universes.length}U · {graph.wormholes.length}W
            </span>
          </div>
          <button
            type="button"
            onClick={toggleUi}
            className="flex items-center gap-1.5 rounded border border-slate-700/80 bg-slate-900/80 px-2 py-0.5 text-[10px] font-medium text-slate-300 hover:bg-slate-800 hover:text-fuchsia-200 transition"
            title="Hide user interface (Press Shift)"
          >
            <EyeOff className="w-3 h-3 text-slate-400" />
            <span>Hide</span>
            <kbd className="font-mono text-[9px] text-fuchsia-300">Shift</kbd>
          </button>
        </div>

        {/* Observer Status & Navigation */}
        <div className="flex items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/70 px-2.5 py-2">
          <div className="flex min-w-0 flex-col">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Observer</span>
            <span className="truncate font-mono text-[11px] text-slate-100" title={activeUniverse?.name}>
              {activeUniverse?.name ?? 'Universe A'}
            </span>
          </div>
          <div className="flex shrink-0 gap-1">
            <button
              type="button"
              onClick={() => enterThroatRef.current()}
              disabled={graph.wormholes.length === 0}
              className="rounded border border-fuchsia-400/50 bg-fuchsia-900/50 px-2 py-1 text-[10px] font-semibold text-fuchsia-50 transition hover:bg-fuchsia-800/70 disabled:opacity-40"
            >
              {graph.wormholes.length === 1 ? 'Enter throat' : 'Enter nearest'}
            </button>
            <button
              type="button"
              disabled={graph.universes.length <= 1}
              onClick={() => {
                const count = Math.max(1, graph.universes.length);
                const next = (safeRoomIndex + 1) % count;
                roomRef.current = next;
                setRoom(next);
              }}
              className="rounded border border-fuchsia-500/40 bg-fuchsia-950/50 px-2 py-1 text-[10px] font-semibold text-fuchsia-100 transition hover:bg-fuchsia-900/60 disabled:opacity-40"
            >
              {graph.universes.length > 2 ? 'Next univ' : 'Swap side'}
            </button>
          </div>
        </div>

        {/* Multiverse Topology & Stage 3 Architect Workspace Launcher */}
        <div className="flex flex-col gap-2 rounded-lg border border-fuchsia-500/30 bg-slate-950/80 p-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-200/90">
              Multiverse Topology
            </span>
            <span className="font-mono text-[9px] text-fuchsia-300/80">
              {activePreset === 'custom'
                ? 'Custom'
                : MULTIVERSE_PRESETS.find((p) => p.id === activePreset)?.badge}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-1">
            {MULTIVERSE_PRESETS.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => handleSelectPreset(option.id)}
                className={`flex flex-col items-center rounded border px-1.5 py-1 text-center transition ${
                  activePreset === option.id
                    ? 'border-fuchsia-400 bg-fuchsia-950/85 text-fuchsia-100'
                    : 'border-slate-700 bg-slate-900/80 text-slate-300 hover:border-fuchsia-500/40 hover:text-fuchsia-100'
                }`}
              >
                <span className="text-[10px] font-semibold leading-tight">{option.label}</span>
                <span className="font-mono text-[8px] text-fuchsia-300/80">{option.badge}</span>
              </button>
            ))}
          </div>

          <div className="grid grid-cols-[1fr_auto] gap-1.5">
            <button
              type="button"
              onClick={() => setWorkspaceOpen((prev) => !prev)}
              className={`flex items-center justify-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11px] font-semibold transition ${
                workspaceOpen
                  ? 'border-fuchsia-300 bg-fuchsia-900/90 text-white'
                  : 'border-fuchsia-400/70 bg-gradient-to-r from-fuchsia-950 to-purple-950 text-fuchsia-100 hover:border-fuchsia-300 hover:from-fuchsia-900 hover:to-purple-900'
              }`}
            >
              <GitBranch className="h-3.5 w-3.5 text-fuchsia-300" />
              <span>{workspaceOpen ? 'Hide Architect' : 'Multiverse Architect'}</span>
            </button>

            <button
              type="button"
              onClick={() => setShowOverlay3d((prev) => !prev)}
              title="Toggle translucent 3D wireframe cages over wormhole mouths"
              className={`flex items-center gap-1 rounded-lg border px-2 py-1.5 text-[10px] font-semibold transition ${
                showOverlay3d
                  ? 'border-sky-400 bg-sky-950/90 text-sky-100'
                  : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
              }`}
            >
              <Box className="h-3.5 w-3.5" />
              <span>3D</span>
            </button>
          </div>

          <div className="flex flex-col gap-1 pt-0.5">
            <div className="flex items-center justify-between">
              <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                Render Mode
              </span>
              {diagnosticMode !== 0 && (
                <span className="font-mono text-[9px] text-amber-300/90">
                  {diagnosticMode === 1 ? 'Phase + |det J|' : 'RGBA Float Map'}
                </span>
              )}
            </div>
            <div className="grid grid-cols-3 gap-1">
              {(
                [
                  { mode: 0 as const, label: 'Normal', title: 'Standard sky grid / HDRI render' },
                  {
                    mode: 1 as const,
                    label: 'Diagnostic',
                    title: 'Topological phase regions with caustic fold contours (|det J| → 0)',
                  },
                  {
                    mode: 2 as const,
                    label: 'Raw Data',
                    title: 'Raw RGBA output: R=u, G=v, B=topology, A=foldMetric',
                  },
                ]
              ).map((item) => (
                <button
                  key={item.mode}
                  type="button"
                  title={item.title}
                  onClick={() => setDiagnosticMode(item.mode)}
                  className={`rounded border px-1.5 py-1 text-[10px] font-semibold transition ${
                    diagnosticMode === item.mode
                      ? item.mode === 0
                        ? 'border-fuchsia-400 bg-fuchsia-950/85 text-fuchsia-100'
                        : 'border-amber-400 bg-amber-950/80 text-amber-100'
                      : 'border-slate-700 bg-slate-900/80 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {diagnosticMode === 1 && (
              <p className="font-mono text-[9px] leading-snug text-slate-400">
                Flat tints = (universe, crossings) · Gold/white = caustic folds (det J = 0)
              </p>
            )}
            {diagnosticMode === 2 && (
              <p className="font-mono text-[9px] leading-snug text-slate-400">
                R: u · G: v · B: uIdx + xings/max (-1 cap) · A: foldMetric
              </p>
            )}
          </div>

          {overlaps.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setWorkspaceOpen(true);
                setWorkspaceSelection({
                  kind: 'wormhole',
                  id: overlaps[0].wormholeAId,
                  end: overlaps[0].endA,
                });
              }}
              className="flex items-center gap-1.5 rounded border border-rose-500/50 bg-rose-950/75 px-2 py-1 text-left text-[10px] font-medium text-rose-100 transition hover:bg-rose-900/80"
            >
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-300" />
              <span className="truncate">
                {overlaps.length} mouth overlap{overlaps.length > 1 ? 's' : ''} — click to fix
              </span>
            </button>
          )}
        </div>

        {/* Stage 4: Multiverse Camera Choreography & Offline Movie Render */}
        <div className="flex flex-col gap-2 rounded-lg border border-fuchsia-500/30 bg-slate-950/80 p-2.5">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-fuchsia-200">
              <Clapperboard className="h-3.5 w-3.5 text-fuchsia-400" />
              Camera Animation
            </span>
            {keyframes.length > 0 && (
              <button
                type="button"
                onClick={handleClearKeyframes}
                className="flex items-center gap-1 rounded border border-slate-800 bg-slate-900/80 px-1.5 py-0.5 text-[9px] text-slate-400 transition hover:border-slate-700 hover:text-slate-200"
                title="Clear all captured keyframes"
              >
                <Trash2 className="h-2.5 w-2.5" />
                <span>Clear</span>
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={handleAddKeyframe}
              className="flex items-center justify-center gap-1.5 rounded border border-fuchsia-500/50 bg-fuchsia-950/55 px-2 py-1.5 text-[11px] font-semibold text-fuchsia-100 transition hover:border-fuchsia-400 hover:bg-fuchsia-900/65"
              title="Capture current camera position, orientation, and active universe as next keyframe"
            >
              <Plus className="h-3.5 w-3.5 shrink-0" />
              <span>Add Keyframe</span>
            </button>

            <button
              type="button"
              onClick={handleGenerateAutoTour}
              className="flex items-center justify-center gap-1.5 rounded border border-amber-500/40 bg-amber-950/35 px-2 py-1.5 text-[11px] font-semibold text-amber-100 transition hover:border-amber-400/70 hover:bg-amber-900/50"
              title="Automatically build a collision-free tour visiting every connected universe"
            >
              <Compass className="h-3.5 w-3.5 shrink-0 text-amber-300" />
              <span>Auto Tour</span>
            </button>
          </div>

          {keyframes.length === 0 ? (
            <p className="text-[10px] leading-snug text-slate-400">
              Capture 2+ poses across any universes, or click Auto Tour for an instant multiverse flythrough.
            </p>
          ) : (
            <div className="flex flex-col gap-1 max-h-40 overflow-y-auto pr-0.5">
              {keyframes.map((kf, idx) => {
                const uIdx = resolveKeyframeUniverseIndex(kf, graph);
                const uNode = graph.universes[uIdx];
                return (
                  <div
                    key={kf.id ?? `kf-${idx}`}
                    className="flex items-center gap-1 rounded border border-slate-800/90 bg-slate-900/65 px-1.5 py-1 text-[10px]"
                  >
                    <button
                      type="button"
                      onClick={() => handleJumpToKeyframe(idx)}
                      className="flex min-w-0 flex-1 items-center gap-1.5 text-left transition hover:text-fuchsia-100"
                      title={`Jump camera to Keyframe #${idx + 1} (${uNode?.name ?? `Universe ${uIdx + 1}`})`}
                    >
                      <span className="font-mono text-[9px] font-semibold text-fuchsia-300 tabular-nums">
                        #{idx + 1}
                      </span>
                      <span
                        className="h-2 w-2 shrink-0 rounded-sm border border-white/25"
                        style={{ backgroundColor: uNode?.gridVerticalColor ?? '#e879f9' }}
                      />
                      <span className="font-mono text-[9px] font-semibold text-slate-200 shrink-0">
                        {uNode?.id ?? `U${uIdx + 1}`}
                      </span>
                      <span className="truncate font-mono text-[9px] text-slate-400 tabular-nums">
                        {kf.position[0].toFixed(1)},{kf.position[1].toFixed(1)},{kf.position[2].toFixed(1)}
                      </span>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleUpdateKeyframe(idx)}
                      className="rounded p-0.5 text-slate-400 transition hover:bg-slate-800 hover:text-emerald-300"
                      title="Overwrite this keyframe with current camera pose"
                    >
                      <RefreshCw className="h-2.5 w-2.5" />
                    </button>
                    <button
                      type="button"
                      disabled={idx === 0}
                      onClick={() => handleMoveKeyframe(idx, -1)}
                      className="rounded p-0.5 text-slate-400 transition hover:bg-slate-800 hover:text-slate-100 disabled:opacity-25"
                      title="Move earlier"
                    >
                      <ArrowUp className="h-2.5 w-2.5" />
                    </button>
                    <button
                      type="button"
                      disabled={idx === keyframes.length - 1}
                      onClick={() => handleMoveKeyframe(idx, 1)}
                      className="rounded p-0.5 text-slate-400 transition hover:bg-slate-800 hover:text-slate-100 disabled:opacity-25"
                      title="Move later"
                    >
                      <ArrowDown className="h-2.5 w-2.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeleteKeyframe(idx)}
                      className="rounded p-0.5 text-slate-400 transition hover:bg-slate-800 hover:text-rose-300"
                      title="Delete keyframe"
                    >
                      <Trash2 className="h-2.5 w-2.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {keyframes.length === 1 && (
            <p className="text-[10px] leading-snug text-slate-400">
              Move or cross into another universe and add a second keyframe to build the route.
            </p>
          )}

          {hasReadyTimeline && (
            <div className="flex flex-col gap-1 border-t border-slate-800/80 pt-1.5">
              <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[10px] text-slate-400 tabular-nums">
                <span>{keyframes.length} poses</span>
                <span aria-hidden="true">·</span>
                <span>
                  {timeline.totalHops} portal {timeline.totalHops === 1 ? 'crossing' : 'crossings'}
                </span>
                {routeChainLabel && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="font-mono text-[9px] text-fuchsia-200">{routeChainLabel}</span>
                  </>
                )}
              </div>

              {timeline.disconnectedSegmentIndices.length > 0 && (
                <div className="flex items-start gap-1.5 rounded border border-amber-500/40 bg-amber-950/50 px-2 py-1 text-[10px] leading-snug text-amber-200">
                  <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-amber-300" />
                  <span>
                    {timeline.disconnectedSegmentIndices.length} disconnected jump
                    {timeline.disconnectedSegmentIndices.length > 1 ? 's' : ''} (no wormhole bridge — cuts directly).
                  </span>
                </div>
              )}
            </div>
          )}

          <div className="flex flex-col gap-1 border-t border-slate-800/80 pt-2">
            <div className="flex items-center justify-between text-[10px]">
              <span className="font-semibold text-slate-300">Interpolation</span>
              <span className="font-mono text-[9px] text-fuchsia-300">{activeInterpOption.formula}</span>
            </div>
            <div className="grid grid-cols-3 gap-1">
              {INTERPOLATION_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setInterpMode(opt.id)}
                  title={opt.description}
                  className={`rounded border px-1.5 py-1 text-[10px] font-semibold transition ${
                    interpMode === opt.id
                      ? 'border-fuchsia-400 bg-fuchsia-950/80 text-fuchsia-100'
                      : 'border-slate-700 bg-slate-900/80 text-slate-400 hover:border-fuchsia-500/40 hover:text-slate-200'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 border-t border-slate-800/80 pt-2">
            <label className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold text-slate-300">Duration (sec)</span>
              <div className="flex items-center rounded border border-slate-700 bg-slate-900/90 px-2 py-1 focus-within:border-fuchsia-400">
                <input
                  type="number"
                  min={0.5}
                  max={60}
                  step={0.5}
                  value={durationInput}
                  onChange={(e) => {
                    const raw = e.target.value;
                    setDurationInput(raw);
                    const parsed = parseFloat(raw);
                    if (Number.isFinite(parsed) && parsed > 0) {
                      setDurationSec(clamp(parsed, 0.5, 60));
                    }
                  }}
                  onBlur={() => {
                    const parsed = parseFloat(durationInput);
                    const safe = Number.isFinite(parsed) ? clamp(parsed, 0.5, 60) : 6;
                    setDurationSec(safe);
                    setDurationInput(String(safe));
                  }}
                  aria-label="Animation duration in seconds"
                  className="w-full bg-transparent font-mono text-xs text-fuchsia-100 outline-none tabular-nums"
                />
                <span className="font-mono text-[10px] text-slate-500 ml-1">s</span>
              </div>
            </label>

            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold text-slate-300">Format</span>
              <div className="grid grid-cols-2 gap-1 h-full">
                {(['mp4', 'webm'] as const).map((fmt) => (
                  <button
                    key={fmt}
                    type="button"
                    onClick={() => setMovieFormat(fmt)}
                    className={`rounded border py-1 font-mono text-[10px] font-semibold uppercase transition ${
                      movieFormat === fmt
                        ? 'border-fuchsia-400 bg-fuchsia-950/80 text-fuchsia-100'
                        : 'border-slate-700 bg-slate-900/80 text-slate-400 hover:border-fuchsia-500/40 hover:text-slate-200'
                    }`}
                  >
                    {fmt}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {hasReadyTimeline && (
            <div className="flex flex-col gap-1 border-t border-slate-800/80 pt-2">
              <div className="flex items-center justify-between font-mono text-[10px] text-slate-400 tabular-nums">
                <span>{(animProgress * durationSec).toFixed(2)}s</span>
                <span className="text-fuchsia-300">{Math.round(animProgress * 100)}%</span>
                <span>{durationSec.toFixed(1)}s</span>
              </div>
              <div className="relative flex items-center">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.002}
                  value={animProgress}
                  aria-label="Animation timeline scrubber"
                  onChange={(e) => handleScrub(parseFloat(e.target.value))}
                  className={`w-full cursor-pointer ${ACCENT.track}`}
                />
              </div>
            </div>
          )}

          <div className="grid grid-cols-[1fr_auto_1fr] gap-1.5 pt-0.5">
            <button
              type="button"
              disabled={!hasReadyTimeline}
              onClick={handleTogglePlay}
              className="flex items-center justify-center gap-1.5 rounded border border-slate-700 bg-slate-900/90 py-1.5 text-[11px] font-semibold text-slate-100 transition hover:border-fuchsia-400/60 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isPlayingAnim ? (
                <>
                  <Pause className="h-3.5 w-3.5 text-fuchsia-300" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 text-fuchsia-300" />
                  <span>Play</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={() => setLoopAnim((prev) => !prev)}
              title={loopAnim ? 'Loop playback enabled' : 'Loop playback disabled'}
              aria-label="Toggle loop playback"
              className={`flex items-center justify-center rounded border px-2 py-1.5 transition ${
                loopAnim
                  ? 'border-fuchsia-400 bg-fuchsia-950/85 text-fuchsia-100'
                  : 'border-slate-700 bg-slate-900/80 text-slate-400 hover:border-slate-500 hover:text-slate-200'
              }`}
            >
              <Repeat className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              disabled={!hasReadyTimeline}
              onClick={handleRenderMovie}
              className="flex items-center justify-center gap-1.5 rounded border border-fuchsia-400/70 bg-gradient-to-r from-fuchsia-900/80 to-purple-900/80 py-1.5 text-[11px] font-semibold text-fuchsia-50 shadow-sm transition hover:border-fuchsia-300 hover:from-fuchsia-800 hover:to-purple-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Download className="h-3.5 w-3.5 text-fuchsia-200" />
              <span>Render {movieFormat.toUpperCase()}</span>
            </button>
          </div>

          <p className="text-[9px] leading-snug text-slate-500">
            Renders {Math.max(2, Math.round(durationSec * 30))} frames one-by-one at 30fps directly from the shader — UI & 3D guides excluded. Enable 3D overlay to preview the camera path.
          </p>

          {renderNotice && (
            <div
              className={`rounded border px-2 py-1.5 text-[10px] leading-snug ${
                renderNotice.kind === 'success'
                  ? 'border-emerald-500/40 bg-emerald-950/60 text-emerald-200'
                  : 'border-amber-500/40 bg-amber-950/60 text-amber-200'
              }`}
            >
              {renderNotice.text}
            </div>
          )}
        </div>

        {/* Keyboard Flight Quick Reference */}
        <div className="rounded-lg border border-slate-800/80 bg-slate-950/70 p-2.5 text-[10px]">
          <div className="flex items-center justify-between font-semibold text-slate-200 mb-1.5">
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-fuchsia-400" />
              Keyboard Flight & Controls
            </span>
            <span className="font-mono text-[9px] text-fuchsia-300">6-DOF</span>
          </div>
          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-slate-400 leading-tight">
            <div><span className="font-mono font-semibold text-slate-200">W / S</span> forward / back</div>
            <div><span className="font-mono font-semibold text-slate-200">A / D</span> strafe left / right</div>
            <div><span className="font-mono font-semibold text-slate-200">Space / C</span> move up / down</div>
            <div><span className="font-mono font-semibold text-slate-200">Q / E</span> rotate left / right</div>
            <div><span className="font-mono font-semibold text-fuchsia-300">H / Z</span> center level facing</div>
            <div><span className="font-mono font-semibold text-cyan-300">Shift</span> hide / show UI</div>
          </div>
          <div className="mt-1.5 pt-1 border-t border-slate-800/80 text-[9px] text-slate-500">
            Also supports <span className="font-mono text-slate-400">R / F</span> (up/down) and <span className="font-mono text-slate-400">Arrows</span>.
          </div>
        </div>

        {/* Quick Universe Sky Cards (Shows A & B for 2-universe scenes, or Active Universe + link to Architect for 3+) */}
        {quickUniverses.map((u) => {
          const meta = skyMetaById[u.id] ?? { fileLabel: null, isLoading: false, error: null };
          return (
            <UniverseSource
              key={u.id}
              title={u.name}
              detail={
                u.skyMode === 'grid'
                  ? `${u.gridCols}×${u.gridRows} · Rot ${Math.round(u.rotationY)}°`
                  : u.isHdr && u.textureLayer >= 0
                    ? `HDRI · ${u.exposureEv >= 0 ? '+' : ''}${u.exposureEv.toFixed(1)} EV`
                    : `Rot ${Math.round(u.rotationY)}°`
              }
              mode={u.skyMode}
              gridConfig={{
                verticalColor: u.gridVerticalColor,
                horizontalColor: u.gridHorizontalColor,
                cols: u.gridCols,
                rows: u.gridRows,
                rotationY: u.rotationY,
                rotationX: u.rotationX,
              }}
              onGridConfigChange={(updater) => {
                setGraph((prev) => ({
                  ...prev,
                  universes: prev.universes.map((node) => {
                    if (node.id !== u.id) return node;
                    const curr: GridConfig = {
                      verticalColor: node.gridVerticalColor,
                      horizontalColor: node.gridHorizontalColor,
                      cols: node.gridCols,
                      rows: node.gridRows,
                      rotationY: node.rotationY,
                      rotationX: node.rotationX,
                    };
                    const next = typeof updater === 'function' ? updater(curr) : updater;
                    return {
                      ...node,
                      gridVerticalColor: next.verticalColor,
                      gridHorizontalColor: next.horizontalColor,
                      gridCols: next.cols,
                      gridRows: next.rows,
                      rotationY: next.rotationY,
                      rotationX: next.rotationX,
                    };
                  }),
                }));
              }}
              lonOverrides={u.gridLonOverrides}
              latOverrides={u.gridLatOverrides}
              onLonOverridesChange={(next) => updateUniverseField(u.id, { gridLonOverrides: next })}
              onLatOverridesChange={(next) => updateUniverseField(u.id, { gridLatOverrides: next })}
              fileLabel={meta.fileLabel}
              hasTexture={u.textureLayer >= 0}
              isHdr={u.isHdr && u.textureLayer >= 0}
              exposure={u.exposureEv}
              onExposureChange={(ev) => updateUniverseField(u.id, { exposureEv: ev })}
              isLoading={meta.isLoading}
              errorMessage={meta.error}
              onMode={(mode) => updateUniverseField(u.id, { skyMode: mode })}
              onUpload={(file) => uploadUniverseSkyRef.current(u.id, file)}
              onClear={() => clearUniverseSkyRef.current(u.id)}
            />
          );
        })}

        {graph.universes.length > 2 && (
          <button
            type="button"
            onClick={() => setWorkspaceOpen(true)}
            className="rounded-lg border border-slate-700 bg-slate-900/90 px-2.5 py-1.5 text-[10px] font-semibold text-slate-300 transition hover:border-fuchsia-400/60 hover:text-fuchsia-100"
          >
            Inspect all {graph.universes.length} universes & {graph.wormholes.length} wormholes in Architect →
          </button>
        )}

        {/* Quick Throat Geometry Controls for Single-Wormhole Scenes */}
        {singleWormhole && (
          <div className="flex flex-col gap-2.5 border-t border-fuchsia-500/20 pt-2.5">
            <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-200/90">
              Throat shape
            </span>
            <div className="grid grid-cols-3 gap-1">
              {WORMHOLE_SHAPES.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  title={option.title}
                  onClick={() =>
                    updateSingleWormhole((wh) => ({ ...wh, shape: option.id as WormholeShapeId }))
                  }
                  className={`rounded border px-1.5 py-1 text-[10px] font-semibold transition ${
                    singleWormhole.shape === option.id
                      ? 'border-fuchsia-400 bg-fuchsia-950/80 text-fuchsia-100'
                      : 'border-slate-700 bg-slate-900/80 text-slate-300 hover:border-fuchsia-500/40 hover:text-fuchsia-100'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <p className="text-[10px] leading-snug text-slate-500">
              Light bends along the surface normal. Crossing the throat preserves vertical orientation across universes without flipping the horizon.
            </p>
            <div className="flex flex-col gap-2 rounded-lg border border-slate-800/80 bg-slate-950/60 p-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-200/90">
                  Shape Dimensions
                </span>
                {(Math.abs(singleWormhole.scale[0] - 1) > 1e-3 ||
                  Math.abs(singleWormhole.scale[1] - 1) > 1e-3 ||
                  Math.abs(singleWormhole.scale[2] - 1) > 1e-3) && (
                  <button
                    type="button"
                    onClick={() =>
                      updateSingleWormhole((wh) => ({ ...wh, scale: [...DEFAULT_SHAPE_SCALE] }))
                    }
                    className="rounded border border-slate-700 bg-slate-900/80 px-1.5 py-0.5 font-mono text-[9px] text-slate-300 transition hover:border-fuchsia-400/60 hover:text-fuchsia-200"
                  >
                    Reset 1:1:1
                  </button>
                )}
              </div>
              {(['Width (X)', 'Height (Y)', 'Length (Z)'] as const).map((label, axis) => (
                <SliderRow
                  key={label}
                  label={label}
                  min={0.25}
                  max={3}
                  step={0.05}
                  value={singleWormhole.scale[axis]}
                  display={`${singleWormhole.scale[axis].toFixed(2)}×`}
                  onChange={(value) =>
                    updateSingleWormhole((wh) => {
                      const next: [number, number, number] = [wh.scale[0], wh.scale[1], wh.scale[2]];
                      next[axis] = value;
                      return { ...wh, scale: next };
                    })
                  }
                  presets={[
                    { label: '0.5×', value: 0.5 },
                    { label: '1×', value: 1 },
                    { label: '2×', value: 2 },
                  ]}
                />
              ))}
            </div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-200/90">
              Throat Placement
            </span>
            {(['Position X', 'Position Y', 'Position Z'] as const).map((label, axis) => (
              <SliderRow
                key={label}
                label={label}
                min={-8}
                max={8}
                step={0.05}
                value={singleWormhole.mouthA.position[axis]}
                display={singleWormhole.mouthA.position[axis].toFixed(2)}
                onChange={(value) =>
                  updateSingleWormhole((wh) => {
                    const nextA: [number, number, number] = [...wh.mouthA.position];
                    const nextB: [number, number, number] = [...wh.mouthB.position];
                    nextA[axis] = value;
                    if (wh.mouthA.universeId !== wh.mouthB.universeId) {
                      nextB[axis] = value;
                    }
                    return {
                      ...wh,
                      mouthA: { ...wh.mouthA, position: nextA },
                      mouthB: { ...wh.mouthB, position: nextB },
                    };
                  })
                }
              />
            ))}
            <SliderRow
              label="Wormhole radius"
              min={0.1}
              max={5}
              step={0.01}
              value={singleWormhole.radius}
              display={singleWormhole.radius.toFixed(2)}
              onChange={(value) => updateSingleWormhole((wh) => ({ ...wh, radius: value }))}
              presets={[
                { label: '0.5', value: 0.5 },
                { label: '1.0', value: 1 },
                { label: '2.5', value: 2.5 },
              ]}
            />
          </div>
        )}

        {/* Raymarcher & Physics */}
        <div className="flex flex-col gap-2.5 border-t border-fuchsia-500/20 pt-2.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-200/90">
            Raymarcher & Physics
          </span>
          {graph.wormholes.length > 0 && (
            <SliderRow
              label={graph.wormholes.length === 1 ? 'Gravity strength' : 'All wormholes gravity'}
              min={0}
              max={2}
              step={0.01}
              value={avgGravity}
              display={avgGravity.toFixed(2)}
              onChange={(value) =>
                setGraph((prev) => ({
                  ...prev,
                  wormholes: prev.wormholes.map((wh) => ({ ...wh, gravityRatio: value })),
                }))
              }
              presets={[
                { label: '0', value: 0 },
                { label: '0.5', value: 0.5 },
                { label: '1.2', value: 1.2 },
              ]}
            />
          )}
          <SliderRow
            label="Max steps"
            min={20}
            max={150}
            step={1}
            value={maxSteps}
            display={String(Math.round(maxSteps))}
            onChange={(value) => setMaxSteps(Math.round(value))}
            presets={[
              { label: '40', value: 40 },
              { label: '80', value: 80 },
              { label: '140', value: 140 },
            ]}
          />
          <SliderRow
            label="Max portal bounces"
            min={1}
            max={16}
            step={1}
            value={maxCrossings}
            display={String(Math.round(maxCrossings))}
            onChange={(value) => setMaxCrossings(Math.round(value))}
            presets={[
              { label: '2', value: 2 },
              { label: '6', value: 6 },
              { label: '12', value: 12 },
            ]}
          />
        </div>

        <div className="flex flex-col gap-1 border-t border-fuchsia-500/20 pt-2.5">
          <SliderRow
            label="Render scale"
            min={0.35}
            max={1}
            step={0.05}
            value={quality}
            display={`${Math.round(quality * 100)}%`}
            onChange={setQuality}
          />
          <p className="text-[10px] leading-snug text-slate-500">
            Resolution cap for the ray integrator. Pixel ratio is also limited to 1.5. Rays bend with the throat distance and escape outbound past the cluster boundary.
          </p>
        </div>

        <button
          type="button"
          onClick={() => centerDirectRef.current()}
          className="flex items-center justify-center gap-2 rounded-lg border border-fuchsia-400/80 bg-gradient-to-r from-fuchsia-950/90 via-purple-950/90 to-fuchsia-950/90 py-2 px-2.5 text-xs font-semibold text-fuchsia-100 shadow-md transition hover:border-fuchsia-300 hover:from-fuchsia-900 hover:to-purple-900 active:scale-[0.98]"
          title="Center at exact height and horizontal position directly facing the wormhole (Hotkey: H or Z)"
        >
          <Crosshair className="w-3.5 h-3.5 text-fuchsia-300" />
          <span>Center Level on Wormhole</span>
          <kbd className="ml-auto rounded bg-fuchsia-950/90 border border-fuchsia-600/50 px-1 font-mono text-[9px] text-fuchsia-300">
            H
          </kbd>
        </button>

        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            onClick={() => frameRef.current()}
            className="rounded border border-fuchsia-500/40 bg-fuchsia-950/40 py-1.5 text-[11px] font-semibold text-fuchsia-100 transition hover:bg-fuchsia-900/50"
          >
            Frame throat
          </button>
          <button
            type="button"
            onClick={resetAll}
            className="rounded border border-slate-700 bg-slate-900/80 py-1.5 text-[11px] font-semibold text-slate-200 transition hover:bg-slate-800"
          >
            Reset
          </button>
        </div>

        <p className="font-mono text-[10px] leading-relaxed text-slate-500 tabular-nums">
          {singleWormhole ? (
            <>
              uMouth ({singleWormhole.mouthA.position[0].toFixed(2)},{' '}
              {singleWormhole.mouthA.position[1].toFixed(2)},{' '}
              {singleWormhole.mouthA.position[2].toFixed(2)}, {singleWormhole.radius.toFixed(2)})
              <br />
              shape {WORMHOLE_SHAPES[singleWormhole.shape].label.toLowerCase()} (
              {singleWormhole.scale[0].toFixed(2)}×{singleWormhole.scale[1].toFixed(2)}×
              {singleWormhole.scale[2].toFixed(2)}) · g {singleWormhole.gravityRatio.toFixed(2)} · steps{' '}
              {Math.round(maxSteps)} · bounces {Math.round(maxCrossings)}
            </>
          ) : (
            <>
              multiverse {graph.universes.length}U · {graph.wormholes.length}W · active{' '}
              {activeUniverse?.name ?? '—'}
              <br />
              steps {Math.round(maxSteps)} · bounces {Math.round(maxCrossings)}
            </>
          )}
        </p>
      </div>

      {/* Stage 3 Multiverse Architect Workspace Drawer */}
      <MultiverseWorkspaceDrawer
        open={workspaceOpen && showControlsOverlay}
        onClose={() => setWorkspaceOpen(false)}
        graph={graph}
        onChangeGraph={handleMutateGraph}
        activePreset={activePreset}
        onLoadPreset={handleSelectPreset}
        activeUniverseIndex={safeRoomIndex}
        onTeleportToUniverse={handleTeleportToUniverse}
        onFrameMouth={(uIdx, pos, rad) => frameMouthRef.current(uIdx, pos, rad)}
        selection={workspaceSelection}
        onSelect={setWorkspaceSelection}
        overlaps={overlaps}
        showOverlay3d={showOverlay3d}
        onToggleOverlay3d={() => setShowOverlay3d((prev) => !prev)}
        skyMetaById={skyMetaById}
        onUploadUniverseSky={(id, file) => uploadUniverseSkyRef.current(id, file)}
        onClearUniverseSky={(id) => clearUniverseSkyRef.current(id)}
        onDeleteUniverseCleanup={(id) => clearUniverseSkyRef.current(id)}
        keyframes={keyframes}
        onImportGraph={handleImportGraph}
      />

      {!isUiVisible && !isRenderingMovie && (
        <button
          type="button"
          onClick={toggleUi}
          className="absolute top-4 right-4 z-40 flex items-center gap-2 rounded-full border border-fuchsia-500/50 bg-slate-950/92 px-3.5 py-1.5 text-xs font-semibold text-fuchsia-200 shadow-2xl transition hover:bg-slate-900 hover:border-fuchsia-400 hover:scale-105 active:scale-95"
          title="Restore user interface (or press Shift)"
        >
          <Eye className="w-3.5 h-3.5 text-fuchsia-400" />
          <span>Show UI</span>
          <kbd className="rounded bg-slate-800/90 px-1.5 py-0.5 font-mono text-[10px] text-fuchsia-300 border border-slate-700">
            Shift
          </kbd>
        </button>
      )}

      {/* Offline Movie Render Progress HUD */}
      {renderStatus?.active && (
        <div className="absolute right-4 top-4 z-40 flex w-72 flex-col gap-2 rounded-xl border border-fuchsia-400/60 bg-slate-950/95 p-3 shadow-2xl">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-fuchsia-200">
              <Clapperboard className="h-3.5 w-3.5 text-fuchsia-400" />
              Rendering {renderStatus.format.toUpperCase()}
            </span>
            <span className="font-mono text-xs font-semibold text-fuchsia-100 tabular-nums">
              {renderPct}%
            </span>
          </div>

          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full bg-gradient-to-r from-fuchsia-500 to-purple-400 transition-[width] duration-100"
              style={{ width: `${renderPct}%` }}
            />
          </div>

          <div className="flex items-center justify-between font-mono text-[10px] text-slate-300 tabular-nums">
            <span>
              Frame {renderStatus.frame} / {renderStatus.total}
            </span>
            {renderStatus.width > 0 && (
              <span className="text-slate-400">
                {renderStatus.width}×{renderStatus.height}
              </span>
            )}
          </div>

          <div className="flex items-center justify-between gap-2 pt-0.5">
            <span className="text-[9px] text-slate-400">Clean shader pass · UI excluded</span>
            <button
              type="button"
              onClick={handleCancelRender}
              className="flex items-center gap-1 rounded border border-red-500/50 bg-red-950/70 px-2 py-1 text-[10px] font-semibold text-red-200 transition hover:bg-red-900/80"
            >
              <Square className="h-2.5 w-2.5 fill-current" />
              <span>Cancel</span>
            </button>
          </div>
        </div>
      )}

      {/* Bottom Universe Swatch & Telemetry Legend */}
      <div
        className={`pointer-events-none absolute bottom-4 left-1/2 z-10 hidden max-w-[calc(100%-2rem)] -translate-x-1/2 flex-wrap items-center justify-center gap-x-3 gap-y-1 rounded-full border border-fuchsia-500/25 bg-slate-950/90 px-4 py-1.5 text-[11px] text-slate-300 shadow-xl transition-all duration-300 sm:flex ${
          showControlsOverlay && !workspaceOpen ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        {graph.universes.slice(0, 6).map((u, idx) => (
          <span
            key={u.id}
            className={`inline-flex items-center gap-1.5 ${
              idx === safeRoomIndex ? 'font-semibold text-white' : 'text-slate-400'
            }`}
          >
            <span
              className="h-2 w-2 rounded-full border border-white/20 shadow-sm"
              style={{ backgroundColor: u.gridVerticalColor }}
            />
            <span
              className="h-2 w-2 rounded-full border border-white/20 shadow-sm"
              style={{ backgroundColor: u.gridHorizontalColor }}
            />
            {u.id}
          </span>
        ))}
        {graph.universes.length > 6 && (
          <span className="font-mono text-[10px] text-slate-400">
            +{graph.universes.length - 6}
          </span>
        )}
        <span className="text-slate-400">
          <span className="font-mono text-fuchsia-200">WASD</span> move ·{' '}
          <span className="font-mono text-fuchsia-200">Space/C</span> up/down ·{' '}
          <span className="font-mono text-fuchsia-200">Q/E</span> turn ·{' '}
          <span className="font-mono text-fuchsia-200">H</span> center ·{' '}
          <span className="font-mono text-fuchsia-200">Shift</span> UI
        </span>
        <span className="font-mono text-slate-400 tabular-nums">
          cam {hud.x.toFixed(1)}, {hud.y.toFixed(1)}, {hud.z.toFixed(1)} · Δ {clearance.toFixed(2)}
        </span>
      </div>

      {shaderError && (
        <div className="absolute bottom-16 right-4 z-20 max-w-sm rounded-lg border border-red-500/50 bg-red-950/90 p-3 text-[11px] text-red-100">
          {shaderError}
        </div>
      )}
    </div>
  );
};

interface UniverseSourceProps {
  title: string;
  detail: string;
  mode: UniverseSkyMode;
  gridConfig: GridConfig;
  onGridConfigChange: React.Dispatch<React.SetStateAction<GridConfig>>;
  lonOverrides?: GridLineColorOverrides;
  latOverrides?: GridLineColorOverrides;
  onLonOverridesChange: (next: GridLineColorOverrides | undefined) => void;
  onLatOverridesChange: (next: GridLineColorOverrides | undefined) => void;
  fileLabel: string | null;
  hasTexture?: boolean;
  isHdr: boolean;
  exposure: number;
  onExposureChange: (ev: number) => void;
  isLoading: boolean;
  errorMessage: string | null;
  onMode: (mode: UniverseSkyMode) => void;
  onUpload: (file: File) => void;
  onClear: () => void;
}

const PALETTES = [
  { v: '#ff0000', h: '#0000ff', name: 'Red/Blue' },
  { v: '#00ffff', h: '#ff00ff', name: 'Cyan/Magenta' },
  { v: '#00ff66', h: '#006622', name: 'Emerald' },
  { v: '#ffaa00', h: '#ff3300', name: 'Solar' },
  { v: '#b026ff', h: '#4d0099', name: 'Violet' },
  { v: '#ffffff', h: '#555555', name: 'Mono' },
];

const UniverseSource: React.FC<UniverseSourceProps> = ({
  title,
  detail,
  mode,
  gridConfig,
  onGridConfigChange,
  lonOverrides,
  latOverrides,
  onLonOverridesChange,
  onLatOverridesChange,
  fileLabel,
  hasTexture = false,
  isHdr,
  exposure,
  onExposureChange,
  isLoading,
  errorMessage,
  onMode,
  onUpload,
  onClear,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
  <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-2.5">
    <div className="mb-1.5 flex items-center justify-between">
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] font-bold text-slate-100">{title}</span>
        <div className="flex items-center gap-1">
          <span
            className="h-2 w-2 rounded-full border border-white/20 shadow-sm"
            style={{ backgroundColor: gridConfig.verticalColor }}
            title={`Longitude: ${gridConfig.verticalColor}`}
          />
          <span
            className="h-2 w-2 rounded-full border border-white/20 shadow-sm"
            style={{ backgroundColor: gridConfig.horizontalColor }}
            title={`Latitude: ${gridConfig.horizontalColor}`}
          />
        </div>
        {isHdr && fileLabel && (
          <span className="rounded border border-amber-400/50 bg-amber-950/70 px-1 py-0.5 font-mono text-[8px] font-bold uppercase tracking-wider text-amber-200">
            HDRI
          </span>
        )}
      </div>
      <span className="font-mono text-[10px] text-slate-400 tabular-nums">{detail}</span>
    </div>

    <div className="grid grid-cols-2 gap-1 mb-2">
      <button
        type="button"
        onClick={() => onMode('grid')}
        className={`rounded border py-1 text-[10px] font-semibold transition ${
          mode === 'grid'
            ? 'border-fuchsia-400/70 bg-fuchsia-950/70 text-fuchsia-100'
            : 'border-slate-700 bg-slate-900/60 text-slate-400 hover:text-slate-200'
        }`}
      >
        Grid Pattern
      </button>
      <button
        type="button"
        onClick={() => {
          onMode('texture');
          if (!fileLabel && !hasTexture && !isLoading) {
            fileInputRef.current?.click();
          }
        }}
        className={`rounded border py-1 text-[10px] font-semibold transition ${
          mode === 'texture'
            ? 'border-fuchsia-400/70 bg-fuchsia-950/70 text-fuchsia-100'
            : 'border-slate-700 bg-slate-900/60 text-slate-400 hover:text-slate-200'
        }`}
      >
        Image / HDRI
      </button>
    </div>

    {mode === 'texture' && isHdr && fileLabel && (
      <div className="mb-2 flex flex-col gap-1.5 rounded-md border border-amber-500/30 bg-amber-950/20 p-2">
        <div className="flex items-center justify-between text-[10px]">
          <span className="font-semibold text-amber-100">HDRI Exposure</span>
          <span className="font-mono text-amber-200 tabular-nums">
            {exposure >= 0 ? `+${exposure.toFixed(1)}` : exposure.toFixed(1)} EV
          </span>
        </div>
        <input
          type="range"
          min={-4}
          max={4}
          step={0.1}
          value={exposure}
          aria-label={`${title} HDRI exposure`}
          onChange={(e) => onExposureChange(parseFloat(e.target.value))}
          className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-amber-400"
        />
        <div className="flex items-center justify-between gap-1">
          <span className="text-[9px] text-slate-500">ACES Filmic</span>
          <div className="flex items-center gap-1">
            {[-2, -1, 0, 1, 2].map((ev) => (
              <button
                key={ev}
                type="button"
                onClick={() => onExposureChange(ev)}
                className={`rounded border px-1.5 py-0.5 font-mono text-[9px] transition ${
                  Math.abs(exposure - ev) < 0.05
                    ? 'border-amber-400/80 bg-amber-950/80 text-amber-100'
                    : 'border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200'
                }`}
              >
                {ev > 0 ? `+${ev}` : `${ev}`}
              </button>
            ))}
          </div>
        </div>
      </div>
    )}

    {mode === 'grid' && (
      <div className="flex flex-col gap-2 rounded-md border border-slate-800/80 bg-slate-900/40 p-2">
        <div className="flex flex-col gap-1.5">
          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Grid Colors</span>
          <div className="grid grid-cols-2 gap-1.5">
            <label className="flex items-center justify-between rounded border border-slate-800 bg-slate-950/90 px-2 py-1 cursor-pointer hover:border-slate-700 transition">
              <div className="flex items-center gap-1.5 min-w-0">
                <div className="relative h-4 w-4 shrink-0 rounded border border-white/30 overflow-hidden shadow-sm">
                  <div className="h-full w-full" style={{ backgroundColor: gridConfig.verticalColor }} />
                  <input
                    type="color"
                    value={gridConfig.verticalColor}
                    onChange={(e) =>
                      onGridConfigChange((prev) => ({ ...prev, verticalColor: e.target.value }))
                    }
                    className="absolute inset-0 h-full w-full opacity-0 cursor-pointer"
                  />
                </div>
                <span className="text-[10px] text-slate-300 truncate">Longitude</span>
              </div>
              <span className="font-mono text-[9px] uppercase text-fuchsia-300 ml-1">
                {gridConfig.verticalColor}
              </span>
            </label>

            <label className="flex items-center justify-between rounded border border-slate-800 bg-slate-950/90 px-2 py-1 cursor-pointer hover:border-slate-700 transition">
              <div className="flex items-center gap-1.5 min-w-0">
                <div className="relative h-4 w-4 shrink-0 rounded border border-white/30 overflow-hidden shadow-sm">
                  <div className="h-full w-full" style={{ backgroundColor: gridConfig.horizontalColor }} />
                  <input
                    type="color"
                    value={gridConfig.horizontalColor}
                    onChange={(e) =>
                      onGridConfigChange((prev) => ({ ...prev, horizontalColor: e.target.value }))
                    }
                    className="absolute inset-0 h-full w-full opacity-0 cursor-pointer"
                  />
                </div>
                <span className="text-[10px] text-slate-300 truncate">Latitude</span>
              </div>
              <span className="font-mono text-[9px] uppercase text-fuchsia-300 ml-1">
                {gridConfig.horizontalColor}
              </span>
            </label>
          </div>

          <div className="flex items-center gap-1 flex-wrap pt-0.5">
            <span className="text-[9px] text-slate-500 mr-0.5">Palettes:</span>
            {PALETTES.map((pal, idx) => (
              <button
                key={idx}
                type="button"
                title={pal.name}
                onClick={() =>
                  onGridConfigChange((prev) => ({
                    ...prev,
                    verticalColor: pal.v,
                    horizontalColor: pal.h,
                  }))
                }
                className="flex h-3.5 w-6 overflow-hidden rounded border border-slate-700/80 hover:scale-110 transition shadow-sm"
              >
                <div className="h-full w-1/2" style={{ backgroundColor: pal.v }} />
                <div className="h-full w-1/2" style={{ backgroundColor: pal.h }} />
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1.5 border-t border-slate-800/80 pt-1.5">
          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Grid Density</span>

          <div className="flex flex-col gap-0.5">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-slate-300 font-medium">Longitude lines</span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() =>
                    onGridConfigChange((prev) => ({
                      ...prev,
                      cols: Math.max(4, prev.cols - 2),
                    }))
                  }
                  className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
                >
                  -
                </button>
                <span className="font-mono text-fuchsia-200 w-6 text-center tabular-nums">{gridConfig.cols}</span>
                <button
                  type="button"
                  onClick={() =>
                    onGridConfigChange((prev) => ({
                      ...prev,
                      cols: Math.min(72, prev.cols + 2),
                    }))
                  }
                  className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
                >
                  +
                </button>
              </div>
            </div>
            <input
              type="range"
              min={4}
              max={72}
              step={2}
              value={gridConfig.cols}
              onChange={(e) => {
                const val = Number(e.target.value);
                onGridConfigChange((prev) => ({ ...prev, cols: val }));
              }}
              className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-fuchsia-400"
            />
          </div>

          <div className="flex flex-col gap-0.5">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-slate-300 font-medium">Latitude lines</span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() =>
                    onGridConfigChange((prev) => ({
                      ...prev,
                      rows: Math.max(2, prev.rows - 1),
                    }))
                  }
                  className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
                >
                  -
                </button>
                <span className="font-mono text-fuchsia-200 w-6 text-center tabular-nums">{gridConfig.rows}</span>
                <button
                  type="button"
                  onClick={() =>
                    onGridConfigChange((prev) => ({
                      ...prev,
                      rows: Math.min(36, prev.rows + 1),
                    }))
                  }
                  className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
                >
                  +
                </button>
              </div>
            </div>
            <input
              type="range"
              min={2}
              max={36}
              step={1}
              value={gridConfig.rows}
              onChange={(e) => {
                const val = Number(e.target.value);
                onGridConfigChange((prev) => ({ ...prev, rows: val }));
              }}
              className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-fuchsia-400"
            />
          </div>

          <div className="flex items-center gap-1 pt-0.5">
            <span className="text-[9px] text-slate-500 mr-0.5">Presets:</span>
            {[
              { label: '8×4', c: 8, r: 4 },
              { label: '18×9', c: 18, r: 9 },
              { label: '36×18', c: 36, r: 18 },
              { label: '64×32', c: 64, r: 32 },
            ].map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() =>
                  onGridConfigChange((prev) => ({
                    ...prev,
                    cols: p.c,
                    rows: p.r,
                  }))
                }
                className={`rounded border px-1.5 py-0.5 font-mono text-[9px] transition ${
                  gridConfig.cols === p.c && gridConfig.rows === p.r
                    ? 'border-fuchsia-400 bg-fuchsia-950 text-fuchsia-100'
                    : 'border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <GridLineOverridesEditor
          cols={gridConfig.cols}
          rows={gridConfig.rows}
          lonOverrides={lonOverrides}
          latOverrides={latOverrides}
          onChangeLonOverrides={onLonOverridesChange}
          onChangeLatOverrides={onLatOverridesChange}
        />
      </div>
    )}

    <div className="flex flex-col gap-1.5 rounded-md border border-slate-800/80 bg-slate-900/50 p-2 mt-2">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-bold uppercase tracking-wider text-slate-300">
          Image / Sky Rotation
        </span>
        <button
          type="button"
          onClick={() =>
            onGridConfigChange((prev) => ({
              ...prev,
              rotationY: 0,
              rotationX: 0,
            }))
          }
          className="rounded border border-fuchsia-500/40 bg-fuchsia-950/60 px-1.5 py-0.5 font-mono text-[9px] text-fuchsia-200 hover:bg-fuchsia-900/60 transition"
          title="Align image center directly with the wormhole (0° yaw, 0° pitch)"
        >
          Center on Wormhole
        </button>
      </div>

      <div className="flex flex-col gap-0.5">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-slate-300 font-medium">Horizontal Yaw (Pan)</span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() =>
                onGridConfigChange((prev) => {
                  let next = prev.rotationY - 5;
                  if (next < -180) next += 360;
                  return { ...prev, rotationY: next };
                })
              }
              className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
              title="Rotate 5° left"
            >
              -
            </button>
            <span className="font-mono text-fuchsia-200 w-12 text-center text-[10px] tabular-nums">
              {gridConfig.rotationY > 0 ? `+${Math.round(gridConfig.rotationY)}°` : `${Math.round(gridConfig.rotationY)}°`}
            </span>
            <button
              type="button"
              onClick={() =>
                onGridConfigChange((prev) => {
                  let next = prev.rotationY + 5;
                  if (next > 180) next -= 360;
                  return { ...prev, rotationY: next };
                })
              }
              className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
              title="Rotate 5° right"
            >
              +
            </button>
          </div>
        </div>
        <input
          type="range"
          min={-180}
          max={180}
          step={1}
          value={gridConfig.rotationY}
          onChange={(e) => {
            const val = Number(e.target.value);
            onGridConfigChange((prev) => ({ ...prev, rotationY: val }));
          }}
          className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-fuchsia-400"
        />
        <div className="flex items-center justify-between gap-1 pt-0.5">
          <span className="text-[9px] text-slate-500">Presets:</span>
          {[
            { label: 'Center (0°)', deg: 0 },
            { label: 'Left (-90°)', deg: -90 },
            { label: 'Right (+90°)', deg: 90 },
            { label: 'Back (180°)', deg: 180 },
          ].map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() =>
                onGridConfigChange((prev) => ({
                  ...prev,
                  rotationY: p.deg,
                }))
              }
              className={`rounded border px-1.5 py-0.5 font-mono text-[9px] transition ${
                Math.round(gridConfig.rotationY) === p.deg
                  ? 'border-fuchsia-400 bg-fuchsia-950 text-fuchsia-100'
                  : 'border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-0.5 pt-1 border-t border-slate-800/80">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-slate-300 font-medium">Vertical Pitch (Tilt)</span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() =>
                onGridConfigChange((prev) => {
                  let next = prev.rotationX - 5;
                  if (next < -180) next += 360;
                  return { ...prev, rotationX: next };
                })
              }
              className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
              title="Tilt 5° down"
            >
              -
            </button>
            <span className="font-mono text-fuchsia-200 w-12 text-center text-[10px] tabular-nums">
              {gridConfig.rotationX > 0 ? `+${Math.round(gridConfig.rotationX)}°` : `${Math.round(gridConfig.rotationX)}°`}
            </span>
            <button
              type="button"
              onClick={() =>
                onGridConfigChange((prev) => {
                  let next = prev.rotationX + 5;
                  if (next > 180) next -= 360;
                  return { ...prev, rotationX: next };
                })
              }
              className="h-4 w-4 rounded bg-slate-800 text-[10px] text-slate-300 hover:bg-slate-700 flex items-center justify-center font-bold"
              title="Tilt 5° up"
            >
              +
            </button>
          </div>
        </div>
        <input
          type="range"
          min={-180}
          max={180}
          step={1}
          value={gridConfig.rotationX}
          onChange={(e) => {
            const val = Number(e.target.value);
            onGridConfigChange((prev) => ({ ...prev, rotationX: val }));
          }}
          className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-fuchsia-400"
        />
        <div className="flex items-center justify-between gap-1 pt-0.5">
          <span className="text-[9px] text-slate-500">Presets:</span>
          {[
            { label: 'Level (0°)', deg: 0 },
            { label: 'Down (-90°)', deg: -90 },
            { label: 'Up (+90°)', deg: 90 },
            { label: 'Flip (180°)', deg: 180 },
          ].map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() =>
                onGridConfigChange((prev) => ({
                  ...prev,
                  rotationX: p.deg,
                }))
              }
              className={`rounded border px-1.5 py-0.5 font-mono text-[9px] transition ${
                Math.round(gridConfig.rotationX) === p.deg
                  ? 'border-fuchsia-400 bg-fuchsia-950 text-fuchsia-100'
                  : 'border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
    </div>

    <div className="mt-2 flex items-center gap-1.5">
      <label className="cursor-pointer whitespace-nowrap rounded border border-fuchsia-500/40 bg-fuchsia-950/40 px-2 py-1 text-[10px] font-semibold text-fuchsia-100 transition hover:border-fuchsia-400/70 hover:bg-fuchsia-900/50">
        {isLoading ? 'Loading…' : 'Upload Image / HDRI'}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.hdr,.exr,.pic,.rgbe"
          className="hidden"
          disabled={isLoading}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = '';
          }}
        />
      </label>
      <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-500" title={fileLabel ?? undefined}>
        {isLoading
          ? 'Decoding sky map…'
          : (fileLabel ?? (mode === 'texture' ? '.hdr, .exr, or image' : 'Procedural lat/long'))}
      </span>
      {fileLabel && !isLoading && (
        <button type="button" onClick={onClear} className="text-[10px] text-slate-400 hover:text-slate-200">
          Clear
        </button>
      )}
    </div>
    {errorMessage && (
      <p className="mt-1.5 rounded border border-red-500/40 bg-red-950/50 px-2 py-1 text-[10px] leading-snug text-red-200">
        {errorMessage}
      </p>
    )}
  </div>
  );
};
