import {
  DEFAULT_SHAPE_SCALE,
  glueDirectionThroughThroat,
  parseWormholeShape,
  throatCrossing,
  wormholeSdf,
  wormholeShapeIdToName,
  type ShapeScale,
  type WormholeShapeId,
  type WormholeShapeName,
} from './wormholeSdf.ts';

export type UniverseSkyMode = 'grid' | 'texture';

export type Vec3 = readonly [number, number, number];
export type Quat4 = readonly [number, number, number, number];

export const IDENTITY_QUATERNION: Quat4 = [0, 0, 0, 1];

/**
 * GPU DataTexture row widths (in RGBA32F texels) for the WebGL2 raymarching pipeline.
 *
 * uUniverseData (width = 5, height = max(1, universeCount), RGBA32F):
 *   texel(0, u) = (skyMode, textureLayer, isHdr, linearExposure)
 *   texel(1, u) = (gridCols, gridRows, rotYawRad, rotPitchRad)
 *   texel(2, u) = (gridLonR, gridLonG, gridLonB, mouthOffset)
 *   texel(3, u) = (gridLatR, gridLatG, gridLatB, mouthCount)
 *   texel(4, u) = (clusterCenterX, clusterCenterY, clusterCenterZ, clusterRadius)
 *
 * uMouthData (width = 4, height = max(1, mouthCount), RGBA32F):
 *   texel(0, m) = (centerX, centerY, centerZ, radius)
 *   texel(1, m) = (quatX, quatY, quatZ, quatW)
 *   texel(2, m) = (scaleX, scaleY, scaleZ, shapeId)
 *   texel(3, m) = (gravityRatio, universeIndex, destUniverseIndex, destMouthIndex)
 */
export const UNIVERSE_TEXELS_PER_ROW = 5;
export const MOUTH_TEXELS_PER_ROW = 4;

/**
 * Per-universe grid line color override lookup table (RGBA8 DataTexture):
 *   width = GRID_OVERRIDE_TEX_WIDTH (112), height = max(1, universeCount)
 *   x in [0, 71]   -> longitude line index 0..71 (a = 255 if overridden, 0 otherwise)
 *   x in [72, 108] -> latitude  line index 0..36 (a = 255 if overridden, 0 otherwise)
 */
export const GRID_OVERRIDE_MAX_LON = 72;
export const GRID_OVERRIDE_MAX_LAT = 37;
export const GRID_OVERRIDE_LAT_OFFSET = 72;
export const GRID_OVERRIDE_TEX_WIDTH = 112;

export type GridLineColorOverrides = Record<number, string>;

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

export interface UniverseNode {
  id: string;
  name: string;
  skyMode: UniverseSkyMode;
  gridVerticalColor: string;
  gridHorizontalColor: string;
  gridCols: number;
  gridRows: number;
  /** Sky yaw rotation in degrees (-180..180). */
  rotationY: number;
  /** Sky pitch rotation in degrees (-180..180). */
  rotationX: number;
  isHdr: boolean;
  /** Exposure stop offset in EV; linear gain is 2^exposureEv. */
  exposureEv: number;
  /** Layer index in sampler2DArray uSkyArray, or -1 if no custom sky texture is loaded. */
  textureLayer: number;
  /** Optional per-line hex color overrides keyed by integer longitude line index (0..gridCols-1). */
  gridLonOverrides?: GridLineColorOverrides;
  /** Optional per-line hex color overrides keyed by integer latitude line index (1..gridRows-1). */
  gridLatOverrides?: GridLineColorOverrides;
}

export interface WormholeMouthPlacement {
  universeId: string;
  position: [number, number, number];
  /** Orientation quaternion [x, y, z, w] of this mouth in its universe. */
  quaternion: [number, number, number, number];
}

/**
 * A traversable wormhole edge bridging two mouths (mouthA <-> mouthB).
 * Both ends always share the same throat shape, radius, scale, and gravity ratio
 * so surface gluing across the two universes is isometric and distortion-free.
 */
export interface WormholeEdge {
  id: string;
  name: string;
  shape: WormholeShapeId;
  radius: number;
  scale: [number, number, number];
  gravityRatio: number;
  mouthA: WormholeMouthPlacement;
  mouthB: WormholeMouthPlacement;
}

export interface MultiverseGraph {
  universes: UniverseNode[];
  wormholes: WormholeEdge[];
}

export interface UniverseMouthSlice {
  offset: number;
  count: number;
  clusterCenter: [number, number, number];
  clusterRadius: number;
}

export interface PackedMouthMetadata {
  wormholeId: string;
  end: 'A' | 'B';
  universeIndex: number;
  destUniverseIndex: number;
  destMouthIndex: number;
}

export interface PackedMultiverseBuffers {
  universeData: Float32Array;
  universeWidth: typeof UNIVERSE_TEXELS_PER_ROW;
  universeHeight: number;
  universeCount: number;
  mouthData: Float32Array;
  mouthWidth: typeof MOUTH_TEXELS_PER_ROW;
  mouthHeight: number;
  mouthCount: number;
  gridOverrideData: Uint8Array;
  gridOverrideWidth: typeof GRID_OVERRIDE_TEX_WIDTH;
  gridOverrideHeight: number;
  universeIndexById: Map<string, number>;
  universeMouthSlices: UniverseMouthSlice[];
  mouthLookup: PackedMouthMetadata[];
}

export interface PackBufferReuse {
  universeData?: Float32Array;
  mouthData?: Float32Array;
  gridOverrideData?: Uint8Array;
}

/**
 * Parses a #RGB or #RRGGBB hex string into [r, g, b] sRGB 0..1 components.
 * Matches the shader's direct sRGB output pipeline.
 */
export function parseHexColorSrgb(
  hex: string,
  fallback: readonly [number, number, number] = [1, 1, 1]
): [number, number, number] {
  const raw = hex.trim().replace('#', '');
  const full = raw.length === 3 ? raw.replace(/./g, (ch) => ch + ch) : raw;
  const value = Number.parseInt(full, 16);
  if (!Number.isFinite(value) || full.length !== 6) {
    return [fallback[0], fallback[1], fallback[2]];
  }
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

export function normalizeQuaternion(q: Quat4): [number, number, number, number] {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(len) || len < 1e-8) {
    return [0, 0, 0, 1];
  }
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

/**
 * Rotates vector v by unit quaternion q = [x, y, z, w].
 * Matches GLSL quatRotate(quat, dir) bit-for-bit.
 */
export function quatRotateVec3(q: Quat4, v: Vec3): [number, number, number] {
  const qx = q[0];
  const qy = q[1];
  const qz = q[2];
  const qw = q[3];
  const vx = v[0];
  const vy = v[1];
  const vz = v[2];

  // t = cross(q.xyz, v) + q.w * v
  const tx = qy * vz - qz * vy + qw * vx;
  const ty = qz * vx - qx * vz + qw * vy;
  const tz = qx * vy - qy * vx + qw * vz;

  // v + 2 * cross(q.xyz, t)
  return [
    vx + 2 * (qy * tz - qz * ty),
    vy + 2 * (qz * tx - qx * tz),
    vz + 2 * (qx * ty - qy * tx),
  ];
}

/**
 * Rotates vector v by the inverse (conjugate) of unit quaternion q = [x, y, z, w].
 * Matches GLSL quatRotateInv(quat, dir) bit-for-bit.
 */
export function quatRotateInvVec3(q: Quat4, v: Vec3): [number, number, number] {
  return quatRotateVec3([-q[0], -q[1], -q[2], q[3]], v);
}

/**
 * Builds a unit quaternion from intrinsic Y (yaw), X (pitch), Z (roll) angles in degrees.
 */
export function quaternionFromEulerDeg(
  yawDeg: number,
  pitchDeg: number = 0,
  rollDeg: number = 0
): [number, number, number, number] {
  const hy = (yawDeg * DEG2RAD) * 0.5;
  const hx = (pitchDeg * DEG2RAD) * 0.5;
  const hz = (rollDeg * DEG2RAD) * 0.5;

  const cy = Math.cos(hy);
  const sy = Math.sin(hy);
  const cx = Math.cos(hx);
  const sx = Math.sin(hx);
  const cz = Math.cos(hz);
  const sz = Math.sin(hz);

  // q = qY * qX * qZ
  const x = cy * sx * cz + sy * cx * sz;
  const y = sy * cx * cz - cy * sx * sz;
  const z = cy * cx * sz - sy * sx * cz;
  const w = cy * cx * cz + sy * sx * sz;
  return normalizeQuaternion([x, y, z, w]);
}

/**
 * Inverse of quaternionFromEulerDeg: extracts intrinsic Y (yaw), X (pitch), Z (roll)
 * angles in degrees from a unit quaternion q = [x, y, z, w].
 */
export function eulerDegFromQuaternion(q: Quat4): [number, number, number] {
  const [x, y, z, w] = normalizeQuaternion(q);
  // Rotation matrix entries for intrinsic YXZ (R = Ry * Rx * Rz):
  // m12 = -sin(pitch) = 2(yz - wx)
  const sinPitch = Math.max(-1, Math.min(1, -(2 * (y * z - w * x))));
  const pitch = Math.asin(sinPitch);

  let yaw: number;
  let roll: number;

  if (Math.abs(sinPitch) < 0.99999) {
    // m02 = sin(yaw)*cos(pitch), m22 = cos(yaw)*cos(pitch)
    yaw = Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y));
    // m10 = cos(pitch)*sin(roll), m11 = cos(pitch)*cos(roll)
    roll = Math.atan2(2 * (x * y + w * z), 1 - 2 * (x * x + z * z));
  } else {
    // Gimbal lock: set roll = 0 and fold angle into yaw
    yaw = Math.atan2(-(2 * (x * z - w * y)), 1 - 2 * (y * y + z * z));
    roll = 0;
  }

  const normalizeAngle = (deg: number) => {
    const rounded = Math.abs(deg) < 1e-4 ? 0 : deg;
    return rounded;
  };

  return [
    normalizeAngle(yaw * RAD2DEG),
    normalizeAngle(pitch * RAD2DEG),
    normalizeAngle(roll * RAD2DEG),
  ];
}

/**
 * Evaluates a mouth's signed distance field at a world-space point by
 * transforming into the mouth's local position + orientation frame.
 */
export function evaluateMouthSdfWorld(
  mouth: WormholeMouthPlacement,
  shape: WormholeShapeId,
  radius: number,
  scale: ShapeScale,
  worldPos: Vec3
): number {
  const q = normalizeQuaternion(mouth.quaternion);
  const local = quatRotateInvVec3(q, [
    worldPos[0] - mouth.position[0],
    worldPos[1] - mouth.position[1],
    worldPos[2] - mouth.position[2],
  ]);
  return wormholeSdf(shape, local[0], local[1], local[2], radius, scale);
}

export interface NearestMouthMatch {
  wormholeIndex: number;
  wormhole: WormholeEdge;
  end: 'A' | 'B';
  mouth: WormholeMouthPlacement;
  partnerMouth: WormholeMouthPlacement;
  destUniverseIndex: number;
  gap: number;
  distance: number;
  effectiveR: number;
  maxExtent: number;
}

/**
 * Finds the nearest wormhole mouth (by signed surface gap) located in `universeIndex`.
 * Returns null if the universe has no mouths.
 */
export function findNearestMouthInUniverse(
  graph: MultiverseGraph,
  universeIndex: number,
  worldPos: Vec3
): NearestMouthMatch | null {
  const universe = graph.universes[universeIndex];
  if (!universe) return null;

  const universeIndexById = new Map<string, number>();
  for (let i = 0; i < graph.universes.length; i++) {
    universeIndexById.set(graph.universes[i].id, i);
  }

  let best: NearestMouthMatch | null = null;

  for (let wIdx = 0; wIdx < graph.wormholes.length; wIdx++) {
    const wh = graph.wormholes[wIdx];
    const ends: Array<{ end: 'A' | 'B'; mouth: WormholeMouthPlacement; partner: WormholeMouthPlacement }> = [
      { end: 'A', mouth: wh.mouthA, partner: wh.mouthB },
      { end: 'B', mouth: wh.mouthB, partner: wh.mouthA },
    ];

    for (const candidate of ends) {
      if (candidate.mouth.universeId !== universe.id) continue;
      const destUniverseIndex = universeIndexById.get(candidate.partner.universeId);
      if (destUniverseIndex === undefined) continue;

      const safeRadius = Math.max(wh.radius, 0.001);
      const sx = Math.max(wh.scale[0], 0.05);
      const sy = Math.max(wh.scale[1], 0.05);
      const sz = Math.max(wh.scale[2], 0.05);
      const safeScale: ShapeScale = [sx, sy, sz];
      const effectiveR = Math.max(safeRadius * Math.min(sx, sy, sz), 0.001);
      const maxExtent = safeRadius * Math.max(sx, sy, sz);

      const gap = evaluateMouthSdfWorld(candidate.mouth, wh.shape, safeRadius, safeScale, worldPos);
      const distance = Math.hypot(
        worldPos[0] - candidate.mouth.position[0],
        worldPos[1] - candidate.mouth.position[1],
        worldPos[2] - candidate.mouth.position[2]
      );

      if (!best || gap < best.gap) {
        best = {
          wormholeIndex: wIdx,
          wormhole: wh,
          end: candidate.end,
          mouth: candidate.mouth,
          partnerMouth: candidate.partner,
          destUniverseIndex,
          gap,
          distance,
          effectiveR,
          maxExtent,
        };
      }
    }
  }

  return best;
}

export interface WorldMouthCrossingResult {
  exitPosition: [number, number, number];
  exitForward: [number, number, number];
  destUniverseIndex: number;
}

/**
 * Computes the exact world-space exit position and forward direction when the
 * camera crosses an oriented wormhole mouth into its linked partner mouth.
 * Matches the GLSL shader's mouth-local R_z * reflect gluing bit-for-bit.
 */
export function crossMouthWorld(
  match: NearestMouthMatch,
  camPos: Vec3,
  camForward: Vec3,
  forceAim: boolean
): WorldMouthCrossingResult {
  const wh = match.wormhole;
  const safeRadius = Math.max(wh.radius, 0.001);
  const safeScale: ShapeScale = [
    Math.max(wh.scale[0], 0.05),
    Math.max(wh.scale[1], 0.05),
    Math.max(wh.scale[2], 0.05),
  ];

  const srcQuat = normalizeQuaternion(match.mouth.quaternion);
  const dstQuat = normalizeQuaternion(match.partnerMouth.quaternion);

  const localOffset = quatRotateInvVec3(srcQuat, [
    camPos[0] - match.mouth.position[0],
    camPos[1] - match.mouth.position[1],
    camPos[2] - match.mouth.position[2],
  ]);
  const crossed = throatCrossing(
    localOffset[0],
    localOffset[1],
    localOffset[2],
    wh.shape,
    safeRadius,
    safeScale
  );

  let localFwd = quatRotateInvVec3(srcQuat, camForward);
  const fwdLen = Math.hypot(localFwd[0], localFwd[1], localFwd[2]);
  if (fwdLen > 1e-8) {
    localFwd = [localFwd[0] / fwdLen, localFwd[1] / fwdLen, localFwd[2] / fwdLen];
  } else {
    localFwd = [-crossed.nx, -crossed.ny, -crossed.nz];
  }

  if (forceAim && localFwd[0] * crossed.nx + localFwd[1] * crossed.ny + localFwd[2] * crossed.nz > -0.2) {
    localFwd = [-crossed.nx, -crossed.ny, -crossed.nz];
  }

  const exitLocalFwd = glueDirectionThroughThroat(
    localFwd[0],
    localFwd[1],
    localFwd[2],
    crossed.nx,
    crossed.ny,
    crossed.nz
  );

  const exitWorldOffset = quatRotateVec3(dstQuat, [crossed.exitX, crossed.exitY, crossed.exitZ]);
  const exitWorldFwd = quatRotateVec3(dstQuat, exitLocalFwd);
  const outLen = Math.hypot(exitWorldFwd[0], exitWorldFwd[1], exitWorldFwd[2]) || 1;

  return {
    exitPosition: [
      match.partnerMouth.position[0] + exitWorldOffset[0],
      match.partnerMouth.position[1] + exitWorldOffset[1],
      match.partnerMouth.position[2] + exitWorldOffset[2],
    ],
    exitForward: [exitWorldFwd[0] / outLen, exitWorldFwd[1] / outLen, exitWorldFwd[2] / outLen],
    destUniverseIndex: match.destUniverseIndex,
  };
}

interface PendingMouth {
  wormholeId: string;
  end: 'A' | 'B';
  universeIndex: number;
  destUniverseIndex: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  radius: number;
  scale: [number, number, number];
  shape: WormholeShapeId;
  gravityRatio: number;
  partner: PendingMouth | null;
  assignedRow: number;
}

/**
 * Packs a MultiverseGraph into contiguous RGBA32F Float32Arrays suitable for direct
 * upload into WebGL2 DataTextures (uUniverseData & uMouthData).
 *
 * Mouths are grouped by universeIndex so the shader can iterate strictly over
 * [mouthOffset, mouthOffset + mouthCount) for the ray's current universe.
 */
export function packMultiverseToFloatBuffers(
  graph: MultiverseGraph,
  reuse?: PackBufferReuse
): PackedMultiverseBuffers {
  const universeCount = graph.universes.length;
  const universeIndexById = new Map<string, number>();
  for (let i = 0; i < universeCount; i++) {
    universeIndexById.set(graph.universes[i].id, i);
  }

  // Bucket mouths by containing universe so each universe's mouths are contiguous.
  const buckets: PendingMouth[][] = Array.from({ length: universeCount }, () => []);

  for (const wh of graph.wormholes) {
    const uA = universeIndexById.get(wh.mouthA.universeId);
    const uB = universeIndexById.get(wh.mouthB.universeId);
    if (uA === undefined || uB === undefined) continue;

    const safeRadius = Math.max(wh.radius, 0.001);
    const safeScale: [number, number, number] = [
      Math.max(wh.scale[0], 0.05),
      Math.max(wh.scale[1], 0.05),
      Math.max(wh.scale[2], 0.05),
    ];
    const safeGravity = Math.max(wh.gravityRatio, 0);

    const mA: PendingMouth = {
      wormholeId: wh.id,
      end: 'A',
      universeIndex: uA,
      destUniverseIndex: uB,
      position: [wh.mouthA.position[0], wh.mouthA.position[1], wh.mouthA.position[2]],
      quaternion: normalizeQuaternion(wh.mouthA.quaternion),
      radius: safeRadius,
      scale: safeScale,
      shape: wh.shape,
      gravityRatio: safeGravity,
      partner: null,
      assignedRow: -1,
    };

    const mB: PendingMouth = {
      wormholeId: wh.id,
      end: 'B',
      universeIndex: uB,
      destUniverseIndex: uA,
      position: [wh.mouthB.position[0], wh.mouthB.position[1], wh.mouthB.position[2]],
      quaternion: normalizeQuaternion(wh.mouthB.quaternion),
      radius: safeRadius,
      scale: safeScale,
      shape: wh.shape,
      gravityRatio: safeGravity,
      partner: mA,
      assignedRow: -1,
    };

    mA.partner = mB;
    buckets[uA].push(mA);
    buckets[uB].push(mB);
  }

  // Flatten buckets and assign contiguous row indices.
  const orderedMouths: PendingMouth[] = [];
  const universeMouthSlices: UniverseMouthSlice[] = [];

  for (let u = 0; u < universeCount; u++) {
    const list = buckets[u];
    const offset = orderedMouths.length;
    const count = list.length;

    for (const mouth of list) {
      mouth.assignedRow = orderedMouths.length;
      orderedMouths.push(mouth);
    }

    let cx = 0;
    let cy = 0;
    let cz = 0;
    let clusterRadius = 0;

    if (count === 1) {
      const m = list[0];
      cx = m.position[0];
      cy = m.position[1];
      cz = m.position[2];
      clusterRadius = m.radius * Math.max(m.scale[0], m.scale[1], m.scale[2]);
    } else if (count > 1) {
      for (const m of list) {
        cx += m.position[0];
        cy += m.position[1];
        cz += m.position[2];
      }
      cx /= count;
      cy /= count;
      cz /= count;
      for (const m of list) {
        const extent = m.radius * Math.max(m.scale[0], m.scale[1], m.scale[2]);
        const d = Math.hypot(m.position[0] - cx, m.position[1] - cy, m.position[2] - cz) + extent;
        if (d > clusterRadius) clusterRadius = d;
      }
    }

    universeMouthSlices.push({
      offset,
      count,
      clusterCenter: [cx, cy, cz],
      clusterRadius,
    });
  }

  const universeHeight = Math.max(1, universeCount);
  const mouthCount = orderedMouths.length;
  const mouthHeight = Math.max(1, mouthCount);

  const neededUniverseFloats = universeHeight * UNIVERSE_TEXELS_PER_ROW * 4;
  const universeData =
    reuse?.universeData && reuse.universeData.length === neededUniverseFloats
      ? reuse.universeData
      : new Float32Array(neededUniverseFloats);

  const neededMouthFloats = mouthHeight * MOUTH_TEXELS_PER_ROW * 4;
  const mouthData =
    reuse?.mouthData && reuse.mouthData.length === neededMouthFloats
      ? reuse.mouthData
      : new Float32Array(neededMouthFloats);

  const neededOverrideBytes = universeHeight * GRID_OVERRIDE_TEX_WIDTH * 4;
  const gridOverrideData =
    reuse?.gridOverrideData && reuse.gridOverrideData.length === neededOverrideBytes
      ? reuse.gridOverrideData
      : new Uint8Array(neededOverrideBytes);
  gridOverrideData.fill(0);

  // Zero unused sentinel rows when graph is empty
  if (universeCount === 0) universeData.fill(0);
  if (mouthCount === 0) mouthData.fill(0);

  for (let u = 0; u < universeCount; u++) {
    const node = graph.universes[u];
    const slice = universeMouthSlices[u];
    const [lonR, lonG, lonB] = parseHexColorSrgb(node.gridVerticalColor, [1, 0, 0]);
    const [latR, latG, latB] = parseHexColorSrgb(node.gridHorizontalColor, [0, 0, 1]);
    const base = u * UNIVERSE_TEXELS_PER_ROW * 4;

    // Texel 0: (skyMode, textureLayer, isHdr, linearExposure)
    universeData[base + 0] = node.skyMode === 'texture' ? 1 : 0;
    universeData[base + 1] = node.textureLayer;
    universeData[base + 2] = node.textureLayer >= 0 && node.isHdr ? 1 : 0;
    universeData[base + 3] = Math.pow(2, node.exposureEv);

    // Texel 1: (gridCols, gridRows, rotYawRad, rotPitchRad)
    universeData[base + 4] = Math.max(1, node.gridCols);
    universeData[base + 5] = Math.max(1, node.gridRows);
    universeData[base + 6] = node.rotationY * DEG2RAD;
    universeData[base + 7] = node.rotationX * DEG2RAD;

    // Texel 2: (colorLon.rgb, mouthOffset)
    universeData[base + 8] = lonR;
    universeData[base + 9] = lonG;
    universeData[base + 10] = lonB;
    universeData[base + 11] = slice.offset;

    // Texel 3: (colorLat.rgb, mouthCount)
    universeData[base + 12] = latR;
    universeData[base + 13] = latG;
    universeData[base + 14] = latB;
    universeData[base + 15] = slice.count;

    // Texel 4: (clusterCenter.xyz, clusterRadius)
    universeData[base + 16] = slice.clusterCenter[0];
    universeData[base + 17] = slice.clusterCenter[1];
    universeData[base + 18] = slice.clusterCenter[2];
    universeData[base + 19] = slice.clusterRadius;

    // Pack per-line longitude & latitude color overrides into RGBA8 row u
    const rowByteBase = u * GRID_OVERRIDE_TEX_WIDTH * 4;
    if (node.gridLonOverrides) {
      for (const [rawKey, hex] of Object.entries(node.gridLonOverrides)) {
        const lineIdx = Number(rawKey);
        if (!Number.isInteger(lineIdx) || lineIdx < 0 || lineIdx >= GRID_OVERRIDE_MAX_LON) continue;
        const [r, g, b] = parseHexColorSrgb(hex, [1, 1, 1]);
        const p = rowByteBase + lineIdx * 4;
        gridOverrideData[p + 0] = Math.round(r * 255);
        gridOverrideData[p + 1] = Math.round(g * 255);
        gridOverrideData[p + 2] = Math.round(b * 255);
        gridOverrideData[p + 3] = 255;
      }
    }
    if (node.gridLatOverrides) {
      for (const [rawKey, hex] of Object.entries(node.gridLatOverrides)) {
        const lineIdx = Number(rawKey);
        if (!Number.isInteger(lineIdx) || lineIdx < 0 || lineIdx >= GRID_OVERRIDE_MAX_LAT) continue;
        const [r, g, b] = parseHexColorSrgb(hex, [1, 1, 1]);
        const p = rowByteBase + (GRID_OVERRIDE_LAT_OFFSET + lineIdx) * 4;
        gridOverrideData[p + 0] = Math.round(r * 255);
        gridOverrideData[p + 1] = Math.round(g * 255);
        gridOverrideData[p + 2] = Math.round(b * 255);
        gridOverrideData[p + 3] = 255;
      }
    }
  }

  const mouthLookup: PackedMouthMetadata[] = [];

  for (let m = 0; m < mouthCount; m++) {
    const entry = orderedMouths[m];
    const destMouthIndex = entry.partner ? entry.partner.assignedRow : m;
    const base = m * MOUTH_TEXELS_PER_ROW * 4;

    // Texel 0: (center.xyz, radius)
    mouthData[base + 0] = entry.position[0];
    mouthData[base + 1] = entry.position[1];
    mouthData[base + 2] = entry.position[2];
    mouthData[base + 3] = entry.radius;

    // Texel 1: (quaternion.xyzw)
    mouthData[base + 4] = entry.quaternion[0];
    mouthData[base + 5] = entry.quaternion[1];
    mouthData[base + 6] = entry.quaternion[2];
    mouthData[base + 7] = entry.quaternion[3];

    // Texel 2: (scale.xyz, shapeId)
    mouthData[base + 8] = entry.scale[0];
    mouthData[base + 9] = entry.scale[1];
    mouthData[base + 10] = entry.scale[2];
    mouthData[base + 11] = entry.shape;

    // Texel 3: (gravityRatio, universeIndex, destUniverseIndex, destMouthIndex)
    mouthData[base + 12] = entry.gravityRatio;
    mouthData[base + 13] = entry.universeIndex;
    mouthData[base + 14] = entry.destUniverseIndex;
    mouthData[base + 15] = destMouthIndex;

    mouthLookup.push({
      wormholeId: entry.wormholeId,
      end: entry.end,
      universeIndex: entry.universeIndex,
      destUniverseIndex: entry.destUniverseIndex,
      destMouthIndex,
    });
  }

  return {
    universeData,
    universeWidth: UNIVERSE_TEXELS_PER_ROW,
    universeHeight,
    universeCount,
    mouthData,
    mouthWidth: MOUTH_TEXELS_PER_ROW,
    mouthHeight,
    mouthCount,
    gridOverrideData,
    gridOverrideWidth: GRID_OVERRIDE_TEX_WIDTH,
    gridOverrideHeight: universeHeight,
    universeIndexById,
    universeMouthSlices,
    mouthLookup,
  };
}

export interface TwoUniverseGridParams {
  verticalColor: string;
  horizontalColor: string;
  cols: number;
  rows: number;
  rotationY: number;
  rotationX: number;
  lonOverrides?: GridLineColorOverrides;
  latOverrides?: GridLineColorOverrides;
}

export interface TwoUniverseSceneParams {
  position: readonly [number, number, number];
  radius: number;
  shape: WormholeShapeId;
  scale?: ShapeScale;
  gravityRatio: number;
  universeA: {
    mode: UniverseSkyMode;
    grid: TwoUniverseGridParams;
    isHdr: boolean;
    exposureEv: number;
    textureLayer: number;
  };
  universeB: {
    mode: UniverseSkyMode;
    grid: TwoUniverseGridParams;
    isHdr: boolean;
    exposureEv: number;
    textureLayer: number;
  };
}

/**
 * Builds the canonical two-universe / single-wormhole MultiverseGraph
 * matching the classic A <-> B configuration.
 */
export function createTwoUniverseGraph(params: TwoUniverseSceneParams): MultiverseGraph {
  const scale = params.scale ?? DEFAULT_SHAPE_SCALE;
  return {
    universes: [
      {
        id: 'A',
        name: 'Universe A',
        skyMode: params.universeA.mode,
        gridVerticalColor: params.universeA.grid.verticalColor,
        gridHorizontalColor: params.universeA.grid.horizontalColor,
        gridCols: params.universeA.grid.cols,
        gridRows: params.universeA.grid.rows,
        rotationY: params.universeA.grid.rotationY,
        rotationX: params.universeA.grid.rotationX,
        isHdr: params.universeA.isHdr,
        exposureEv: params.universeA.exposureEv,
        textureLayer: params.universeA.textureLayer,
        gridLonOverrides: params.universeA.grid.lonOverrides
          ? { ...params.universeA.grid.lonOverrides }
          : undefined,
        gridLatOverrides: params.universeA.grid.latOverrides
          ? { ...params.universeA.grid.latOverrides }
          : undefined,
      },
      {
        id: 'B',
        name: 'Universe B',
        skyMode: params.universeB.mode,
        gridVerticalColor: params.universeB.grid.verticalColor,
        gridHorizontalColor: params.universeB.grid.horizontalColor,
        gridCols: params.universeB.grid.cols,
        gridRows: params.universeB.grid.rows,
        rotationY: params.universeB.grid.rotationY,
        rotationX: params.universeB.grid.rotationX,
        isHdr: params.universeB.isHdr,
        exposureEv: params.universeB.exposureEv,
        textureLayer: params.universeB.textureLayer,
        gridLonOverrides: params.universeB.grid.lonOverrides
          ? { ...params.universeB.grid.lonOverrides }
          : undefined,
        gridLatOverrides: params.universeB.grid.latOverrides
          ? { ...params.universeB.grid.latOverrides }
          : undefined,
      },
    ],
    wormholes: [
      {
        id: 'wh-0',
        name: 'Primary Throat',
        shape: params.shape,
        radius: params.radius,
        scale: [scale[0], scale[1], scale[2]],
        gravityRatio: params.gravityRatio,
        mouthA: {
          universeId: 'A',
          position: [params.position[0], params.position[1], params.position[2]],
          quaternion: [0, 0, 0, 1],
        },
        mouthB: {
          universeId: 'B',
          position: [params.position[0], params.position[1], params.position[2]],
          quaternion: [0, 0, 0, 1],
        },
      },
    ],
  };
}

export type MultiversePresetId = 'classic-binary' | 'trinity-loop' | 'portal-hall';

export interface MultiversePresetInfo {
  id: MultiversePresetId;
  label: string;
  badge: string;
  description: string;
}

export const MULTIVERSE_PRESETS: readonly MultiversePresetInfo[] = [
  {
    id: 'classic-binary',
    label: 'Classic',
    badge: '2U · 1W',
    description: 'Single adjustable wormhole bridging Universe A and Universe B.',
  },
  {
    id: 'trinity-loop',
    label: 'Trinity',
    badge: '3U · 3W',
    description:
      'Three universes (A, B, C) linked in a triangle by an angled Sphere, Torus, and Octahedron.',
  },
  {
    id: 'portal-hall',
    label: 'Portal Hall',
    badge: '2U · 2W',
    description:
      'Facing same-universe Cuboid portals in Universe A (recursive bounces) plus an upper Cylinder to B.',
  },
] as const;

/**
 * Constructs a MultiverseGraph for any Stage 2 verification preset, preserving
 * the user's Universe A & B skies, grids, HDRI exposure, and gravity strength.
 */
export function createPresetMultiverseGraph(
  preset: MultiversePresetId,
  params: TwoUniverseSceneParams
): MultiverseGraph {
  if (preset === 'classic-binary') {
    return createTwoUniverseGraph(params);
  }

  const baseA: UniverseNode = {
    id: 'A',
    name: 'Universe A',
    skyMode: params.universeA.mode,
    gridVerticalColor: params.universeA.grid.verticalColor,
    gridHorizontalColor: params.universeA.grid.horizontalColor,
    gridCols: params.universeA.grid.cols,
    gridRows: params.universeA.grid.rows,
    rotationY: params.universeA.grid.rotationY,
    rotationX: params.universeA.grid.rotationX,
    isHdr: params.universeA.isHdr,
    exposureEv: params.universeA.exposureEv,
    textureLayer: params.universeA.textureLayer,
    gridLonOverrides: params.universeA.grid.lonOverrides
      ? { ...params.universeA.grid.lonOverrides }
      : undefined,
    gridLatOverrides: params.universeA.grid.latOverrides
      ? { ...params.universeA.grid.latOverrides }
      : undefined,
  };

  const baseB: UniverseNode = {
    id: 'B',
    name: 'Universe B',
    skyMode: params.universeB.mode,
    gridVerticalColor: params.universeB.grid.verticalColor,
    gridHorizontalColor: params.universeB.grid.horizontalColor,
    gridCols: params.universeB.grid.cols,
    gridRows: params.universeB.grid.rows,
    rotationY: params.universeB.grid.rotationY,
    rotationX: params.universeB.grid.rotationX,
    isHdr: params.universeB.isHdr,
    exposureEv: params.universeB.exposureEv,
    textureLayer: params.universeB.textureLayer,
    gridLonOverrides: params.universeB.grid.lonOverrides
      ? { ...params.universeB.grid.lonOverrides }
      : undefined,
    gridLatOverrides: params.universeB.grid.latOverrides
      ? { ...params.universeB.grid.latOverrides }
      : undefined,
  };

  const g = Math.max(params.gravityRatio, 0);

  if (preset === 'trinity-loop') {
    const universeC: UniverseNode = {
      id: 'C',
      name: 'Universe C',
      skyMode: 'grid',
      gridVerticalColor: '#f59e0b',
      gridHorizontalColor: '#10b981',
      gridCols: 20,
      gridRows: 10,
      rotationY: 20,
      rotationX: 0,
      isHdr: false,
      exposureEv: 0,
      textureLayer: -1,
    };

    const leftYaw = quaternionFromEulerDeg(18, 0, 0);
    const rightYaw = quaternionFromEulerDeg(-18, 0, 0);

    return {
      universes: [baseA, baseB, universeC],
      wormholes: [
        {
          id: 'wh-ab-sphere',
          name: 'A ↔ B Sphere',
          shape: 0, // Sphere
          radius: 0.95,
          scale: [1, 1, 1],
          gravityRatio: g,
          mouthA: {
            universeId: 'A',
            position: [-2.35, 0, 3.2],
            quaternion: leftYaw,
          },
          mouthB: {
            universeId: 'B',
            position: [-2.35, 0, 3.2],
            quaternion: leftYaw,
          },
        },
        {
          id: 'wh-ac-torus',
          name: 'A ↔ C Torus',
          shape: 3, // Torus
          radius: 0.92,
          scale: [1, 1, 1],
          gravityRatio: g,
          mouthA: {
            universeId: 'A',
            position: [2.35, 0, 3.2],
            quaternion: rightYaw,
          },
          mouthB: {
            universeId: 'C',
            position: [2.35, 0, 3.2],
            quaternion: rightYaw,
          },
        },
        {
          id: 'wh-bc-octa',
          name: 'B ↔ C Octahedron',
          shape: 4, // Octahedron
          radius: 0.85,
          scale: [1, 1.15, 1],
          gravityRatio: g,
          mouthA: {
            universeId: 'B',
            position: [2.35, 0, 3.2],
            quaternion: rightYaw,
          },
          mouthB: {
            universeId: 'C',
            position: [-2.35, 0, 3.2],
            quaternion: leftYaw,
          },
        },
      ],
    };
  }

  // 'portal-hall': Same-universe facing Cuboids in A (A <-> A) + Cylinder to B (A <-> B)
  return {
    universes: [baseA, baseB],
    wormholes: [
      {
        id: 'wh-aa-hall',
        name: 'A ↔ A Mirror Portals',
        shape: 1, // Cuboid
        radius: 0.82,
        scale: [0.95, 1.25, 0.55],
        gravityRatio: g * 0.7,
        mouthA: {
          universeId: 'A',
          position: [-2.25, -0.15, 3.2],
          quaternion: quaternionFromEulerDeg(26, 0, 0),
        },
        mouthB: {
          universeId: 'A',
          position: [2.25, -0.15, 3.2],
          quaternion: quaternionFromEulerDeg(-26, 0, 0),
        },
      },
      {
        id: 'wh-ab-cyl',
        name: 'A ↔ B Spire',
        shape: 2, // Cylinder
        radius: 0.68,
        scale: [1, 0.95, 1],
        gravityRatio: g,
        mouthA: {
          universeId: 'A',
          position: [0, 2.35, 3.6],
          quaternion: quaternionFromEulerDeg(0, 12, 0),
        },
        mouthB: {
          universeId: 'B',
          position: [0, 0, 3.0],
          quaternion: [0, 0, 0, 1],
        },
      },
    ],
  };
}

/**
 * Conservative local-space bounding sphere radius multiplier (times radius * maxScale)
 * for each wormhole shape, matching wormholeSdf.ts proportions.
 */
export function getShapeBoundingMultiplier(shape: WormholeShapeId): number {
  switch (shape) {
    case 1: // Cuboid half-extents (1.2, 0.72, 0.9)
      return Math.hypot(1.2, 0.72, 0.9);
    case 2: // Cylinder radius 0.85, half-height 1.45
      return Math.hypot(0.85, 1.45);
    case 3: // Torus major 1.0 + minor 0.36
      return 1.36;
    case 4: // Octahedron size 1.55
      return 1.55;
    case 5: // Bicone height 1.5, radius 1.0
      return 1.5;
    case 0:
    default:
      return 1.0;
  }
}

export interface MouthOverlapWarning {
  universeId: string;
  universeName: string;
  wormholeAId: string;
  wormholeAName: string;
  endA: 'A' | 'B';
  wormholeBId: string;
  wormholeBName: string;
  endB: 'A' | 'B';
  penetrationDepth: number;
}

interface PlacedMouthRecord {
  wormhole: WormholeEdge;
  end: 'A' | 'B';
  mouth: WormholeMouthPlacement;
  boundRadius: number;
}

function evalMouthGradientWorld(
  record: PlacedMouthRecord,
  p: Vec3
): [number, number, number] {
  const wh = record.wormhole;
  const r = Math.max(wh.radius, 0.001);
  const sMin = Math.min(Math.max(wh.scale[0], 0.05), Math.max(wh.scale[1], 0.05), Math.max(wh.scale[2], 0.05));
  const eps = Math.max(r * sMin * 2e-3, 1e-4);
  const dx =
    evaluateMouthSdfWorld(record.mouth, wh.shape, r, wh.scale, [p[0] + eps, p[1], p[2]]) -
    evaluateMouthSdfWorld(record.mouth, wh.shape, r, wh.scale, [p[0] - eps, p[1], p[2]]);
  const dy =
    evaluateMouthSdfWorld(record.mouth, wh.shape, r, wh.scale, [p[0], p[1] + eps, p[2]]) -
    evaluateMouthSdfWorld(record.mouth, wh.shape, r, wh.scale, [p[0], p[1] - eps, p[2]]);
  const dz =
    evaluateMouthSdfWorld(record.mouth, wh.shape, r, wh.scale, [p[0], p[1], p[2] + eps]) -
    evaluateMouthSdfWorld(record.mouth, wh.shape, r, wh.scale, [p[0], p[1], p[2] - eps]);
  const len = Math.hypot(dx, dy, dz);
  if (len > 1e-8) return [dx / len, dy / len, dz / len];
  return [0, 0, 1];
}

/**
 * Checks whether two distinct mouths in the same universe overlap or graze
 * within a small safety margin. Returns penetration depth (> 0) if overlapping,
 * or 0 if safely separated.
 */
function estimateMouthPairPenetration(a: PlacedMouthRecord, b: PlacedMouthRecord): number {
  const margin = 0.03;
  const dx = b.mouth.position[0] - a.mouth.position[0];
  const dy = b.mouth.position[1] - a.mouth.position[1];
  const dz = b.mouth.position[2] - a.mouth.position[2];
  const centerDist = Math.hypot(dx, dy, dz);

  // Fast broadphase reject using conservative bounding spheres
  if (centerDist > a.boundRadius + b.boundRadius + margin) {
    return 0;
  }

  const evalA = (p: Vec3) =>
    evaluateMouthSdfWorld(a.mouth, a.wormhole.shape, Math.max(a.wormhole.radius, 0.001), a.wormhole.scale, p);
  const evalB = (p: Vec3) =>
    evaluateMouthSdfWorld(b.mouth, b.wormhole.shape, Math.max(b.wormhole.radius, 0.001), b.wormhole.scale, p);

  let maxPen = 0;
  const testPoint = (p: Vec3) => {
    const fA = evalA(p);
    const fB = evalB(p);
    const combined = Math.max(fA, fB);
    if (combined < margin) {
      const pen = margin - combined;
      if (pen > maxPen) maxPen = pen;
    }
  };

  // 1. Sample along center-to-center segment
  const segSamples = 9;
  for (let i = 0; i <= segSamples; i++) {
    const t = i / segSamples;
    testPoint([
      a.mouth.position[0] + dx * t,
      a.mouth.position[1] + dy * t,
      a.mouth.position[2] + dz * t,
    ]);
  }

  // 2. Alternating projection from A's center toward B, and B's center toward A
  const runAlternating = (start: Vec3) => {
    let px = start[0];
    let py = start[1];
    let pz = start[2];
    for (let iter = 0; iter < 6; iter++) {
      testPoint([px, py, pz]);
      const fA = evalA([px, py, pz]);
      const fB = evalB([px, py, pz]);
      if (fA > fB && fA > 0) {
        const g = evalMouthGradientWorld(a, [px, py, pz]);
        px -= g[0] * fA;
        py -= g[1] * fA;
        pz -= g[2] * fA;
      } else if (fB > 0) {
        const g = evalMouthGradientWorld(b, [px, py, pz]);
        px -= g[0] * fB;
        py -= g[1] * fB;
        pz -= g[2] * fB;
      } else {
        break;
      }
    }
    testPoint([px, py, pz]);
  };

  runAlternating([
    (a.mouth.position[0] + b.mouth.position[0]) * 0.5,
    (a.mouth.position[1] + b.mouth.position[1]) * 0.5,
    (a.mouth.position[2] + b.mouth.position[2]) * 0.5,
  ]);
  runAlternating(a.mouth.position);
  runAlternating(b.mouth.position);

  // 3. Probe local principal axes of both mouths (catches rotated torus tubes & elongated cuboids)
  const probeAxes = (rec: PlacedMouthRecord) => {
    const q = normalizeQuaternion(rec.mouth.quaternion);
    const rx = Math.max(rec.wormhole.radius, 0.001) * Math.max(rec.wormhole.scale[0], 0.05);
    const ry = Math.max(rec.wormhole.radius, 0.001) * Math.max(rec.wormhole.scale[1], 0.05);
    const rz = Math.max(rec.wormhole.radius, 0.001) * Math.max(rec.wormhole.scale[2], 0.05);
    const locals: Array<[number, number, number]> = [
      [rx, 0, 0],
      [-rx, 0, 0],
      [0, ry, 0],
      [0, -ry, 0],
      [0, 0, rz],
      [0, 0, -rz],
    ];
    for (const loc of locals) {
      const worldOff = quatRotateVec3(q, loc);
      testPoint([
        rec.mouth.position[0] + worldOff[0],
        rec.mouth.position[1] + worldOff[1],
        rec.mouth.position[2] + worldOff[2],
      ]);
    }
  };
  probeAxes(a);
  probeAxes(b);

  return maxPen;
}

/**
 * Validates the multiverse graph for overlapping wormhole mouths inside the
 * same universe (which would create conflicting negative SDF regions).
 */
export function detectMouthOverlaps(graph: MultiverseGraph): MouthOverlapWarning[] {
  const warnings: MouthOverlapWarning[] = [];
  const universeMap = new Map<string, UniverseNode>();
  const byUniverse = new Map<string, PlacedMouthRecord[]>();

  for (const u of graph.universes) {
    universeMap.set(u.id, u);
    byUniverse.set(u.id, []);
  }

  for (const wh of graph.wormholes) {
    const maxScale = Math.max(wh.scale[0], wh.scale[1], wh.scale[2], 0.05);
    const boundRadius = Math.max(wh.radius, 0.001) * maxScale * getShapeBoundingMultiplier(wh.shape);

    const listA = byUniverse.get(wh.mouthA.universeId);
    if (listA) {
      listA.push({ wormhole: wh, end: 'A', mouth: wh.mouthA, boundRadius });
    }
    const listB = byUniverse.get(wh.mouthB.universeId);
    if (listB) {
      listB.push({ wormhole: wh, end: 'B', mouth: wh.mouthB, boundRadius });
    }
  }

  for (const [universeId, records] of byUniverse.entries()) {
    if (records.length < 2) continue;
    const uNode = universeMap.get(universeId);
    const uName = uNode?.name ?? universeId;

    for (let i = 0; i < records.length; i++) {
      for (let j = i + 1; j < records.length; j++) {
        const pen = estimateMouthPairPenetration(records[i], records[j]);
        if (pen > 0) {
          warnings.push({
            universeId,
            universeName: uName,
            wormholeAId: records[i].wormhole.id,
            wormholeAName: records[i].wormhole.name,
            endA: records[i].end,
            wormholeBId: records[j].wormhole.id,
            wormholeBName: records[j].wormhole.name,
            endB: records[j].end,
            penetrationDepth: pen,
          });
        }
      }
    }
  }

  return warnings;
}

const DEFAULT_UNIVERSE_PALETTES: readonly { lon: string; lat: string }[] = [
  { lon: '#ff0000', lat: '#0000ff' },
  { lon: '#00ffff', lat: '#ff00ff' },
  { lon: '#f59e0b', lat: '#10b981' },
  { lon: '#a855f7', lat: '#38bdf8' },
  { lon: '#fb7185', lat: '#facc15' },
  { lon: '#22c55e', lat: '#0ea5e9' },
  { lon: '#f97316', lat: '#ec4899' },
];

/**
 * Creates a new UniverseNode with a unique ID, readable letter/number name,
 * and distinct procedural grid palette.
 */
export function createDefaultUniverseNode(existing: readonly UniverseNode[]): UniverseNode {
  const usedIds = new Set(existing.map((u) => u.id));
  const usedNames = new Set(existing.map((u) => u.name.toLowerCase()));
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

  let id = '';
  let name = '';
  for (let i = 0; i < letters.length; i++) {
    const ch = letters[i];
    if (!usedIds.has(ch) && !usedNames.has(`universe ${ch.toLowerCase()}`)) {
      id = ch;
      name = `Universe ${ch}`;
      break;
    }
  }
  if (!id) {
    let n = existing.length + 1;
    while (usedIds.has(`U${n}`)) n += 1;
    id = `U${n}`;
    name = `Universe ${n}`;
  }

  const palette = DEFAULT_UNIVERSE_PALETTES[existing.length % DEFAULT_UNIVERSE_PALETTES.length];
  return {
    id,
    name,
    skyMode: 'grid',
    gridVerticalColor: palette.lon,
    gridHorizontalColor: palette.lat,
    gridCols: 18,
    gridRows: 9,
    rotationY: 0,
    rotationX: 0,
    isHdr: false,
    exposureEv: 0,
    textureLayer: -1,
  };
}

function findOpenMouthPosition(
  graph: MultiverseGraph,
  universeId: string,
  newRadius: number,
  extraOccupied?: readonly [number, number, number][]
): [number, number, number] {
  const occupied: Array<{ pos: [number, number, number]; bound: number }> = [];
  for (const wh of graph.wormholes) {
    const maxScale = Math.max(wh.scale[0], wh.scale[1], wh.scale[2], 0.05);
    const bound = Math.max(wh.radius, 0.001) * maxScale * getShapeBoundingMultiplier(wh.shape);
    if (wh.mouthA.universeId === universeId) {
      occupied.push({ pos: wh.mouthA.position, bound });
    }
    if (wh.mouthB.universeId === universeId) {
      occupied.push({ pos: wh.mouthB.position, bound });
    }
  }
  if (extraOccupied) {
    for (const pos of extraOccupied) {
      occupied.push({ pos, bound: newRadius });
    }
  }

  if (occupied.length === 0) {
    return [0, 0, 3];
  }

  const candidates: Array<[number, number, number]> = [
    [0, 0, 3],
    [-2.6, 0, 3],
    [2.6, 0, 3],
    [0, 2.6, 3],
    [0, -2.6, 3],
    [-2.6, 2.4, 3.2],
    [2.6, 2.4, 3.2],
    [-2.6, -2.4, 3.2],
    [2.6, -2.4, 3.2],
    [-4.8, 0, 3.2],
    [4.8, 0, 3.2],
    [0, 4.8, 3.2],
    [0, -4.8, 3.2],
  ];

  const isFree = (cand: [number, number, number]) => {
    for (const item of occupied) {
      const dist = Math.hypot(
        cand[0] - item.pos[0],
        cand[1] - item.pos[1],
        cand[2] - item.pos[2]
      );
      if (dist < item.bound + newRadius + 0.5) return false;
    }
    return true;
  };

  for (const cand of candidates) {
    if (isFree(cand)) return cand;
  }

  // Fallback spiral placement
  for (let ring = 1; ring <= 8; ring++) {
    const rad = 2.6 + ring * 2.2;
    for (let k = 0; k < 12; k++) {
      const angle = (k / 12) * Math.PI * 2;
      const cand: [number, number, number] = [
        Number((Math.cos(angle) * rad).toFixed(2)),
        Number((Math.sin(angle) * rad).toFixed(2)),
        3,
      ];
      if (isFree(cand)) return cand;
    }
  }

  return [occupied.length * 2.6, 0, 3];
}

/**
 * Creates a new WormholeEdge linking universeAId <-> universeBId, choosing
 * non-overlapping mouth coordinates in both universes automatically.
 */
export function createDefaultWormholeEdge(
  graph: MultiverseGraph,
  universeAId: string,
  universeBId: string,
  shape: WormholeShapeId = 0,
  gravityRatio: number = 0.5
): WormholeEdge {
  const usedIds = new Set(graph.wormholes.map((w) => w.id));
  let idx = graph.wormholes.length + 1;
  while (usedIds.has(`wh-${idx}`)) idx += 1;

  const uA = graph.universes.find((u) => u.id === universeAId);
  const uB = graph.universes.find((u) => u.id === universeBId);
  const labelA = uA ? uA.id : universeAId;
  const labelB = uB ? uB.id : universeBId;

  const radius = 0.9;
  const posA = findOpenMouthPosition(graph, universeAId, radius);
  const posB =
    universeAId === universeBId
      ? findOpenMouthPosition(graph, universeBId, radius, [posA])
      : findOpenMouthPosition(graph, universeBId, radius);

  return {
    id: `wh-${idx}`,
    name: `${labelA} ↔ ${labelB} #${idx}`,
    shape,
    radius,
    scale: [1, 1, 1],
    gravityRatio: Math.max(0, Math.min(2, gravityRatio)),
    mouthA: {
      universeId: universeAId,
      position: posA,
      quaternion: [0, 0, 0, 1],
    },
    mouthB: {
      universeId: universeBId,
      position: posB,
      quaternion: [0, 0, 0, 1],
    },
  };
}

export interface SerializedCameraKeyframe {
  id?: string;
  label?: string;
  universeId?: string;
  room: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  target: [number, number, number];
}

export interface SerializedMultiversePayload {
  version: 1;
  universes: Array<
    Omit<UniverseNode, 'textureLayer'> & {
      hadCustomTexture?: boolean;
    }
  >;
  wormholes: Array<Omit<WormholeEdge, 'shape'> & { shape: WormholeShapeName }>;
  keyframes?: SerializedCameraKeyframe[];
}

/**
 * Serializes a MultiverseGraph (and optional camera keyframes) into a portable,
 * human-readable JSON string. Local uploaded image/HDRI pixel buffers are not
 * embedded; universes using custom textures are marked with `hadCustomTexture: true`
 * so import can report a clear notice.
 */
export function serializeMultiverseGraph(
  graph: MultiverseGraph,
  keyframes?: readonly SerializedCameraKeyframe[]
): string {
  const serializedKeyframes: SerializedCameraKeyframe[] | undefined =
    keyframes && keyframes.length > 0
      ? keyframes.map((kf, idx) => ({
          ...(kf.id ? { id: kf.id } : { id: `kf-${idx + 1}` }),
          ...(kf.label ? { label: kf.label } : {}),
          universeId:
            kf.universeId ?? graph.universes[Math.max(0, Math.min(graph.universes.length - 1, Math.round(kf.room)))]?.id,
          room: Math.max(0, Math.min(Math.max(0, graph.universes.length - 1), Math.round(kf.room))),
          position: [kf.position[0], kf.position[1], kf.position[2]],
          quaternion: normalizeQuaternion(kf.quaternion),
          target: [kf.target[0], kf.target[1], kf.target[2]],
        }))
      : undefined;

  const payload: SerializedMultiversePayload = {
    version: 1,
    universes: graph.universes.map((u) => {
      const { textureLayer, ...rest } = u;
      return {
        ...rest,
        hadCustomTexture: u.skyMode === 'texture' || textureLayer >= 0 ? true : undefined,
      };
    }),
    wormholes: graph.wormholes.map((wh) => ({
      id: wh.id,
      name: wh.name,
      shape: wormholeShapeIdToName(wh.shape),
      radius: wh.radius,
      scale: [wh.scale[0], wh.scale[1], wh.scale[2]],
      gravityRatio: wh.gravityRatio,
      mouthA: {
        universeId: wh.mouthA.universeId,
        position: [wh.mouthA.position[0], wh.mouthA.position[1], wh.mouthA.position[2]],
        quaternion: normalizeQuaternion(wh.mouthA.quaternion),
      },
      mouthB: {
        universeId: wh.mouthB.universeId,
        position: [wh.mouthB.position[0], wh.mouthB.position[1], wh.mouthB.position[2]],
        quaternion: normalizeQuaternion(wh.mouthB.quaternion),
      },
    })),
    ...(serializedKeyframes ? { keyframes: serializedKeyframes } : {}),
  };
  return JSON.stringify(payload, null, 2);
}

function clampFinite(value: unknown, min: number, max: number, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, num));
}

function sanitizeHexColor(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback;
  const trimmed = value.trim();
  if (/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(trimmed)) return trimmed;
  return fallback;
}

function sanitizeGridLineOverrides(
  value: unknown,
  maxIndexExclusive: number
): GridLineColorOverrides | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const out: GridLineColorOverrides = {};
  let count = 0;
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    const idx = Number(k);
    if (!Number.isInteger(idx) || idx < 0 || idx >= maxIndexExclusive) continue;
    if (typeof v !== 'string') continue;
    const trimmed = v.trim();
    if (!/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(trimmed)) continue;
    out[idx] = trimmed;
    count += 1;
  }
  return count > 0 ? out : undefined;
}

function sanitizeVec3(value: unknown, fallback: [number, number, number], min: number, max: number): [number, number, number] {
  if (!Array.isArray(value) || value.length < 3) return [fallback[0], fallback[1], fallback[2]];
  return [
    clampFinite(value[0], min, max, fallback[0]),
    clampFinite(value[1], min, max, fallback[1]),
    clampFinite(value[2], min, max, fallback[2]),
  ];
}

function sanitizeQuat(value: unknown): [number, number, number, number] {
  if (!Array.isArray(value) || value.length < 4) return [0, 0, 0, 1];
  const nums = [Number(value[0]), Number(value[1]), Number(value[2]), Number(value[3])];
  if (nums.some((n) => !Number.isFinite(n))) return [0, 0, 0, 1];
  return normalizeQuaternion([nums[0], nums[1], nums[2], nums[3]]);
}

export interface ParsedMultiverseResult {
  graph: MultiverseGraph;
  keyframes: SerializedCameraKeyframe[];
  warnings: string[];
}

/**
 * Parses and sanitizes a MultiverseGraph JSON document.
 * Throws a descriptive Error if the JSON is malformed or contains no valid universes.
 */
export function parseMultiverseGraphJson(json: string): ParsedMultiverseResult {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    throw new Error('Invalid JSON syntax.');
  }

  if (!raw || typeof raw !== 'object') {
    throw new Error('Multiverse file must be a JSON object.');
  }

  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.universes) || obj.universes.length === 0) {
    throw new Error('Multiverse file must define at least one universe.');
  }

  const warnings: string[] = [];
  const universes: UniverseNode[] = [];
  const seenUniverseIds = new Set<string>();

  for (let i = 0; i < obj.universes.length; i++) {
    const item = obj.universes[i];
    if (!item || typeof item !== 'object') continue;
    const rec = item as Record<string, unknown>;

    let id = typeof rec.id === 'string' && rec.id.trim().length > 0 ? rec.id.trim() : `U${i + 1}`;
    while (seenUniverseIds.has(id)) {
      id = `${id}_${i + 1}`;
    }
    seenUniverseIds.add(id);

    const name =
      typeof rec.name === 'string' && rec.name.trim().length > 0
        ? rec.name.trim().slice(0, 48)
        : `Universe ${id}`;

    const palette = DEFAULT_UNIVERSE_PALETTES[universes.length % DEFAULT_UNIVERSE_PALETTES.length];
    const requestedTexture = rec.skyMode === 'texture' || rec.hadCustomTexture === true;
    if (requestedTexture) {
      warnings.push(
        `${name} referenced a custom image/HDRI (not embedded in JSON) and was switched to Grid Pattern.`
      );
    }

    const gridLonOverrides = sanitizeGridLineOverrides(rec.gridLonOverrides, GRID_OVERRIDE_MAX_LON);
    const gridLatOverrides = sanitizeGridLineOverrides(rec.gridLatOverrides, GRID_OVERRIDE_MAX_LAT);

    universes.push({
      id,
      name,
      skyMode: 'grid',
      gridVerticalColor: sanitizeHexColor(rec.gridVerticalColor, palette.lon),
      gridHorizontalColor: sanitizeHexColor(rec.gridHorizontalColor, palette.lat),
      gridCols: Math.round(clampFinite(rec.gridCols, 4, 72, 18)),
      gridRows: Math.round(clampFinite(rec.gridRows, 2, 36, 9)),
      rotationY: clampFinite(rec.rotationY, -180, 180, 0),
      rotationX: clampFinite(rec.rotationX, -180, 180, 0),
      isHdr: false,
      exposureEv: clampFinite(rec.exposureEv, -4, 4, 0),
      textureLayer: -1,
      ...(gridLonOverrides ? { gridLonOverrides } : {}),
      ...(gridLatOverrides ? { gridLatOverrides } : {}),
    });
  }

  if (universes.length === 0) {
    throw new Error('No valid universe entries found in multiverse file.');
  }

  const wormholes: WormholeEdge[] = [];
  const seenWhIds = new Set<string>();
  const rawWormholes = Array.isArray(obj.wormholes) ? obj.wormholes : [];

  for (let i = 0; i < rawWormholes.length; i++) {
    const item = rawWormholes[i];
    if (!item || typeof item !== 'object') continue;
    const rec = item as Record<string, unknown>;

    const rawA = rec.mouthA && typeof rec.mouthA === 'object' ? (rec.mouthA as Record<string, unknown>) : null;
    const rawB = rec.mouthB && typeof rec.mouthB === 'object' ? (rec.mouthB as Record<string, unknown>) : null;
    if (!rawA || !rawB) {
      warnings.push(`Skipped wormhole #${i + 1}: missing mouthA or mouthB.`);
      continue;
    }

    const uA = typeof rawA.universeId === 'string' ? rawA.universeId.trim() : '';
    const uB = typeof rawB.universeId === 'string' ? rawB.universeId.trim() : '';
    if (!seenUniverseIds.has(uA) || !seenUniverseIds.has(uB)) {
      warnings.push(`Skipped wormhole #${i + 1}: references unknown universe.`);
      continue;
    }

    let id = typeof rec.id === 'string' && rec.id.trim().length > 0 ? rec.id.trim() : `wh-${i + 1}`;
    while (seenWhIds.has(id)) {
      id = `${id}_${i + 1}`;
    }
    seenWhIds.add(id);

    const name =
      typeof rec.name === 'string' && rec.name.trim().length > 0
        ? rec.name.trim().slice(0, 48)
        : `${uA} ↔ ${uB}`;

    const parsedShape = parseWormholeShape(rec.shape);
    if (parsedShape === null && rec.shape !== undefined) {
      warnings.push(
        `Wormhole "${name}" had unrecognized shape "${String(rec.shape)}" and defaulted to sphere.`
      );
    }
    const rawShape: WormholeShapeId = parsedShape ?? 0;

    wormholes.push({
      id,
      name,
      shape: rawShape,
      radius: clampFinite(rec.radius, 0.1, 5.0, 1.0),
      scale: sanitizeVec3(rec.scale, [1, 1, 1], 0.25, 3.0),
      gravityRatio: clampFinite(rec.gravityRatio, 0, 2.0, 0.5),
      mouthA: {
        universeId: uA,
        position: sanitizeVec3(rawA.position, [0, 0, 3], -25, 25),
        quaternion: sanitizeQuat(rawA.quaternion),
      },
      mouthB: {
        universeId: uB,
        position: sanitizeVec3(rawB.position, [0, 0, 3], -25, 25),
        quaternion: sanitizeQuat(rawB.quaternion),
      },
    });
  }

  const keyframes: SerializedCameraKeyframe[] = [];
  if (Array.isArray(obj.keyframes)) {
    for (let i = 0; i < obj.keyframes.length; i++) {
      const item = obj.keyframes[i];
      if (!item || typeof item !== 'object') continue;
      const rec = item as Record<string, unknown>;
      const rawUid = typeof rec.universeId === 'string' ? rec.universeId.trim() : '';
      let room = rawUid ? universes.findIndex((u) => u.id === rawUid) : -1;
      if (room < 0) {
        room = Math.round(clampFinite(rec.room, 0, Math.max(0, universes.length - 1), 0));
      }
      const resolvedUid = universes[room]?.id;
      keyframes.push({
        id:
          typeof rec.id === 'string' && rec.id.trim().length > 0
            ? rec.id.trim().slice(0, 48)
            : `kf-${i + 1}`,
        ...(typeof rec.label === 'string' && rec.label.trim().length > 0
          ? { label: rec.label.trim().slice(0, 48) }
          : {}),
        ...(resolvedUid ? { universeId: resolvedUid } : {}),
        room,
        position: sanitizeVec3(rec.position, [0, 0, -0.15], -100, 100),
        quaternion: sanitizeQuat(rec.quaternion),
        target: sanitizeVec3(rec.target, [0, 0, 3], -100, 100),
      });
    }
  }

  return {
    graph: { universes, wormholes },
    keyframes,
    warnings,
  };
}

