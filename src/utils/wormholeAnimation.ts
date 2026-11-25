import {
  DEFAULT_SHAPE_SCALE,
  throatTransitionQuaternion,
  wormholeNormal,
  wormholeSdf,
  type ShapeScale,
  type WormholeShapeId,
} from './wormholeSdf.ts';
import {
  evaluateMouthSdfWorld,
  normalizeQuaternion,
  quatRotateInvVec3,
  quatRotateVec3,
  type MultiverseGraph,
  type WormholeEdge,
  type WormholeMouthPlacement,
} from './wormholeMultiverse.ts';

export type InterpolationMode = 'lerp' | 'smoothstep' | 'exp_decay';

export type MovieFormat = 'mp4' | 'webm';

export interface CameraKeyframe {
  /** Optional stable identifier for timeline UI editing & reordering. */
  id?: string;
  /** Optional display label (e.g. "Approach", "Universe B"). */
  label?: string;
  /** Optional stable universe ID so keyframes survive universe insertion/reordering. */
  universeId?: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  target: [number, number, number];
  /** Universe index (0..N-1). */
  room: number;
}

export interface WormholeGeometry {
  center: readonly [number, number, number];
  radius: number;
  shape: WormholeShapeId;
  scale?: ShapeScale;
}

export interface InterpolationOption {
  id: InterpolationMode;
  label: string;
  formula: string;
  description: string;
}

export const EXP_DECAY_RATE = 5.0;

export const INTERPOLATION_OPTIONS: readonly InterpolationOption[] = [
  {
    id: 'lerp',
    label: 'LERP',
    formula: 'u(t) = t',
    description: 'Constant linear speed from start to end.',
  },
  {
    id: 'smoothstep',
    label: 'Smoothstep',
    formula: 'u(t) = 3t² − 2t³',
    description: 'Smooth acceleration and deceleration at both ends.',
  },
  {
    id: 'exp_decay',
    label: 'Exp Decay',
    formula: 'u(t) = (1 − e⁻⁵ᵗ) / (1 − e⁻⁵)',
    description: 'Fast initial motion that settles exponentially into the end pose.',
  },
];

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function sub3(a: readonly [number, number, number], b: readonly [number, number, number]): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add3(a: readonly [number, number, number], b: readonly [number, number, number]): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function lerp3(a: readonly [number, number, number], b: readonly [number, number, number], u: number): Vec3 {
  return [
    a[0] + (b[0] - a[0]) * u,
    a[1] + (b[1] - a[1]) * u,
    a[2] + (b[2] - a[2]) * u,
  ];
}

function dist3(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function len3(v: readonly [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function dot3(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: readonly [number, number, number], b: readonly [number, number, number]): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v: readonly [number, number, number], fallback: Vec3 = [0, 0, 1]): Vec3 {
  const l = len3(v);
  if (l > 1e-8) return [v[0] / l, v[1] / l, v[2] / l];
  return [fallback[0], fallback[1], fallback[2]];
}

export function evaluateEasing(t: number, mode: InterpolationMode): number {
  const clamped = clamp01(t);
  if (clamped <= 0) return 0;
  if (clamped >= 1) return 1;

  switch (mode) {
    case 'lerp':
      return clamped;
    case 'smoothstep':
      return clamped * clamped * (3 - 2 * clamped);
    case 'exp_decay': {
      const denom = 1 - Math.exp(-EXP_DECAY_RATE);
      return (1 - Math.exp(-EXP_DECAY_RATE * clamped)) / denom;
    }
  }
}

export function slerpQuaternion(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number],
  u: number
): Quat {
  const t = clamp01(u);
  let ax = a[0];
  let ay = a[1];
  let az = a[2];
  let aw = a[3];

  let bx = b[0];
  let by = b[1];
  let bz = b[2];
  let bw = b[3];

  const aLen = Math.hypot(ax, ay, az, aw) || 1;
  ax /= aLen;
  ay /= aLen;
  az /= aLen;
  aw /= aLen;

  const bLen = Math.hypot(bx, by, bz, bw) || 1;
  bx /= bLen;
  by /= bLen;
  bz /= bLen;
  bw /= bLen;

  let dot = ax * bx + ay * by + az * bz + aw * bw;
  if (dot < 0) {
    // Take shortest arc on S^3
    dot = -dot;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }

  let rx: number;
  let ry: number;
  let rz: number;
  let rw: number;

  if (dot > 0.9995) {
    rx = ax + (bx - ax) * t;
    ry = ay + (by - ay) * t;
    rz = az + (bz - az) * t;
    rw = aw + (bw - aw) * t;
  } else {
    const theta0 = Math.acos(Math.min(1, Math.max(-1, dot)));
    const sinTheta0 = Math.sin(theta0);
    const theta = theta0 * t;
    const sinTheta = Math.sin(theta);
    const s0 = Math.cos(theta) - (dot * sinTheta) / sinTheta0;
    const s1 = sinTheta / sinTheta0;

    rx = s0 * ax + s1 * bx;
    ry = s0 * ay + s1 * by;
    rz = s0 * az + s1 * bz;
    rw = s0 * aw + s1 * bw;
  }

  const rLen = Math.hypot(rx, ry, rz, rw) || 1;
  return [rx / rLen, ry / rLen, rz / rLen, rw / rLen];
}

function quatMultiply(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number]
): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  const rx = aw * bx + ax * bw + ay * bz - az * by;
  const ry = aw * by - ax * bz + ay * bw + az * bx;
  const rz = aw * bz + ax * by - ay * bx + az * bw;
  const rw = aw * bw - ax * bx - ay * by - az * bz;
  const len = Math.hypot(rx, ry, rz, rw) || 1;
  return [rx / len, ry / len, rz / len, rw / len];
}

function quatConjugate(
  q: readonly [number, number, number, number]
): Quat {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [-q[0] / len, -q[1] / len, -q[2] / len, q[3] / len];
}

/**
 * Builds a Three.js-compatible camera quaternion looking from `eye` toward `target`
 * with world up `+Y` (camera local `-Z` points along `target - eye`).
 */
export function lookAtCameraQuaternion(
  eye: readonly [number, number, number],
  target: readonly [number, number, number],
  worldUp: readonly [number, number, number] = [0, 1, 0]
): Quat {
  const zAxis = normalize3(sub3(eye, target), [0, 0, 1]); // camera +Z points opposite look dir
  let xAxis = cross3(worldUp, zAxis);
  if (len3(xAxis) < 1e-5) {
    xAxis = cross3([0, 0, 1], zAxis);
  }
  xAxis = normalize3(xAxis, [1, 0, 0]);
  const yAxis = normalize3(cross3(zAxis, xAxis), [0, 1, 0]);

  // Rotation matrix columns: [xAxis, yAxis, zAxis]
  const m00 = xAxis[0], m01 = yAxis[0], m02 = zAxis[0];
  const m10 = xAxis[1], m11 = yAxis[1], m12 = zAxis[1];
  const m20 = xAxis[2], m21 = yAxis[2], m22 = zAxis[2];

  const trace = m00 + m11 + m22;
  let qx = 0, qy = 0, qz = 0, qw = 1;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    qw = 0.25 / s;
    qx = (m21 - m12) * s;
    qy = (m02 - m20) * s;
    qz = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }

  return normalizeQuaternion([qx, qy, qz, qw]);
}

/**
 * Spherically interpolates two unit directions, using a consistent horizontal
 * orbit plane if the endpoints are nearly antipodal.
 */
function slerpDirection(d0: Vec3, d1: Vec3, s: number): Vec3 {
  const dot = Math.min(1, Math.max(-1, dot3(d0, d1)));
  if (dot > 0.9995) {
    return normalize3(lerp3(d0, d1, s), d0);
  }
  if (dot < -0.9995) {
    // Pick a deterministic perpendicular axis (prefer orbiting horizontally around Y)
    let ortho = cross3(d0, [0, 1, 0]);
    if (len3(ortho) < 1e-4) {
      ortho = cross3(d0, [1, 0, 0]);
    }
    const tangent = normalize3(ortho, [1, 0, 0]);
    const angle = Math.PI * s;
    const c = Math.cos(angle);
    const sn = Math.sin(angle);
    return normalize3([
      d0[0] * c + tangent[0] * sn,
      d0[1] * c + tangent[1] * sn,
      d0[2] * c + tangent[2] * sn,
    ]);
  }
  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const w0 = Math.sin((1 - s) * theta) / sinTheta;
  const w1 = Math.sin(s * theta) / sinTheta;
  return normalize3([
    d0[0] * w0 + d1[0] * w1,
    d0[1] * w0 + d1[1] * w1,
    d0[2] * w0 + d1[2] * w1,
  ]);
}

function mouthEffectiveRadius(radius: number, scale: ShapeScale): number {
  const r = Math.max(radius, 1e-4);
  const minScale = Math.min(
    Math.max(scale[0], 0.05),
    Math.max(scale[1], 0.05),
    Math.max(scale[2], 0.05)
  );
  return Math.max(r * minScale, 1e-4);
}

function mouthSafeGap(radius: number, scale: ShapeScale): number {
  const effR = mouthEffectiveRadius(radius, scale);
  return Math.max(effR * 0.024, 0.0025);
}

/**
 * Ensures a local point stays strictly outside the wormhole throat surface
 * (gap >= minGap > shader hitEps) so rays never originate inside the negative SDF.
 */
function pushOutsideThroat(
  local: Vec3,
  shape: WormholeShapeId,
  radius: number,
  minGap: number,
  scale: ShapeScale
): Vec3 {
  let x = local[0];
  let y = local[1];
  let z = local[2];
  const r = Math.max(radius, 1e-4);
  const sx = Math.max(scale[0], 0.05);
  const sy = Math.max(scale[1], 0.05);
  const sz = Math.max(scale[2], 0.05);
  const effR = Math.max(r * Math.min(sx, sy, sz), 1e-4);

  for (let i = 0; i < 8; i++) {
    // Guard against the degenerate centerline of the stretched torus tube
    if (shape === 3 && Math.abs(Math.hypot(x / sx, y / sy) - r) < 1e-4 * r && Math.abs(z / sz) < 1e-4 * r) {
      z += 0.05 * r * sz;
    }
    const f = wormholeSdf(shape, x, y, z, r, scale);
    if (f >= minGap) break;
    const n = wormholeNormal(shape, x, y, z, r, scale);
    const push = minGap - f + effR * 0.005;
    x += n[0] * push;
    y += n[1] * push;
    z += n[2] * push;
  }

  return [x, y, z];
}

/**
 * Picks a point strictly inside the wormhole interior (f(core) < 0) so we can
 * bisect from any exterior point to find the exact throat surface crossing.
 */
function interiorCorePoint(
  pA: Vec3,
  pB: Vec3,
  shape: WormholeShapeId,
  radius: number,
  scale: ShapeScale
): Vec3 {
  if (shape !== 3) {
    // Sphere, cuboid, cylinder, octahedron, and double cone all contain the origin.
    return [0, 0, 0];
  }
  const sx = Math.max(scale[0], 0.05);
  const sy = Math.max(scale[1], 0.05);
  // For the stretched torus, the solid tube core is the ellipse (x/sx)^2 + (y/sy)^2 = r^2, z = 0.
  const rhoA = Math.hypot(pA[0] / sx, pA[1] / sy);
  if (rhoA > 1e-4 * radius) {
    return [(radius * pA[0]) / rhoA, (radius * pA[1]) / rhoA, 0];
  }
  const rhoB = Math.hypot(pB[0] / sx, pB[1] / sy);
  if (rhoB > 1e-4 * radius) {
    return [(-radius * pB[0]) / rhoB, (-radius * pB[1]) / rhoB, 0];
  }
  return [radius * sx, 0, 0];
}

/**
 * Computes the throat entry point just outside the surface (in mouth local space),
 * its outward entry normal (in mouth local space), and its R_z-mapped exit point
 * just outside the destination surface (in partner mouth local space).
 */
function computeThroatPortals(
  pA: Vec3,
  pB: Vec3,
  shape: WormholeShapeId,
  radius: number,
  entryMargin: number,
  exitMargin: number,
  scale: ShapeScale
): { entryPos: Vec3; exitPos: Vec3; entryNormal: Vec3 } {
  const r = Math.max(radius, 1e-4);
  const fA = wormholeSdf(shape, pA[0], pA[1], pA[2], r, scale);

  let surfacePoint: Vec3;
  if (fA <= 0) {
    // Start was already on/inside throat: project directly to surface
    surfacePoint = pushOutsideThroat(pA, shape, r, 0, scale);
  } else {
    // Bisect segment from exterior start pA (f > 0) to interior core (f < 0)
    const core = interiorCorePoint(pA, pB, shape, r, scale);
    let lo = 0; // f > 0 (pA)
    let hi = 1; // f < 0 (core)
    for (let i = 0; i < 24; i++) {
      const mid = 0.5 * (lo + hi);
      const pt = lerp3(pA, core, mid);
      const fm = wormholeSdf(shape, pt[0], pt[1], pt[2], r, scale);
      if (fm > 0) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    surfacePoint = lerp3(pA, core, 0.5 * (lo + hi));
  }

  const nA = wormholeNormal(shape, surfacePoint[0], surfacePoint[1], surfacePoint[2], r, scale);
  const rawEntry: Vec3 = [
    surfacePoint[0] + nA[0] * entryMargin,
    surfacePoint[1] + nA[1] * entryMargin,
    surfacePoint[2] + nA[2] * entryMargin,
  ];
  const entryPos = pushOutsideThroat(rawEntry, shape, r, entryMargin, scale);

  // Every wormhole SDF is XY-plane symmetric: f(x, y, -z) = f(x, y, z) with
  // outward normal (nA.x, nA.y, -nA.z). Place exitPos at exitMargin
  // (~hitEps + 0.06r) to match where head-on rays emerge in the destination
  // universe, preventing any magnification or orientation snap across crossing.
  const rawExit: Vec3 = [
    surfacePoint[0] + nA[0] * exitMargin,
    surfacePoint[1] + nA[1] * exitMargin,
    -surfacePoint[2] - nA[2] * exitMargin,
  ];
  const exitPos = pushOutsideThroat(rawExit, shape, r, exitMargin, scale);
  return { entryPos, exitPos, entryNormal: nA };
}

/**
 * Interpolates a single-universe segment from local p0 to local p1 while guaranteeing
 * the camera never slices through the wormhole interior (gap < safeGap).
 */
function interpolateOutsideSegment(
  p0: Vec3,
  p1: Vec3,
  s: number,
  shape: WormholeShapeId,
  radius: number,
  safeGap: number,
  scale: ShapeScale
): Vec3 {
  const t = clamp01(s);
  if (t <= 0) return [p0[0], p0[1], p0[2]];
  if (t >= 1) return [p1[0], p1[1], p1[2]];

  const r = Math.max(radius, 1e-4);

  // Check if the direct straight segment stays safely outside the throat
  let straightClears = true;
  const SAMPLES = 16;
  for (let i = 1; i < SAMPLES; i++) {
    const pt = lerp3(p0, p1, i / SAMPLES);
    if (wormholeSdf(shape, pt[0], pt[1], pt[2], r, scale) < safeGap) {
      straightClears = false;
      break;
    }
  }

  if (straightClears) {
    return lerp3(p0, p1, t);
  }

  // Straight line would slice into the throat: smoothly arc around the wormhole
  const r0 = Math.max(len3(p0), r * 0.01);
  const r1 = Math.max(len3(p1), r * 0.01);
  const d0 = normalize3(p0, [0, 0, -1]);
  const d1 = normalize3(p1, [0, 0, 1]);
  const dir = slerpDirection(d0, d1, t);
  const rad = r0 + (r1 - r0) * t;
  const arcPt: Vec3 = [dir[0] * rad, dir[1] * rad, dir[2] * rad];
  return pushOutsideThroat(arcPt, shape, r, safeGap, scale);
}

function approxSegmentLength(
  p0: Vec3,
  p1: Vec3,
  shape: WormholeShapeId,
  radius: number,
  safeGap: number,
  scale: ShapeScale
): number {
  const STEPS = 8;
  let total = 0;
  let prev = p0;
  for (let i = 1; i <= STEPS; i++) {
    const curr = interpolateOutsideSegment(p0, p1, i / STEPS, shape, radius, safeGap, scale);
    total += dist3(prev, curr);
    prev = curr;
  }
  return total;
}

function forwardFromQuaternion(q: readonly [number, number, number, number]): Vec3 {
  // Rotate Three.js camera forward (0, 0, -1) by unit quaternion q = (x, y, z, w)
  const [x, y, z, w] = q;
  return normalize3([
    -2 * (x * z + w * y),
    -2 * (y * z - w * x),
    -(1 - 2 * (x * x + y * y)),
  ]);
}

export function interpolateKeyframe(
  start: CameraKeyframe,
  end: CameraKeyframe,
  t: number,
  mode: InterpolationMode,
  wormhole?: WormholeGeometry
): CameraKeyframe {
  const u = evaluateEasing(t, mode);
  const quaternion = slerpQuaternion(start.quaternion, end.quaternion, u);

  // Fallback when wormhole geometry is not supplied
  if (!wormhole) {
    if (u <= 0) {
      return {
        position: [...start.position],
        quaternion: [...start.quaternion],
        target: [...start.target],
        room: start.room,
      };
    }
    if (u >= 1) {
      return {
        position: [...end.position],
        quaternion: [...end.quaternion],
        target: [...end.target],
        room: end.room,
      };
    }
    const position = lerp3(start.position, end.position, u);
    const target = lerp3(start.target, end.target, u);
    const room = start.room === end.room ? start.room : u < 0.5 ? start.room : end.room;
    return { position, quaternion, target, room };
  }

  const r = Math.max(wormhole.radius, 1e-4);
  const shapeScale: ShapeScale = wormhole.scale ?? DEFAULT_SHAPE_SCALE;
  const effR = mouthEffectiveRadius(r, shapeScale);
  // Strictly greater than shader hitEps (0.02 * effR) so rays never start inside the throat SDF.
  // entryMargin sits right outside hitEps; exitMargin matches hitEps + shader head-on exitStep (0.06 * effR)
  // so gravitational lensing magnification is continuous across the universe transition.
  const safeGap = Math.max(effR * 0.024, 0.0025);
  const entryMargin = safeGap;

  const rawA = sub3(start.position, wormhole.center);
  const rawB = sub3(end.position, wormhole.center);
  const gapA = wormholeSdf(wormhole.shape, rawA[0], rawA[1], rawA[2], r, shapeScale);
  const gapB = wormholeSdf(wormhole.shape, rawB[0], rawB[1], rawB[2], r, shapeScale);
  const aOutside = gapA >= safeGap;
  const bOutside = gapB >= safeGap;
  const pA = aOutside ? rawA : pushOutsideThroat(rawA, wormhole.shape, r, safeGap, shapeScale);
  const pB = bOutside ? rawB : pushOutsideThroat(rawB, wormhole.shape, r, safeGap, shapeScale);

  if (u <= 0) {
    return {
      position: aOutside ? [...start.position] : add3(wormhole.center, pA),
      quaternion: [...start.quaternion],
      target: [...start.target],
      room: start.room,
    };
  }
  if (u >= 1) {
    return {
      position: bOutside ? [...end.position] : add3(wormhole.center, pB),
      quaternion: [...end.quaternion],
      target: [...end.target],
      room: end.room,
    };
  }

  // Same-universe trajectory: avoid slicing through the wormhole interior
  if (start.room === end.room) {
    const localPos = interpolateOutsideSegment(pA, pB, u, wormhole.shape, r, safeGap, shapeScale);
    const position = add3(wormhole.center, localPos);
    const target = lerp3(start.target, end.target, u);
    return { position, quaternion, target, room: start.room };
  }

  // Cross-universe trajectory:
  // Leg 1 approaches the throat surface in start.room (always outside hitEps),
  // crosses instantaneously across R_z into end.room at matched exitMargin while
  // transporting camera orientation by the exact shader transition q_T, and
  // Leg 2 continues smoothly to end.position.
  const endClearance = wormholeSdf(wormhole.shape, pB[0], pB[1], pB[2], r, shapeScale);
  const targetExitMargin = Math.max(effR * 0.08, 0.008);
  const exitMargin = Math.max(safeGap, Math.min(targetExitMargin, endClearance));

  const { entryPos, exitPos, entryNormal } = computeThroatPortals(
    pA,
    pB,
    wormhole.shape,
    r,
    entryMargin,
    exitMargin,
    shapeScale
  );
  const len1 = approxSegmentLength(pA, entryPos, wormhole.shape, r, safeGap, shapeScale);
  const len2 = approxSegmentLength(exitPos, pB, wormhole.shape, r, safeGap, shapeScale);
  const totalLen = len1 + len2;
  const uCross = totalLen > 1e-6 ? Math.min(0.98, Math.max(0.02, len1 / totalLen)) : 0.5;

  // Transport end.quaternion back into start.room via q_T^{-1} so orientation
  // is continuously parallel-transported through the throat at u = uCross.
  const qTransition = throatTransitionQuaternion(entryNormal[0], entryNormal[1], entryNormal[2]);
  const endQuatInStartFrame = quatMultiply(quatConjugate(qTransition), end.quaternion);
  const baseQuat = slerpQuaternion(start.quaternion, endQuatInStartFrame, u);

  let localPos: Vec3;
  let room: number;
  let crossQuat: Quat;
  if (u < uCross) {
    const legT = u / uCross;
    localPos = interpolateOutsideSegment(pA, entryPos, legT, wormhole.shape, r, safeGap, shapeScale);
    room = start.room;
    crossQuat = baseQuat;
  } else {
    const legT = (u - uCross) / (1 - uCross);
    localPos = interpolateOutsideSegment(exitPos, pB, legT, wormhole.shape, r, safeGap, shapeScale);
    room = end.room;
    crossQuat = quatMultiply(qTransition, baseQuat);
  }

  const position = add3(wormhole.center, localPos);
  const startPivotDist = Math.max(dist3(start.position, start.target), 0.5);
  const endPivotDist = Math.max(dist3(end.position, end.target), 0.5);
  const pivotDist = startPivotDist + (endPivotDist - startPivotDist) * u;
  const fwd = forwardFromQuaternion(crossQuat);
  const target: Vec3 = [
    position[0] + fwd[0] * pivotDist,
    position[1] + fwd[1] * pivotDist,
    position[2] + fwd[2] * pivotDist,
  ];

  return { position, quaternion: crossQuat, target, room };
}

// ============================================================================
// Stage 4: Multiverse Graph-Aware Multi-Hop Routing & Multi-Keyframe Timeline
// ============================================================================

export interface UniverseMouthRef {
  wormhole: WormholeEdge;
  end: 'A' | 'B';
  mouth: WormholeMouthPlacement;
  partnerMouth: WormholeMouthPlacement;
  fromUniverseIdx: number;
  toUniverseIdx: number;
}

export interface PortalHop {
  wormhole: WormholeEdge;
  entryEnd: 'A' | 'B';
  entryMouth: WormholeMouthPlacement;
  exitMouth: WormholeMouthPlacement;
  fromUniverseIdx: number;
  toUniverseIdx: number;
}

export interface WormholeRouteResult {
  reachable: boolean;
  hops: PortalHop[];
}

export function resolveKeyframeUniverseIndex(
  keyframe: CameraKeyframe,
  graph: MultiverseGraph
): number {
  const count = graph.universes.length;
  if (count === 0) return 0;
  if (keyframe.universeId) {
    const byId = graph.universes.findIndex((u) => u.id === keyframe.universeId);
    if (byId >= 0) return byId;
  }
  const idx = Math.round(keyframe.room);
  return Math.max(0, Math.min(count - 1, Number.isFinite(idx) ? idx : 0));
}

function getMouthsInUniverse(graph: MultiverseGraph, universeIdx: number): UniverseMouthRef[] {
  const u = graph.universes[universeIdx];
  if (!u) return [];
  const indexById = new Map<string, number>();
  graph.universes.forEach((node, idx) => indexById.set(node.id, idx));

  const out: UniverseMouthRef[] = [];
  for (const wh of graph.wormholes) {
    const idxA = indexById.get(wh.mouthA.universeId);
    const idxB = indexById.get(wh.mouthB.universeId);
    if (idxA === undefined || idxB === undefined) continue;
    if (idxA === universeIdx) {
      out.push({
        wormhole: wh,
        end: 'A',
        mouth: wh.mouthA,
        partnerMouth: wh.mouthB,
        fromUniverseIdx: idxA,
        toUniverseIdx: idxB,
      });
    }
    if (idxB === universeIdx) {
      out.push({
        wormhole: wh,
        end: 'B',
        mouth: wh.mouthB,
        partnerMouth: wh.mouthA,
        fromUniverseIdx: idxB,
        toUniverseIdx: idxA,
      });
    }
  }
  return out;
}

/**
 * Finds the shortest-hop wormhole route between `fromUniverseIdx` and `toUniverseIdx`,
 * breaking ties among equal-hop paths by minimum total Euclidean travel distance.
 */
export function findWormholeRoute(
  graph: MultiverseGraph,
  fromUniverseIdx: number,
  toUniverseIdx: number,
  startPos: readonly [number, number, number],
  endPos: readonly [number, number, number]
): WormholeRouteResult {
  const uCount = graph.universes.length;
  if (fromUniverseIdx < 0 || fromUniverseIdx >= uCount || toUniverseIdx < 0 || toUniverseIdx >= uCount) {
    return { reachable: false, hops: [] };
  }
  if (fromUniverseIdx === toUniverseIdx) {
    return { reachable: true, hops: [] };
  }

  // Adjacency list of directed portal hops (excluding self-loops for cross-universe shortest paths)
  const adj: PortalHop[][] = Array.from({ length: uCount }, () => []);
  for (let uIdx = 0; uIdx < uCount; uIdx++) {
    for (const ref of getMouthsInUniverse(graph, uIdx)) {
      if (ref.toUniverseIdx === uIdx) continue;
      adj[uIdx].push({
        wormhole: ref.wormhole,
        entryEnd: ref.end,
        entryMouth: ref.mouth,
        exitMouth: ref.partnerMouth,
        fromUniverseIdx: ref.fromUniverseIdx,
        toUniverseIdx: ref.toUniverseIdx,
      });
    }
  }

  interface SearchState {
    uIdx: number;
    pos: Vec3;
    hops: PortalHop[];
    travelDist: number;
  }

  // BFS by hop depth so we always find minimum hop count, keeping best-cost chain to each universe
  let bestDepth = Infinity;
  let bestCost = Infinity;
  let bestHops: PortalHop[] | null = null;

  const bestSeenDepth = new Array<number>(uCount).fill(Infinity);
  bestSeenDepth[fromUniverseIdx] = 0;

  let frontier: SearchState[] = [
    { uIdx: fromUniverseIdx, pos: [startPos[0], startPos[1], startPos[2]], hops: [], travelDist: 0 },
  ];

  const MAX_HOPS = Math.min(uCount, 8);
  for (let depth = 1; depth <= MAX_HOPS && frontier.length > 0; depth++) {
    if (depth > bestDepth) break;
    const nextFrontier: SearchState[] = [];
    for (const state of frontier) {
      for (const hop of adj[state.uIdx]) {
        // Avoid revisiting universes already on this path
        if (hop.toUniverseIdx === fromUniverseIdx) continue;
        if (state.hops.some((h) => h.toUniverseIdx === hop.toUniverseIdx)) continue;
        if (bestSeenDepth[hop.toUniverseIdx] < depth) continue;

        const legDist = dist3(state.pos, hop.entryMouth.position);
        const nextDist = state.travelDist + legDist;
        const nextHops = [...state.hops, hop];

        if (hop.toUniverseIdx === toUniverseIdx) {
          const totalDist = nextDist + dist3(hop.exitMouth.position, endPos);
          if (depth < bestDepth || (depth === bestDepth && totalDist < bestCost)) {
            bestDepth = depth;
            bestCost = totalDist;
            bestHops = nextHops;
          }
        } else {
          bestSeenDepth[hop.toUniverseIdx] = depth;
          nextFrontier.push({
            uIdx: hop.toUniverseIdx,
            pos: [hop.exitMouth.position[0], hop.exitMouth.position[1], hop.exitMouth.position[2]],
            hops: nextHops,
            travelDist: nextDist,
          });
        }
      }
    }
    frontier = nextFrontier;
  }

  if (!bestHops) {
    return { reachable: false, hops: [] };
  }
  return { reachable: true, hops: bestHops };
}

function worldToMouthLocal(mouth: WormholeMouthPlacement, worldPos: Vec3): Vec3 {
  const q = normalizeQuaternion(mouth.quaternion);
  const rel = sub3(worldPos, mouth.position);
  return quatRotateInvVec3(q, rel);
}

function mouthLocalToWorld(mouth: WormholeMouthPlacement, localPos: Vec3): Vec3 {
  const q = normalizeQuaternion(mouth.quaternion);
  const rot = quatRotateVec3(q, localPos);
  return add3(mouth.position, rot);
}

/**
 * Ensures `worldPos` is strictly outside every wormhole mouth in `universeIdx`.
 */
function pushOutsideAllMouthsInUniverse(
  worldPos: Vec3,
  mouths: readonly UniverseMouthRef[],
  extraMarginScale = 1
): Vec3 {
  let curr: Vec3 = [worldPos[0], worldPos[1], worldPos[2]];
  for (let pass = 0; pass < 3; pass++) {
    let adjusted = false;
    for (const ref of mouths) {
      const gap = evaluateMouthSdfWorld(
        ref.mouth,
        ref.wormhole.shape,
        ref.wormhole.radius,
        ref.wormhole.scale,
        curr
      );
      const minGap = mouthSafeGap(ref.wormhole.radius, ref.wormhole.scale) * extraMarginScale;
      if (gap < minGap) {
        const local = worldToMouthLocal(ref.mouth, curr);
        const pushedLocal = pushOutsideThroat(
          local,
          ref.wormhole.shape,
          ref.wormhole.radius,
          minGap,
          ref.wormhole.scale
        );
        curr = mouthLocalToWorld(ref.mouth, pushedLocal);
        adjusted = true;
      }
    }
    if (!adjusted) break;
  }
  return curr;
}

/**
 * Interpolates a trajectory from `p0` to `p1` inside `universeIdx` while steering smoothly
 * around EVERY wormhole mouth present in that universe.
 */
function interpolateSegmentOutsideAllMouths(
  p0: Vec3,
  p1: Vec3,
  s: number,
  mouths: readonly UniverseMouthRef[]
): Vec3 {
  const t = clamp01(s);
  if (t <= 0) return [p0[0], p0[1], p0[2]];
  if (t >= 1) return [p1[0], p1[1], p1[2]];
  if (mouths.length === 0) return lerp3(p0, p1, t);

  // Check whether the straight segment clips any mouth in this universe
  let blocking: UniverseMouthRef | null = null;
  let worstPenetration = 0;
  const SAMPLES = 16;
  for (let i = 1; i < SAMPLES; i++) {
    const pt = lerp3(p0, p1, i / SAMPLES);
    for (const ref of mouths) {
      const minGap = mouthSafeGap(ref.wormhole.radius, ref.wormhole.scale);
      const gap = evaluateMouthSdfWorld(
        ref.mouth,
        ref.wormhole.shape,
        ref.wormhole.radius,
        ref.wormhole.scale,
        pt
      );
      if (gap < minGap) {
        const pen = minGap - gap;
        if (pen > worstPenetration) {
          worstPenetration = pen;
          blocking = ref;
        }
      }
    }
  }

  let candidate: Vec3;
  if (!blocking) {
    candidate = lerp3(p0, p1, t);
  } else {
    // Arc smoothly around the primary obstructing mouth in its oriented local frame
    const local0 = worldToMouthLocal(blocking.mouth, p0);
    const local1 = worldToMouthLocal(blocking.mouth, p1);
    const minGap = mouthSafeGap(blocking.wormhole.radius, blocking.wormhole.scale);
    const localPt = interpolateOutsideSegment(
      local0,
      local1,
      t,
      blocking.wormhole.shape,
      blocking.wormhole.radius,
      minGap,
      blocking.wormhole.scale
    );
    candidate = mouthLocalToWorld(blocking.mouth, localPt);
  }

  // Enforce clearance from all other mouths in the same universe
  return pushOutsideAllMouthsInUniverse(candidate, mouths);
}

function approxWorldLegLength(
  p0: Vec3,
  p1: Vec3,
  mouths: readonly UniverseMouthRef[]
): number {
  const STEPS = 8;
  let total = 0;
  let prev = p0;
  for (let i = 1; i <= STEPS; i++) {
    const curr = interpolateSegmentOutsideAllMouths(p0, p1, i / STEPS, mouths);
    total += dist3(prev, curr);
    prev = curr;
  }
  return total;
}

interface CompiledLeg {
  universeIdx: number;
  universeId: string;
  startPos: Vec3;
  endPos: Vec3;
  mouths: UniverseMouthRef[];
  length: number;
  /** Cumulative world-space orientation transport quaternion active on this leg. */
  cumTransitionQuat: Quat;
  /** Normalized cumulative end u in [0, 1] within the parent keyframe segment. */
  uStart: number;
  uEnd: number;
}

export interface CompiledTimelineSegment {
  index: number;
  startKeyframe: CameraKeyframe;
  endKeyframe: CameraKeyframe;
  startUniverseIdx: number;
  endUniverseIdx: number;
  reachable: boolean;
  hops: PortalHop[];
  legs: CompiledLeg[];
  totalLength: number;
  /** Pullback of endKeyframe.quaternion into the start frame via cumulative Q_k^{-1}. */
  endQuatInStartFrame: Quat;
  /** Global timeline span [tStart, tEnd] within [0, 1]. */
  tStart: number;
  tEnd: number;
}

export interface MultiverseTimeline {
  keyframes: CameraKeyframe[];
  mode: InterpolationMode;
  segments: CompiledTimelineSegment[];
  totalHops: number;
  disconnectedSegmentIndices: number[];
  universeSequence: number[];
}

function compileSegmentBetweenKeyframes(
  index: number,
  rawStart: CameraKeyframe,
  rawEnd: CameraKeyframe,
  graph: MultiverseGraph
): CompiledTimelineSegment {
  const startU = resolveKeyframeUniverseIndex(rawStart, graph);
  const endU = resolveKeyframeUniverseIndex(rawEnd, graph);
  const startMouths = getMouthsInUniverse(graph, startU);
  const endMouths = getMouthsInUniverse(graph, endU);

  const safeStartPos = pushOutsideAllMouthsInUniverse(
    [rawStart.position[0], rawStart.position[1], rawStart.position[2]],
    startMouths
  );
  const safeEndPos = pushOutsideAllMouthsInUniverse(
    [rawEnd.position[0], rawEnd.position[1], rawEnd.position[2]],
    endMouths
  );

  const startKf: CameraKeyframe = {
    ...rawStart,
    room: startU,
    universeId: graph.universes[startU]?.id,
    position: safeStartPos,
    quaternion: normalizeQuaternion(rawStart.quaternion),
  };
  const endKf: CameraKeyframe = {
    ...rawEnd,
    room: endU,
    universeId: graph.universes[endU]?.id,
    position: safeEndPos,
    quaternion: normalizeQuaternion(rawEnd.quaternion),
  };

  // Same-universe segment: 1 leg, 0 hops
  if (startU === endU) {
    const length = Math.max(approxWorldLegLength(safeStartPos, safeEndPos, startMouths), 0.05);
    return {
      index,
      startKeyframe: startKf,
      endKeyframe: endKf,
      startUniverseIdx: startU,
      endUniverseIdx: endU,
      reachable: true,
      hops: [],
      legs: [
        {
          universeIdx: startU,
          universeId: graph.universes[startU]?.id ?? String(startU),
          startPos: safeStartPos,
          endPos: safeEndPos,
          mouths: startMouths,
          length,
          cumTransitionQuat: [0, 0, 0, 1],
          uStart: 0,
          uEnd: 1,
        },
      ],
      totalLength: length,
      endQuatInStartFrame: endKf.quaternion,
      tStart: 0,
      tEnd: 1,
    };
  }

  const route = findWormholeRoute(graph, startU, endU, safeStartPos, safeEndPos);

  // Unreachable universes: clean direct cut at midpoint without crashing
  if (!route.reachable || route.hops.length === 0) {
    const midLen = Math.max(dist3(safeStartPos, safeEndPos), 0.5);
    return {
      index,
      startKeyframe: startKf,
      endKeyframe: endKf,
      startUniverseIdx: startU,
      endUniverseIdx: endU,
      reachable: false,
      hops: [],
      legs: [
        {
          universeIdx: startU,
          universeId: graph.universes[startU]?.id ?? String(startU),
          startPos: safeStartPos,
          endPos: safeStartPos,
          mouths: startMouths,
          length: midLen * 0.5,
          cumTransitionQuat: [0, 0, 0, 1],
          uStart: 0,
          uEnd: 0.5,
        },
        {
          universeIdx: endU,
          universeId: graph.universes[endU]?.id ?? String(endU),
          startPos: safeEndPos,
          endPos: safeEndPos,
          mouths: endMouths,
          length: midLen * 0.5,
          cumTransitionQuat: [0, 0, 0, 1],
          uStart: 0.5,
          uEnd: 1,
        },
      ],
      totalLength: midLen,
      endQuatInStartFrame: endKf.quaternion,
      tStart: 0,
      tEnd: 1,
    };
  }

  // Build k + 1 legs across k oriented portal crossings
  const legs: CompiledLeg[] = [];
  let currPos = safeStartPos;
  let currU = startU;
  let cumQ: Quat = [0, 0, 0, 1];

  for (let h = 0; h < route.hops.length; h++) {
    const hop = route.hops[h];
    const wh = hop.wormhole;
    const effR = mouthEffectiveRadius(wh.radius, wh.scale);
    const safeGap = mouthSafeGap(wh.radius, wh.scale);
    const entryMargin = safeGap;

    const nextTargetWorld =
      h + 1 < route.hops.length ? route.hops[h + 1].entryMouth.position : safeEndPos;

    // Transform approach position and destination reference into entry/exit local frames
    const localStart = worldToMouthLocal(hop.entryMouth, currPos);
    const localNext = worldToMouthLocal(hop.exitMouth, nextTargetWorld);
    const nextClearance = wormholeSdf(
      wh.shape,
      localNext[0],
      localNext[1],
      localNext[2],
      wh.radius,
      wh.scale
    );
    const targetExitMargin = Math.max(effR * 0.08, 0.008);
    const exitMargin = Math.max(
      safeGap,
      nextClearance > safeGap ? Math.min(targetExitMargin, nextClearance) : targetExitMargin
    );

    const { entryPos: entryLocal, exitPos: exitLocal, entryNormal } = computeThroatPortals(
      localStart,
      localNext,
      wh.shape,
      wh.radius,
      entryMargin,
      exitMargin,
      wh.scale
    );

    const mouthsHere = getMouthsInUniverse(graph, currU);
    const entryWorld = pushOutsideAllMouthsInUniverse(
      mouthLocalToWorld(hop.entryMouth, entryLocal),
      mouthsHere
    );
    const legLen = Math.max(approxWorldLegLength(currPos, entryWorld, mouthsHere), 0.05);

    legs.push({
      universeIdx: currU,
      universeId: graph.universes[currU]?.id ?? String(currU),
      startPos: currPos,
      endPos: entryWorld,
      mouths: mouthsHere,
      length: legLen,
      cumTransitionQuat: cumQ,
      uStart: 0,
      uEnd: 0,
    });

    // Exact world-space parallel transport across this oriented throat:
    // q_world_hop = q_exit * q_T(entryNormal_local) * q_entry^{-1}
    const qEntry = normalizeQuaternion(hop.entryMouth.quaternion);
    const qExit = normalizeQuaternion(hop.exitMouth.quaternion);
    const qLocalT = throatTransitionQuaternion(entryNormal[0], entryNormal[1], entryNormal[2]);
    const qWorldHop = quatMultiply(qExit, quatMultiply(qLocalT, quatConjugate(qEntry)));
    cumQ = quatMultiply(qWorldHop, cumQ);

    const nextMouths = getMouthsInUniverse(graph, hop.toUniverseIdx);
    currPos = pushOutsideAllMouthsInUniverse(
      mouthLocalToWorld(hop.exitMouth, exitLocal),
      nextMouths
    );
    currU = hop.toUniverseIdx;
  }

  // Final leg inside endU from last exit portal to safeEndPos
  const finalMouths = getMouthsInUniverse(graph, currU);
  const finalLen = Math.max(approxWorldLegLength(currPos, safeEndPos, finalMouths), 0.05);
  legs.push({
    universeIdx: currU,
    universeId: graph.universes[currU]?.id ?? String(currU),
    startPos: currPos,
    endPos: safeEndPos,
    mouths: finalMouths,
    length: finalLen,
    cumTransitionQuat: cumQ,
    uStart: 0,
    uEnd: 1,
  });

  // Assign proportional [uStart, uEnd] spans along the segment based on physical leg length
  const totalLength = legs.reduce((acc, l) => acc + l.length, 0);
  let cursor = 0;
  for (let i = 0; i < legs.length; i++) {
    const share = totalLength > 1e-6 ? legs[i].length / totalLength : 1 / legs.length;
    legs[i].uStart = cursor;
    cursor = i === legs.length - 1 ? 1 : Math.min(0.995, cursor + Math.max(share, 0.01));
    legs[i].uEnd = cursor;
  }

  const endQuatInStartFrame = quatMultiply(quatConjugate(cumQ), endKf.quaternion);

  return {
    index,
    startKeyframe: startKf,
    endKeyframe: endKf,
    startUniverseIdx: startU,
    endUniverseIdx: endU,
    reachable: true,
    hops: route.hops,
    legs,
    totalLength,
    endQuatInStartFrame,
    tStart: 0,
    tEnd: 1,
  };
}

/**
 * Compiles an ordered sequence of 2..N camera keyframes against a MultiverseGraph
 * into a reusable, collision-free, portal-routed animation timeline.
 */
export function buildMultiverseTimeline(
  keyframes: readonly CameraKeyframe[],
  graph: MultiverseGraph,
  mode: InterpolationMode
): MultiverseTimeline {
  const segments: CompiledTimelineSegment[] = [];
  const disconnectedSegmentIndices: number[] = [];
  let totalHops = 0;
  const universeSequence: number[] = [];

  if (keyframes.length > 0) {
    universeSequence.push(resolveKeyframeUniverseIndex(keyframes[0], graph));
  }

  for (let i = 0; i < keyframes.length - 1; i++) {
    const seg = compileSegmentBetweenKeyframes(i, keyframes[i], keyframes[i + 1], graph);
    segments.push(seg);
    totalHops += seg.hops.length;
    if (!seg.reachable) {
      disconnectedSegmentIndices.push(i);
    }
    for (const hop of seg.hops) {
      universeSequence.push(hop.toUniverseIdx);
    }
    const lastU = universeSequence[universeSequence.length - 1];
    if (lastU !== seg.endUniverseIdx) {
      universeSequence.push(seg.endUniverseIdx);
    }
  }

  // Distribute timeline parameter [0, 1] proportionally to each segment's path length
  // (with a sensible minimum weight so in-place rotation keyframes still take real time)
  const weights = segments.map((seg) => Math.max(seg.totalLength, 0.75));
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  let tCursor = 0;
  for (let i = 0; i < segments.length; i++) {
    const span = totalWeight > 1e-6 ? weights[i] / totalWeight : 1 / Math.max(1, segments.length);
    segments[i].tStart = tCursor;
    tCursor = i === segments.length - 1 ? 1 : tCursor + span;
    segments[i].tEnd = tCursor;
  }

  return {
    keyframes: keyframes.map((kf) => ({ ...kf })),
    mode,
    segments,
    totalHops,
    disconnectedSegmentIndices,
    universeSequence,
  };
}

function sampleCompiledSegment(
  seg: CompiledTimelineSegment,
  localT: number,
  mode: InterpolationMode
): CameraKeyframe {
  const u = evaluateEasing(localT, mode);

  if (u <= 0) {
    return {
      ...seg.startKeyframe,
      position: [...seg.startKeyframe.position],
      quaternion: [...seg.startKeyframe.quaternion],
      target: [...seg.startKeyframe.target],
    };
  }
  if (u >= 1) {
    return {
      ...seg.endKeyframe,
      position: [...seg.endKeyframe.position],
      quaternion: [...seg.endKeyframe.quaternion],
      target: [...seg.endKeyframe.target],
    };
  }

  // Locate active leg
  let leg = seg.legs[seg.legs.length - 1];
  for (let i = 0; i < seg.legs.length; i++) {
    if (u < seg.legs[i].uEnd || i === seg.legs.length - 1) {
      leg = seg.legs[i];
      break;
    }
  }

  const span = Math.max(leg.uEnd - leg.uStart, 1e-6);
  const legT = clamp01((u - leg.uStart) / span);
  const position = interpolateSegmentOutsideAllMouths(leg.startPos, leg.endPos, legT, leg.mouths);

  // Same-universe direct slerp
  if (seg.hops.length === 0) {
    const quaternion = slerpQuaternion(
      seg.startKeyframe.quaternion,
      seg.endKeyframe.quaternion,
      u
    );
    const target = lerp3(seg.startKeyframe.target, seg.endKeyframe.target, u);
    return {
      position,
      quaternion,
      target,
      room: leg.universeIdx,
      universeId: leg.universeId,
    };
  }

  // Portal-crossed parallel-transported orientation
  const baseQuat = slerpQuaternion(
    seg.startKeyframe.quaternion,
    seg.endQuatInStartFrame,
    u
  );
  const crossQuat = quatMultiply(leg.cumTransitionQuat, baseQuat);

  const startPivotDist = Math.max(dist3(seg.startKeyframe.position, seg.startKeyframe.target), 0.5);
  const endPivotDist = Math.max(dist3(seg.endKeyframe.position, seg.endKeyframe.target), 0.5);
  const pivotDist = startPivotDist + (endPivotDist - startPivotDist) * u;
  const fwd = forwardFromQuaternion(crossQuat);
  const target: Vec3 = [
    position[0] + fwd[0] * pivotDist,
    position[1] + fwd[1] * pivotDist,
    position[2] + fwd[2] * pivotDist,
  ];

  return {
    position,
    quaternion: crossQuat,
    target,
    room: leg.universeIdx,
    universeId: leg.universeId,
  };
}

/**
 * Evaluates a compiled multi-keyframe multiverse timeline at normalized time `tNorm` in [0, 1].
 */
export function sampleMultiverseTimeline(
  timeline: MultiverseTimeline,
  tNorm: number
): CameraKeyframe {
  const t = clamp01(tNorm);
  if (timeline.segments.length === 0) {
    const only = timeline.keyframes[0];
    if (only) {
      return {
        ...only,
        position: [...only.position],
        quaternion: [...only.quaternion],
        target: [...only.target],
      };
    }
    return {
      position: [0, 0, -0.15],
      quaternion: [0, 0, 0, 1],
      target: [0, 0, 3],
      room: 0,
    };
  }

  if (t <= 0) {
    return sampleCompiledSegment(timeline.segments[0], 0, timeline.mode);
  }
  if (t >= 1) {
    return sampleCompiledSegment(timeline.segments[timeline.segments.length - 1], 1, timeline.mode);
  }

  let seg = timeline.segments[timeline.segments.length - 1];
  for (let i = 0; i < timeline.segments.length; i++) {
    if (t < timeline.segments[i].tEnd || i === timeline.segments.length - 1) {
      seg = timeline.segments[i];
      break;
    }
  }
  const span = Math.max(seg.tEnd - seg.tStart, 1e-6);
  const localT = clamp01((t - seg.tStart) / span);
  return sampleCompiledSegment(seg, localT, timeline.mode);
}

/**
 * Samples polyline segments of the compiled camera trajectory that pass through
 * `universeIdx`, used by the 3D placement overlay to visualize the planned path.
 */
export function sampleTimelinePathInUniverse(
  timeline: MultiverseTimeline,
  universeIdx: number,
  samplesPerLeg = 24
): Array<Array<[number, number, number]>> {
  const polylines: Array<Array<[number, number, number]>> = [];
  const steps = Math.max(4, samplesPerLeg);

  for (const seg of timeline.segments) {
    for (const leg of seg.legs) {
      if (leg.universeIdx !== universeIdx) continue;
      if (dist3(leg.startPos, leg.endPos) < 1e-4) continue;
      const pts: Array<[number, number, number]> = [];
      for (let i = 0; i <= steps; i++) {
        pts.push(interpolateSegmentOutsideAllMouths(leg.startPos, leg.endPos, i / steps, leg.mouths));
      }
      polylines.push(pts);
    }
  }

  return polylines;
}

/**
 * Generates a cinematic, collision-free multi-universe camera tour visiting all
 * universes reachable from `startUniverseIdx` via the graph's wormhole bridges.
 */
export function generateMultiverseTourKeyframes(
  graph: MultiverseGraph,
  startUniverseIdx = 0
): CameraKeyframe[] {
  if (graph.universes.length === 0) return [];
  const safeStart = Math.max(0, Math.min(graph.universes.length - 1, startUniverseIdx));

  // Build BFS visiting order of reachable universes
  const visited = new Set<number>([safeStart]);
  const order: number[] = [safeStart];
  const queue: number[] = [safeStart];

  while (queue.length > 0) {
    const uIdx = queue.shift()!;
    const mouths = getMouthsInUniverse(graph, uIdx);
    for (const ref of mouths) {
      if (!visited.has(ref.toUniverseIdx)) {
        visited.add(ref.toUniverseIdx);
        order.push(ref.toUniverseIdx);
        queue.push(ref.toUniverseIdx);
      }
    }
  }

  const keyframes: CameraKeyframe[] = [];
  const makeVantagePose = (
    uIdx: number,
    focusMouth: UniverseMouthRef | undefined,
    offsetSign: number,
    label: string
  ): CameraKeyframe => {
    const uNode = graph.universes[uIdx];
    const mouths = getMouthsInUniverse(graph, uIdx);
    let pos: Vec3 = [0, 0.25, -2.5 * offsetSign];
    let target: Vec3 = [0, 0, 3];

    if (focusMouth) {
      const m = focusMouth.mouth;
      const wh = focusMouth.wormhole;
      const maxExtent = Math.max(
        wh.radius * Math.max(wh.scale[0], wh.scale[1], wh.scale[2]),
        0.5
      );
      const q = normalizeQuaternion(m.quaternion);
      // Place camera cleanly in front of (or behind) the mouth along its local Z axis
      const localApproach: Vec3 = [0, maxExtent * 0.12, offsetSign * Math.max(2.6 * maxExtent, 2.2)];
      const worldApproach = add3(m.position, quatRotateVec3(q, localApproach));
      pos = pushOutsideAllMouthsInUniverse(worldApproach, mouths, 1.5);
      target = [m.position[0], m.position[1], m.position[2]];
    }

    return {
      id: `tour-${keyframes.length + 1}-${uNode.id}`,
      label,
      universeId: uNode.id,
      room: uIdx,
      position: pos,
      quaternion: lookAtCameraQuaternion(pos, target),
      target,
    };
  };

  if (order.length === 1) {
    // Single connected universe: orbit smoothly around the first mouth (or origin)
    const mouths = getMouthsInUniverse(graph, safeStart);
    const first = mouths[0];
    keyframes.push(makeVantagePose(safeStart, first, -1, `${graph.universes[safeStart].name} · Front`));
    keyframes.push(makeVantagePose(safeStart, first, 1, `${graph.universes[safeStart].name} · Far Side`));
    return keyframes;
  }

  // Visit each reachable universe in order, framing the wormhole leading onward (or back)
  for (let i = 0; i < order.length; i++) {
    const uIdx = order[i];
    const nextU = order[(i + 1) % order.length];
    const mouths = getMouthsInUniverse(graph, uIdx);
    const onwardMouth = mouths.find((m) => m.toUniverseIdx === nextU) ?? mouths[0];
    const sign = i % 2 === 0 ? -1 : 1;
    keyframes.push(
      makeVantagePose(
        uIdx,
        onwardMouth,
        sign,
        `${graph.universes[uIdx].name}${i === 0 ? ' · Start' : ''}`
      )
    );
  }

  // Close the tour smoothly back in the starting universe if there are 3+ universes in a cycle
  if (order.length >= 3) {
    const startMouths = getMouthsInUniverse(graph, safeStart);
    const lastU = order[order.length - 1];
    const returnBridge = startMouths.find((m) => m.toUniverseIdx === lastU);
    if (returnBridge) {
      keyframes.push(
        makeVantagePose(
          safeStart,
          returnBridge,
          1,
          `${graph.universes[safeStart].name} · Return`
        )
      );
    }
  }

  return keyframes;
}

/**
 * Computes codec-safe even video dimensions matching the viewport aspect ratio,
 * capped so H.264/VP9 hardware & software encoders never reject the frame size.
 */
export function computeRenderDimensions(
  sourceWidth: number,
  sourceHeight: number,
  maxEdge: number = 1920
): { width: number; height: number } {
  const safeW = Math.max(2, Math.round(sourceWidth));
  const safeH = Math.max(2, Math.round(sourceHeight));
  const longest = Math.max(safeW, safeH);
  const scale = longest > maxEdge ? maxEdge / longest : 1;

  // H.264 / VP9 require even pixel dimensions
  const width = Math.max(2, Math.floor((safeW * scale) / 2) * 2);
  const height = Math.max(2, Math.floor((safeH * scale) / 2) * 2);
  return { width, height };
}
