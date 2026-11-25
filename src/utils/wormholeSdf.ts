// Signed distances for the wormhole throat. Every shape is centered at the
// origin and symmetric across the coordinate planes — in particular
// f(x, y, -z) = f(x, y, z) — so gluing across R_z = diag(1, 1, -1) maps any
// surface point to a matching surface point while keeping world +Y upright.
// Proportions are multiples of the radius slider and must match sdWormhole in
// wormholeShader.ts.

export const WORMHOLE_SHAPES = [
  { id: 0, label: 'Sphere', title: 'Round throat. The normal points away from the center.' },
  { id: 1, label: 'Cuboid', title: 'Rectangular box. The normal is constant on each face.' },
  { id: 2, label: 'Cylinder', title: 'Capped cylinder along Y.' },
  { id: 3, label: 'Torus', title: 'Ring with the hole facing the default view. The hole is outside the throat.' },
  { id: 4, label: 'Octahedron', title: 'Regular octahedron, eight triangular faces.' },
  { id: 5, label: 'Cone', title: 'Double cone, symmetric across both the XY and XZ planes.' },
] as const;

export type WormholeShapeId = (typeof WORMHOLE_SHAPES)[number]['id'];

export type WormholeShapeName = Lowercase<(typeof WORMHOLE_SHAPES)[number]['label']>;

const SHAPE_NAME_BY_ID: Record<WormholeShapeId, WormholeShapeName> = {
  0: 'sphere',
  1: 'cuboid',
  2: 'cylinder',
  3: 'torus',
  4: 'octahedron',
  5: 'cone',
};

const SHAPE_ID_BY_NAME: Record<string, WormholeShapeId> = {
  sphere: 0,
  cuboid: 1,
  cube: 1,
  box: 1,
  cylinder: 2,
  torus: 3,
  ring: 3,
  octahedron: 4,
  cone: 5,
  'double-cone': 5,
  'double cone': 5,
};

export function wormholeShapeIdToName(id: WormholeShapeId): WormholeShapeName {
  return SHAPE_NAME_BY_ID[id] ?? 'sphere';
}

/**
 * Parses a wormhole shape from either a human-readable name (e.g. "torus", "Cuboid")
 * or a legacy numeric ID (0..5). Returns null if the value is not a recognized shape.
 */
export function parseWormholeShape(value: unknown): WormholeShapeId | null {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized in SHAPE_ID_BY_NAME) {
      return SHAPE_ID_BY_NAME[normalized];
    }
    if (/^[0-5]$/.test(normalized)) {
      return Number(normalized) as WormholeShapeId;
    }
    return null;
  }
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 5) {
    return value as WormholeShapeId;
  }
  return null;
}

function sdBox(px: number, py: number, pz: number, bx: number, by: number, bz: number): number {
  const qx = Math.abs(px) - bx;
  const qy = Math.abs(py) - by;
  const qz = Math.abs(pz) - bz;
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
  const inside = Math.min(Math.max(qx, Math.max(qy, qz)), 0);
  return outside + inside;
}

function sdCappedCylinder(px: number, py: number, pz: number, height: number, radius: number): number {
  const dx = Math.hypot(px, pz) - radius;
  const dy = Math.abs(py) - height;
  return Math.min(Math.max(dx, dy), 0) + Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
}

// Hole along Z, so the default camera (looking down Z) sees the ring.
function sdTorus(px: number, py: number, pz: number, major: number, minor: number): number {
  return Math.hypot(Math.hypot(px, py) - major, pz) - minor;
}

function sdOctahedron(px: number, py: number, pz: number, size: number): number {
  const ax = Math.abs(px);
  const ay = Math.abs(py);
  const az = Math.abs(pz);
  const m = ax + ay + az - size;
  let qx = ax;
  let qy = ay;
  let qz = az;
  if (3 * ax < m) {
    qx = ax;
    qy = ay;
    qz = az;
  } else if (3 * ay < m) {
    qx = ay;
    qy = az;
    qz = ax;
  } else if (3 * az < m) {
    qx = az;
    qy = ax;
    qz = ay;
  } else {
    return m * 0.57735026919;
  }
  const k = Math.min(Math.max(0.5 * (qz - qy + size), 0), size);
  return Math.hypot(qx, qy - size + k, qz - k);
}

function distToSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const denom = abx * abx + aby * aby;
  const t = denom > 0 ? Math.min(1, Math.max(0, ((px - ax) * abx + (py - ay) * aby) / denom)) : 0;
  return Math.hypot(px - ax - abx * t, py - ay - aby * t);
}

// Solid of revolution of the two slanted sides. Apexes at y = ±height, equator radius `radius`.
function sdBicone(px: number, py: number, pz: number, height: number, radius: number): number {
  const rho = Math.hypot(px, pz);
  const upper = distToSegment(rho, py, 0, height, radius, 0);
  const lower = distToSegment(rho, py, 0, -height, radius, 0);
  const limit = radius * (1 - Math.abs(py) / height);
  const inside = Math.abs(py) <= height && rho <= limit;
  return inside ? -Math.min(upper, lower) : Math.min(upper, lower);
}

export type ShapeScale = readonly [number, number, number];

export const DEFAULT_SHAPE_SCALE: [number, number, number] = [1, 1, 1];

export function wormholeSdf(
  shape: number,
  x: number,
  y: number,
  z: number,
  radius: number,
  scale: ShapeScale = DEFAULT_SHAPE_SCALE
): number {
  const r = Math.max(radius, 1e-4);
  const sx = Math.max(scale[0], 0.05);
  const sy = Math.max(scale[1], 0.05);
  const sz = Math.max(scale[2], 0.05);
  if (shape === 1) return sdBox(x, y, z, 1.2 * r * sx, 0.72 * r * sy, 0.9 * r * sz);
  const qx = x / sx;
  const qy = y / sy;
  const qz = z / sz;
  const sMin = Math.min(sx, sy, sz);
  if (shape === 2) return sMin * sdCappedCylinder(qx, qy, qz, 1.45 * r, 0.85 * r);
  if (shape === 3) return sMin * sdTorus(qx, qy, qz, r, 0.36 * r);
  if (shape === 4) return sMin * sdOctahedron(qx, qy, qz, 1.55 * r);
  if (shape === 5) return sMin * sdBicone(qx, qy, qz, 1.5 * r, r);
  return sMin * (Math.hypot(qx, qy, qz) - r);
}

export function wormholeNormal(
  shape: number,
  x: number,
  y: number,
  z: number,
  radius: number,
  scale: ShapeScale = DEFAULT_SHAPE_SCALE
): [number, number, number] {
  const sMin = Math.min(Math.max(scale[0], 0.05), Math.max(scale[1], 0.05), Math.max(scale[2], 0.05));
  const e = Math.max(radius * sMin * 1.5e-3, 1e-4);
  const dx = wormholeSdf(shape, x + e, y, z, radius, scale) - wormholeSdf(shape, x - e, y, z, radius, scale);
  const dy = wormholeSdf(shape, x, y + e, z, radius, scale) - wormholeSdf(shape, x, y - e, z, radius, scale);
  const dz = wormholeSdf(shape, x, y, z + e, radius, scale) - wormholeSdf(shape, x, y, z - e, radius, scale);
  const glen = Math.hypot(dx, dy, dz);
  if (glen > 1e-8) return [dx / glen, dy / glen, dz / glen];
  const plen = Math.hypot(x, y, z);
  if (plen > 1e-8) return [x / plen, y / plen, z / plen];
  return [0, 0, 1];
}

/**
 * Maps a ray or camera direction across the throat surface with outward unit
 * normal (nx, ny, nz) using the orientation-preserving SO(3) transition
 * T(d) = R_z * reflect(d, n), where R_z = diag(1, 1, -1).
 */
export function glueDirectionThroughThroat(
  dx: number,
  dy: number,
  dz: number,
  nx: number,
  ny: number,
  nz: number
): [number, number, number] {
  const dn = dx * nx + dy * ny + dz * nz;
  const rx = dx - 2 * dn * nx;
  const ry = dy - 2 * dn * ny;
  const rz = -(dz - 2 * dn * nz);
  const len = Math.hypot(rx, ry, rz);
  if (len > 1e-8) return [rx / len, ry / len, rz / len];
  return [nx, ny, -nz];
}

/**
 * Exact unit quaternion [x, y, z, w] for the SO(3) throat transition
 * T(v) = R_z * reflect(v, n). Since reflect(v, n) = n v n and R_z(w) = k w k,
 * T(v) = (k n) v (k n)^(-1), giving q = (ny, -nx, 0, nz) (sign-flipped so w >= 0).
 */
export function throatTransitionQuaternion(
  nx: number,
  ny: number,
  nz: number
): [number, number, number, number] {
  const nLen = Math.hypot(nx, ny, nz);
  const ux = nLen > 1e-8 ? nx / nLen : 0;
  const uy = nLen > 1e-8 ? ny / nLen : 0;
  const uz = nLen > 1e-8 ? nz / nLen : 1;
  if (uz < 0) {
    return [-uy, ux, 0, -uz];
  }
  return [uy, -ux, 0, uz];
}

/**
 * Closest point on the throat, then its R_z-reflected portal point (x, y, -z),
 * pushed just outside. `normal` is the outward normal at the entry point, for
 * the ray gluing.
 */
export function throatCrossing(
  x: number,
  y: number,
  z: number,
  shape: number,
  radius: number,
  scale: ShapeScale = DEFAULT_SHAPE_SCALE
): { exitX: number; exitY: number; exitZ: number; nx: number; ny: number; nz: number } {
  const r = Math.max(radius, 1e-4);
  const sMin = Math.min(Math.max(scale[0], 0.05), Math.max(scale[1], 0.05), Math.max(scale[2], 0.05));
  const effR = Math.max(r * sMin, 1e-4);
  let sx = x;
  let sy = y;
  let sz = z;
  let sn = wormholeNormal(shape, sx, sy, sz, r, scale);
  for (let i = 0; i < 6; i++) {
    const f = wormholeSdf(shape, sx, sy, sz, r, scale);
    sn = wormholeNormal(shape, sx, sy, sz, r, scale);
    if (Math.abs(f) < effR * 1e-3) break;
    sx -= sn[0] * f;
    sy -= sn[1] * f;
    sz -= sn[2] * f;
  }
  let ex = sx;
  let ey = sy;
  let ez = -sz;
  for (let i = 0; i < 5; i++) {
    const ef = wormholeSdf(shape, ex, ey, ez, r, scale);
    if (ef > effR * 0.12) break;
    const en = wormholeNormal(shape, ex, ey, ez, r, scale);
    const push = effR * 0.12 - ef + effR * 0.03;
    ex += en[0] * push;
    ey += en[1] * push;
    ez += en[2] * push;
  }
  return { exitX: ex, exitY: ey, exitZ: ez, nx: sn[0], ny: sn[1], nz: sn[2] };
}
