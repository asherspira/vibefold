import assert from 'node:assert/strict';
import test from 'node:test';
import {
  glueDirectionThroughThroat,
  throatCrossing,
  throatTransitionQuaternion,
  wormholeNormal,
  wormholeSdf,
  WORMHOLE_SHAPES,
} from './wormholeSdf.ts';

const radius = 1;

test('the sphere distance is |p| - r, and the bend denominator matches the old field', () => {
  assert.ok(Math.abs(wormholeSdf(0, 0, 0, 0, radius) + 1) < 1e-9);
  assert.ok(Math.abs(wormholeSdf(0, 0, 0, 3, radius) - 2) < 1e-9);
  const gap = wormholeSdf(0, 0, 0, 3, radius);
  const gravity = 0.5;
  const denom = gap + radius * gravity;
  const old = 3 - radius * (1 - gravity);
  assert.ok(Math.abs(denom - old) < 1e-9);
  const normal = wormholeNormal(0, 0, 0, 3, radius);
  assert.ok(Math.abs(normal[2] - 1) < 1e-3);
  assert.ok(Math.abs(normal[0]) < 1e-3);
});

test('every throat is centrally symmetric, so the antipode stays on the surface', () => {
  let seed = 0x12345678;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (const shape of WORMHOLE_SHAPES) {
    for (let i = 0; i < 40; i++) {
      const x = (rand() * 2 - 1) * 3;
      const y = (rand() * 2 - 1) * 3;
      const z = (rand() * 2 - 1) * 3;
      const a = wormholeSdf(shape.id, x, y, z, radius);
      const b = wormholeSdf(shape.id, -x, -y, -z, radius);
      assert.ok(Math.abs(a - b) < 1e-6, `${shape.label} f(${x},${y},${z})`);
    }
  }
});

test('known samples sit inside, outside, or on each surface', () => {
  assert.ok(wormholeSdf(1, 0, 0, 0, radius) < 0);
  assert.ok(Math.abs(wormholeSdf(1, 1.2, 0, 0, radius)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(1, 1.5, 0, 0, radius) - 0.3) < 1e-6);

  assert.ok(wormholeSdf(2, 0, 0, 0, radius) < 0);
  assert.ok(Math.abs(wormholeSdf(2, 0.85, 0, 0, radius)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(2, 0, 1.45, 0, radius)) < 1e-6);

  assert.ok(wormholeSdf(3, 0, 0, 0, radius) > 0);
  assert.ok(Math.abs(wormholeSdf(3, 1.36, 0, 0, radius)) < 1e-6);
  assert.ok(wormholeSdf(3, 1, 0, 0, radius) < 0);

  assert.ok(wormholeSdf(4, 0, 0, 0, radius) < 0);
  assert.ok(Math.abs(wormholeSdf(4, 1.55, 0, 0, radius)) < 1e-6);

  assert.ok(wormholeSdf(5, 0, 0, 0, radius) < 0);
  assert.ok(Math.abs(wormholeSdf(5, 0, 1.5, 0, radius)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(5, 1, 0, 0, radius)) < 1e-6);
});

test('the distances are Lipschitz, so a step shorter than f cannot cross the surface', () => {
  let seed = 0xabcdef;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (const shape of WORMHOLE_SHAPES) {
    for (let i = 0; i < 30; i++) {
      const ax = (rand() * 2 - 1) * 4;
      const ay = (rand() * 2 - 1) * 4;
      const az = (rand() * 2 - 1) * 4;
      const bx = ax + (rand() * 2 - 1) * 0.4;
      const by = ay + (rand() * 2 - 1) * 0.4;
      const bz = az + (rand() * 2 - 1) * 0.4;
      const df = Math.abs(wormholeSdf(shape.id, ax, ay, az, radius) - wormholeSdf(shape.id, bx, by, bz, radius));
      const dist = Math.hypot(ax - bx, ay - by, az - bz);
      assert.ok(df <= dist * 1.05 + 1e-6, `${shape.label} lipschitz ${df} > ${dist}`);
    }
  }
});

test('every throat is symmetric across the XY plane, f(x, y, -z) = f(x, y, z)', () => {
  let seed = 0x51a7c3d1;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (const shape of WORMHOLE_SHAPES) {
    for (let i = 0; i < 40; i++) {
      const x = (rand() * 2 - 1) * 3;
      const y = (rand() * 2 - 1) * 3;
      const z = (rand() * 2 - 1) * 3;
      const a = wormholeSdf(shape.id, x, y, z, radius);
      const b = wormholeSdf(shape.id, x, y, -z, radius);
      assert.ok(Math.abs(a - b) < 1e-6, `${shape.label} R_z symmetry at (${x},${y},${z})`);
    }
  }
});

test('crossing the sphere from the default Z axis exits just outside the opposite Z side', () => {
  const exit = throatCrossing(0, 0, 3, 0, radius);
  assert.ok(exit.exitZ < -1);
  assert.ok(Math.abs(exit.exitX) < 1e-3);
  assert.ok(Math.abs(exit.exitY) < 1e-3);
  assert.ok(wormholeSdf(0, exit.exitX, exit.exitY, exit.exitZ, radius) > 0.1);
  assert.ok(exit.nz > 0.9);
});

test('crossing a torus tube from above exits outside the upper rim without flipping Y', () => {
  const exit = throatCrossing(0, 3, 0, 3, radius);
  assert.ok(exit.exitY > 1.36);
  assert.ok(wormholeSdf(3, exit.exitX, exit.exitY, exit.exitZ, radius) > 0.1);
});

test('throat ray gluing keeps sky (+Y) upright and matches throatTransitionQuaternion on all normals', () => {
  const rotateByQuat = (
    q: readonly [number, number, number, number],
    v: readonly [number, number, number]
  ): [number, number, number] => {
    const [qx, qy, qz, qw] = q;
    const [vx, vy, vz] = v;
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return [
      vx + qw * tx + (qy * tz - qz * ty),
      vy + qw * ty + (qz * tx - qx * tz),
      vz + qw * tz + (qx * ty - qy * tx),
    ];
  };

  // Head-on approach along +Z (normal = (0, 0, -1)): T must be the exact identity
  const upRay = glueDirectionThroughThroat(0, 0.5, Math.SQRT1_2, 0, 0, -1);
  assert.ok(upRay[1] > 0, 'Upper ray must stay pointing UP (+Y) after crossing');
  assert.ok(upRay[2] > 0, 'Forward ray must continue along +Z');

  const rightRay = glueDirectionThroughThroat(-0.5, 0, Math.SQRT1_2, 0, 0, -1);
  assert.ok(rightRay[0] < 0, 'Right ray must not horizontally mirror across the threshold');

  // Horizontal equator normals must always preserve world +Y (0, 1, 0)
  for (let i = 0; i < 16; i++) {
    const phi = (i / 16) * Math.PI * 2;
    const nx = Math.cos(phi);
    const nz = Math.sin(phi);
    const gluedUp = glueDirectionThroughThroat(0, 1, 0, nx, 0, nz);
    assert.ok(Math.abs(gluedUp[0]) < 1e-9 && Math.abs(gluedUp[1] - 1) < 1e-9 && Math.abs(gluedUp[2]) < 1e-9);
  }

  // Arbitrary 3D normals: quaternion action == glueDirectionThroughThroat, det = +1, and exact involution
  const normals: [number, number, number][] = [
    [0, 0, -1],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1 / Math.SQRT2, 0, -1 / Math.SQRT2],
    [1 / Math.sqrt(3), 1 / Math.sqrt(3), -1 / Math.sqrt(3)],
    [-0.3, 0.6, 0.7416198487095663],
  ];
  const dirs: [number, number, number][] = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [-0.25, 0.4, 0.8817596044274199],
  ];
  for (const [nx, ny, nz] of normals) {
    const q = throatTransitionQuaternion(nx, ny, nz);
    assert.ok(Math.abs(Math.hypot(...q) - 1) < 1e-9);
    for (const [dx, dy, dz] of dirs) {
      const g = glueDirectionThroughThroat(dx, dy, dz, nx, ny, nz);
      const rq = rotateByQuat(q, [dx, dy, dz]);
      assert.ok(
        Math.hypot(g[0] - rq[0], g[1] - rq[1], g[2] - rq[2]) < 1e-9,
        `Quaternion mismatch for n=(${nx},${ny},${nz}), d=(${dx},${dy},${dz})`
      );
      // Reversing the exit ray and crossing back through exitNormal=(nx, ny, -nz) returns -d
      const back = glueDirectionThroughThroat(-g[0], -g[1], -g[2], nx, ny, -nz);
      assert.ok(
        Math.hypot(back[0] + dx, back[1] + dy, back[2] + dz) < 1e-9,
        `Involution failed for n=(${nx},${ny},${nz})`
      );
    }
  }
});

test('non-uniform width/height/length scaling stretches all 6 shapes while remaining 1-Lipschitz and R_z-symmetric', () => {
  const scale: [number, number, number] = [2.0, 0.5, 1.5];

  // Exact zero level-set checks on stretched axes
  // Sphere -> Ellipsoid with semi-axes (2, 0.5, 1.5)
  assert.ok(Math.abs(wormholeSdf(0, 2.0, 0, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(0, 0, 0.5, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(0, 0, 0, 1.5, radius, scale)) < 1e-6);
  // Cuboid -> exact half-extents (2.4, 0.36, 1.35)
  assert.ok(Math.abs(wormholeSdf(1, 2.4, 0, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(1, 0, 0.36, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(1, 0, 0, 1.35, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(1, 2.9, 0, 0, radius, scale) - 0.5) < 1e-6);
  // Cylinder -> x-radius 1.7, y-height 0.725, z-radius 1.275
  assert.ok(Math.abs(wormholeSdf(2, 1.7, 0, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(2, 0, 0.725, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(2, 0, 0, 1.275, radius, scale)) < 1e-6);
  // Torus -> outer rim at x = 2 * 1.36 = 2.72, y = 0.5 * 1.36 = 0.68
  assert.ok(Math.abs(wormholeSdf(3, 2.72, 0, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(3, 0, 0.68, 0, radius, scale)) < 1e-6);
  assert.ok(wormholeSdf(3, 0, 0, 0, radius, scale) > 0);
  // Octahedron -> vertices at (±3.1, 0, 0), (0, ±0.775, 0), (0, 0, ±2.325)
  assert.ok(Math.abs(wormholeSdf(4, 3.1, 0, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(4, 0, 0.775, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(4, 0, 0, 2.325, radius, scale)) < 1e-6);
  // Double cone -> apex at y = 0.75, equator x = 2.0, z = 1.5
  assert.ok(Math.abs(wormholeSdf(5, 0, 0.75, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(5, 2.0, 0, 0, radius, scale)) < 1e-6);
  assert.ok(Math.abs(wormholeSdf(5, 0, 0, 1.5, radius, scale)) < 1e-6);

  // 1-Lipschitz and R_z symmetry checks for all 6 stretched shapes
  let seed = 0x7711aa33;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (const shape of WORMHOLE_SHAPES) {
    for (let i = 0; i < 30; i++) {
      const ax = (rand() * 2 - 1) * 4;
      const ay = (rand() * 2 - 1) * 4;
      const az = (rand() * 2 - 1) * 4;
      const fa = wormholeSdf(shape.id, ax, ay, az, radius, scale);
      const faz = wormholeSdf(shape.id, ax, ay, -az, radius, scale);
      assert.ok(Math.abs(fa - faz) < 1e-6, `${shape.label} scaled R_z symmetry`);

      const bx = ax + (rand() * 2 - 1) * 0.4;
      const by = ay + (rand() * 2 - 1) * 0.4;
      const bz = az + (rand() * 2 - 1) * 0.4;
      const df = Math.abs(fa - wormholeSdf(shape.id, bx, by, bz, radius, scale));
      const dist = Math.hypot(ax - bx, ay - by, az - bz);
      assert.ok(df <= dist * 1.05 + 1e-6, `${shape.label} scaled Lipschitz ${df} > ${dist}`);
    }
    const crossed = throatCrossing(0, 0, 4, shape.id, radius, scale);
    assert.ok(wormholeSdf(shape.id, crossed.exitX, crossed.exitY, crossed.exitZ, radius, scale) > 0.05);
  }
});
