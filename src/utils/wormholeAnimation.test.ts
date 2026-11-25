import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildMultiverseTimeline,
  computeRenderDimensions,
  evaluateEasing,
  findWormholeRoute,
  generateMultiverseTourKeyframes,
  interpolateKeyframe,
  sampleMultiverseTimeline,
  sampleTimelinePathInUniverse,
  slerpQuaternion,
  type CameraKeyframe,
  type InterpolationMode,
  type WormholeGeometry,
} from './wormholeAnimation.ts';
import {
  createPresetMultiverseGraph,
  evaluateMouthSdfWorld,
  quaternionFromEulerDeg,
  quatRotateInvVec3,
  quatRotateVec3,
  type MultiverseGraph,
} from './wormholeMultiverse.ts';
import {
  glueDirectionThroughThroat,
  wormholeNormal,
  wormholeSdf,
  WORMHOLE_SHAPES,
} from './wormholeSdf.ts';

const MODES: InterpolationMode[] = ['lerp', 'smoothstep', 'exp_decay'];

test('evaluateEasing honors exact endpoints and monotonicity on [0, 1]', () => {
  for (const mode of MODES) {
    assert.equal(evaluateEasing(0, mode), 0, `${mode} at t=0`);
    assert.equal(evaluateEasing(1, mode), 1, `${mode} at t=1`);
    assert.equal(evaluateEasing(-0.5, mode), 0, `${mode} clamps below 0`);
    assert.equal(evaluateEasing(1.5, mode), 1, `${mode} clamps above 1`);

    let prev = -1;
    for (let i = 0; i <= 50; i++) {
      const t = i / 50;
      const val = evaluateEasing(t, mode);
      assert.ok(val >= prev - 1e-12, `${mode} must be non-decreasing at t=${t}`);
      prev = val;
    }
  }
});

test('evaluateEasing distinguishes lerp, smoothstep, and exp_decay curves', () => {
  assert.ok(Math.abs(evaluateEasing(0.5, 'smoothstep') - 0.5) < 1e-12);
  assert.ok(evaluateEasing(0.2, 'smoothstep') < evaluateEasing(0.2, 'lerp'));
  assert.ok(evaluateEasing(0.8, 'smoothstep') > evaluateEasing(0.8, 'lerp'));

  assert.ok(evaluateEasing(0.2, 'exp_decay') > evaluateEasing(0.2, 'lerp'));
  assert.ok(evaluateEasing(0.5, 'exp_decay') > 0.8);
});

test('slerpQuaternion maintains unit length and shortest-arc interpolation', () => {
  const identity: [number, number, number, number] = [0, 0, 0, 1];
  const negatedIdentity: [number, number, number, number] = [0, 0, 0, -1];
  const midSame = slerpQuaternion(identity, negatedIdentity, 0.5);
  assert.ok(Math.abs(Math.hypot(...midSame) - 1) < 1e-12);
  assert.ok(Math.abs(midSame[3]) > 0.9999, 'Antipodal equivalent quaternions should stay stationary');

  const halfAngle = Math.PI / 4;
  const rotY90: [number, number, number, number] = [0, Math.sin(halfAngle), 0, Math.cos(halfAngle)];
  const rotY45 = slerpQuaternion(identity, rotY90, 0.5);
  assert.ok(Math.abs(Math.hypot(...rotY45) - 1) < 1e-12);
  assert.ok(Math.abs(rotY45[1] - Math.sin(Math.PI / 8)) < 1e-9);
  assert.ok(Math.abs(rotY45[3] - Math.cos(Math.PI / 8)) < 1e-9);
});

test('interpolateKeyframe hits exact start and end poses', () => {
  const start: CameraKeyframe = {
    position: [0, 0, -0.15],
    quaternion: [0, 0, 0, 1],
    target: [0, 0, 3],
    room: 0,
  };
  const end: CameraKeyframe = {
    position: [0, 0, 4.15],
    quaternion: [0, 0, 0, 1],
    target: [0, 0, 7],
    room: 1,
  };
  const wormhole: WormholeGeometry = { center: [0, 0, 3], radius: 1, shape: 0 };

  for (const mode of MODES) {
    const at0 = interpolateKeyframe(start, end, 0, mode, wormhole);
    const at1 = interpolateKeyframe(start, end, 1, mode, wormhole);

    assert.deepEqual(at0.position, start.position);
    assert.deepEqual(at0.target, start.target);
    assert.equal(at0.room, start.room);

    assert.deepEqual(at1.position, end.position);
    assert.deepEqual(at1.target, end.target);
    assert.equal(at1.room, end.room);
  }
});

test('cross-universe interpolation never places the camera inside the throat for any shape or easing mode', () => {
  const center: [number, number, number] = [0, 0, 3];
  const radius = 1;
  const shaderHitEps = Math.max(radius * 0.02, 1e-4);

  // Typical workflow: Start in front of wormhole in Universe A, Enter throat into Universe B
  const start: CameraKeyframe = {
    position: [0, 0, -0.15],
    quaternion: [0, 1, 0, 0],
    target: [0, 0, 3],
    room: 0,
  };
  const end: CameraKeyframe = {
    position: [0, 0, 5.5],
    quaternion: [0, 1, 0, 0],
    target: [0, 0, 8.5],
    room: 1,
  };

  for (const { id: shape, label } of WORMHOLE_SHAPES) {
    const wormhole: WormholeGeometry = { center, radius, shape };
    for (const mode of MODES) {
      let sawRoom0 = false;
      let sawRoom1 = false;
      for (let i = 0; i <= 100; i++) {
        const t = i / 100;
        const pose = interpolateKeyframe(start, end, t, mode, wormhole);
        if (pose.room === 0) sawRoom0 = true;
        if (pose.room === 1) sawRoom1 = true;
        const gap = wormholeSdf(
          shape,
          pose.position[0] - center[0],
          pose.position[1] - center[1],
          pose.position[2] - center[2],
          radius
        );
        assert.ok(
          gap > shaderHitEps,
          `${label} (${mode}) at t=${t} dipped inside throat: gap=${gap} <= hitEps=${shaderHitEps}`
        );
      }
      assert.ok(sawRoom0 && sawRoom1, `${label} (${mode}) must traverse both universes`);
    }
  }
});

test('same-universe animation across opposite sides of the wormhole smoothly clears the throat', () => {
  const center: [number, number, number] = [0, 0, 3];
  const radius = 1;
  const shaderHitEps = Math.max(radius * 0.02, 1e-4);

  const start: CameraKeyframe = {
    position: [0, 0, 0],
    quaternion: [0, 1, 0, 0],
    target: [0, 0, 3],
    room: 0,
  };
  const end: CameraKeyframe = {
    position: [0, 0, 6],
    quaternion: [0, 0, 0, 1],
    target: [0, 0, 3],
    room: 0,
  };

  for (const { id: shape, label } of WORMHOLE_SHAPES) {
    const wormhole: WormholeGeometry = { center, radius, shape };
    for (let i = 0; i <= 60; i++) {
      const t = i / 60;
      const pose = interpolateKeyframe(start, end, t, 'smoothstep', wormhole);
      assert.equal(pose.room, 0);
      const gap = wormholeSdf(
        shape,
        pose.position[0] - center[0],
        pose.position[1] - center[1],
        pose.position[2] - center[2],
        radius
      );
      assert.ok(
        gap > shaderHitEps,
        `Same-universe ${label} at t=${t} dipped inside throat: gap=${gap}`
      );
    }
  }
});

test('computeRenderDimensions always produces even dimensions within maxEdge', () => {
  const d1 = computeRenderDimensions(1441, 811, 1920);
  assert.equal(d1.width % 2, 0);
  assert.equal(d1.height % 2, 0);
  assert.ok(d1.width <= 1920 && d1.height <= 1920);

  const d2 = computeRenderDimensions(3840, 2160, 1920);
  assert.equal(d2.width, 1920);
  assert.equal(d2.height, 1080);

  const d3 = computeRenderDimensions(1, 1, 1920);
  assert.equal(d3.width, 2);
  assert.equal(d3.height, 2);
});

test('cross-universe animation maintains continuous vertical and horizontal ray orientation across the threshold', () => {
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

  const center: [number, number, number] = [0, 0, 3];
  const radius = 1;
  const wormhole: WormholeGeometry = { center, radius, shape: 0 };

  // Side approach from +X looking -X (yaw +90 deg: q = [0, sin(45), 0, cos(45)])
  // emerging in Universe B at +X looking +X (yaw -90 deg: q = [0, -sin(45), 0, cos(45)])
  const s = Math.SQRT1_2;
  const start: CameraKeyframe = {
    position: [3.5, 0, 3],
    quaternion: [0, s, 0, s],
    target: [0, 0, 3],
    room: 0,
  };
  const end: CameraKeyframe = {
    position: [3.5, 0, 3],
    quaternion: [0, -s, 0, s],
    target: [6.5, 0, 3],
    room: 1,
  };

  // Find the crossing frame pair
  let before = interpolateKeyframe(start, end, 0, 'lerp', wormhole);
  let after = before;
  for (let i = 1; i <= 200; i++) {
    const curr = interpolateKeyframe(start, end, i / 200, 'lerp', wormhole);
    if (before.room === 0 && curr.room === 1) {
      after = curr;
      break;
    }
    before = curr;
  }
  assert.equal(before.room, 0);
  assert.equal(after.room, 1);

  const entryNormal = wormholeNormal(
    0,
    before.position[0] - center[0],
    before.position[1] - center[1],
    before.position[2] - center[2],
    radius
  );

  // Check center, upper, and right camera-space rays across the transition
  const camRays: [number, number, number][] = [
    [0, 0, -1],
    [0, 0.35, -0.9367496997597597],
    [0.35, 0, -0.9367496997597597],
  ];
  for (const camRay of camRays) {
    const worldBefore = rotateByQuat(before.quaternion, camRay);
    const gluedIntoB = glueDirectionThroughThroat(
      worldBefore[0],
      worldBefore[1],
      worldBefore[2],
      entryNormal[0],
      entryNormal[1],
      entryNormal[2]
    );
    const worldAfter = rotateByQuat(after.quaternion, camRay);
    const diff = Math.hypot(
      gluedIntoB[0] - worldAfter[0],
      gluedIntoB[1] - worldAfter[1],
      gluedIntoB[2] - worldAfter[2]
    );
    assert.ok(diff < 0.02, `Ray discontinuity across threshold: diff=${diff}`);
  }
});

test('cross-universe animation respects non-uniform width/height/length shape scaling for all shapes', () => {
  const center: [number, number, number] = [0, 0, 3];
  const radius = 1;
  const scale: [number, number, number] = [2.2, 0.45, 1.6];
  const effR = radius * Math.min(...scale);
  const shaderHitEps = Math.max(effR * 0.02, 1e-4);

  const start: CameraKeyframe = {
    position: [0, 0, -2.5],
    quaternion: [0, 1, 0, 0],
    target: [0, 0, 3],
    room: 0,
  };
  const end: CameraKeyframe = {
    position: [0, 0, 8.5],
    quaternion: [0, 1, 0, 0],
    target: [0, 0, 12],
    room: 1,
  };

  for (const { id: shape, label } of WORMHOLE_SHAPES) {
    const wormhole: WormholeGeometry = { center, radius, shape, scale };
    for (let i = 0; i <= 80; i++) {
      const t = i / 80;
      const pose = interpolateKeyframe(start, end, t, 'smoothstep', wormhole);
      const gap = wormholeSdf(
        shape,
        pose.position[0] - center[0],
        pose.position[1] - center[1],
        pose.position[2] - center[2],
        radius,
        scale
      );
      assert.ok(
        gap > shaderHitEps,
        `Scaled ${label} at t=${t} dipped inside throat: gap=${gap} <= hitEps=${shaderHitEps}`
      );
    }
  }
});

function makeTestParams() {
  return {
    position: [0, 0, 3] as [number, number, number],
    radius: 1.0,
    shape: 0 as const,
    scale: [1, 1, 1] as [number, number, number],
    gravityRatio: 0.5,
    universeA: {
      mode: 'grid' as const,
      grid: {
        verticalColor: '#ff0000',
        horizontalColor: '#0000ff',
        cols: 18,
        rows: 9,
        rotationY: 0,
        rotationX: 0,
      },
      isHdr: false,
      exposureEv: 0,
      textureLayer: -1,
    },
    universeB: {
      mode: 'grid' as const,
      grid: {
        verticalColor: '#00ffff',
        horizontalColor: '#ff00ff',
        cols: 18,
        rows: 9,
        rotationY: 0,
        rotationX: 0,
      },
      isHdr: false,
      exposureEv: 0,
      textureLayer: -1,
    },
  };
}

test('Stage 4 findWormholeRoute discovers shortest multi-hop paths and detects disconnected universes', () => {
  const trinity = createPresetMultiverseGraph('trinity-loop', makeTestParams());
  // Add an isolated 4th universe D and a 5th universe E chained off C -> E
  const graph: MultiverseGraph = {
    universes: [
      ...trinity.universes,
      { ...trinity.universes[0], id: 'D', name: 'Isolated D' },
      { ...trinity.universes[1], id: 'E', name: 'Universe E' },
    ],
    wormholes: [
      ...trinity.wormholes,
      {
        id: 'wh-ce',
        name: 'C to E',
        shape: 2,
        radius: 0.85,
        scale: [1, 1, 1],
        gravityRatio: 0.5,
        mouthA: { universeId: 'C', position: [0, 3.5, 3], quaternion: [0, 0, 0, 1] },
        mouthB: { universeId: 'E', position: [0, 0, 3], quaternion: [0, 0, 0, 1] },
      },
    ],
  };

  // A (0) -> E (4) must take 2 hops: A -> C -> E
  const routeAE = findWormholeRoute(graph, 0, 4, [0, 0, 0], [0, 0, 0]);
  assert.equal(routeAE.reachable, true);
  assert.equal(routeAE.hops.length, 2);
  assert.equal(routeAE.hops[0].toUniverseIdx, 2);
  assert.equal(routeAE.hops[1].toUniverseIdx, 4);

  // A (0) -> D (3) is isolated
  const routeAD = findWormholeRoute(graph, 0, 3, [0, 0, 0], [0, 0, 0]);
  assert.equal(routeAD.reachable, false);
  assert.equal(routeAD.hops.length, 0);
});

test('Stage 4 multiverse timeline maintains clearance from all mouths and continuous orientation across oriented portals', () => {
  const trinity = createPresetMultiverseGraph('trinity-loop', makeTestParams());

  // Rotate the A<->C torus mouths to non-trivial 3D orientations
  const qEntry = quaternionFromEulerDeg(25, -12, 8);
  const qExit = quaternionFromEulerDeg(-35, 18, -10);
  trinity.wormholes[1].mouthA.quaternion = qEntry;
  trinity.wormholes[1].mouthB.quaternion = qExit;

  const tour = generateMultiverseTourKeyframes(trinity, 0);
  assert.ok(tour.length >= 3, `expected >= 3 tour keyframes for trinity-loop, got ${tour.length}`);

  const timeline = buildMultiverseTimeline(tour, trinity, 'smoothstep');
  assert.equal(timeline.disconnectedSegmentIndices.length, 0);
  assert.ok(timeline.totalHops >= 2, `expected >= 2 portal crossings, got ${timeline.totalHops}`);

  const visitedRooms = new Set<number>();
  for (let i = 0; i <= 160; i++) {
    const t = i / 160;
    const pose = sampleMultiverseTimeline(timeline, t);
    visitedRooms.add(pose.room);
    const uNode = trinity.universes[pose.room];
    assert.ok(uNode, `invalid universe index ${pose.room}`);

    // Camera must stay strictly outside EVERY wormhole mouth in pose.room
    for (const wh of trinity.wormholes) {
      const effR = wh.radius * Math.min(...wh.scale);
      const hitEps = Math.max(effR * 0.02, 1e-4);
      if (wh.mouthA.universeId === uNode.id) {
        const gap = evaluateMouthSdfWorld(wh.mouthA, wh.shape, wh.radius, wh.scale, pose.position);
        assert.ok(gap > hitEps, `Tour clipped ${wh.id}:A in ${uNode.id} at t=${t}: gap=${gap}`);
      }
      if (wh.mouthB.universeId === uNode.id) {
        const gap = evaluateMouthSdfWorld(wh.mouthB, wh.shape, wh.radius, wh.scale, pose.position);
        assert.ok(gap > hitEps, `Tour clipped ${wh.id}:B in ${uNode.id} at t=${t}: gap=${gap}`);
      }
    }
  }

  // Must visit all 3 universes A, B, C
  assert.equal(visitedRooms.size, 3);

  // Path preview polylines exist in every visited universe
  for (let u = 0; u < 3; u++) {
    const polys = sampleTimelinePathInUniverse(timeline, u, 16);
    assert.ok(polys.length >= 1, `expected 3D trajectory preview segments in universe ${u}`);
  }

  // Direct oriented A -> C crossing verifies exact world-space ray continuity
  const kfA: CameraKeyframe = {
    universeId: 'A',
    room: 0,
    position: [2.35, 0, -0.5],
    quaternion: [0, 1, 0, 0],
    target: [2.35, 0, 3],
  };
  const kfC: CameraKeyframe = {
    universeId: 'C',
    room: 2,
    position: [-2.35, 0, 6.2],
    quaternion: [0, 1, 0, 0],
    target: [-2.35, 0, 9.2],
  };
  const directTl = buildMultiverseTimeline([kfA, kfC], trinity, 'lerp');
  assert.equal(directTl.totalHops, 1);

  let prev = sampleMultiverseTimeline(directTl, 0);
  let crossed: { before: CameraKeyframe; after: CameraKeyframe } | null = null;
  for (let i = 1; i <= 240; i++) {
    const curr = sampleMultiverseTimeline(directTl, i / 240);
    if (prev.room === 0 && curr.room === 2) {
      crossed = { before: prev, after: curr };
      break;
    }
    prev = curr;
  }
  assert.ok(crossed, 'expected crossing from A to C');

  const hop = directTl.segments[0].hops[0];
  const localBefore: [number, number, number] = quatRotateInvVec3(hop.entryMouth.quaternion, [
    crossed.before.position[0] - hop.entryMouth.position[0],
    crossed.before.position[1] - hop.entryMouth.position[1],
    crossed.before.position[2] - hop.entryMouth.position[2],
  ]);
  const nLocal = wormholeNormal(
    hop.wormhole.shape,
    localBefore[0],
    localBefore[1],
    localBefore[2],
    hop.wormhole.radius,
    hop.wormhole.scale
  );

  const camRay: [number, number, number] = [0.2, 0.15, -0.9682458365518543];
  const worldBefore = quatRotateVec3(crossed.before.quaternion, camRay);
  const localRayBefore = quatRotateInvVec3(hop.entryMouth.quaternion, worldBefore);
  const gluedLocal = glueDirectionThroughThroat(
    localRayBefore[0],
    localRayBefore[1],
    localRayBefore[2],
    nLocal[0],
    nLocal[1],
    nLocal[2]
  );
  const expectedWorldAfter = quatRotateVec3(hop.exitMouth.quaternion, gluedLocal);
  const actualWorldAfter = quatRotateVec3(crossed.after.quaternion, camRay);
  const rayErr = Math.hypot(
    expectedWorldAfter[0] - actualWorldAfter[0],
    expectedWorldAfter[1] - actualWorldAfter[1],
    expectedWorldAfter[2] - actualWorldAfter[2]
  );
  assert.ok(rayErr < 0.03, `Oriented portal ray discontinuity: err=${rayErr}`);
});

