import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createDefaultUniverseNode,
  createDefaultWormholeEdge,
  createPresetMultiverseGraph,
  createTwoUniverseGraph,
  crossMouthWorld,
  detectMouthOverlaps,
  eulerDegFromQuaternion,
  evaluateMouthSdfWorld,
  findNearestMouthInUniverse,
  GRID_OVERRIDE_LAT_OFFSET,
  GRID_OVERRIDE_TEX_WIDTH,
  MOUTH_TEXELS_PER_ROW,
  packMultiverseToFloatBuffers,
  parseHexColorSrgb,
  parseMultiverseGraphJson,
  quaternionFromEulerDeg,
  quatRotateInvVec3,
  quatRotateVec3,
  serializeMultiverseGraph,
  UNIVERSE_TEXELS_PER_ROW,
  type MultiverseGraph,
  type TwoUniverseSceneParams,
} from './wormholeMultiverse.ts';

function makeDefaultParams(): TwoUniverseSceneParams {
  return {
    position: [0, 0, 3],
    radius: 1.0,
    shape: 0,
    scale: [1, 1, 1],
    gravityRatio: 0.5,
    universeA: {
      mode: 'grid',
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
      mode: 'grid',
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

test('parseHexColorSrgb handles 6-digit, 3-digit, and invalid hex colors', () => {
  assert.deepEqual(parseHexColorSrgb('#ff0000'), [1, 0, 0]);
  assert.deepEqual(parseHexColorSrgb('#0f0'), [0, 1, 0]);
  assert.deepEqual(parseHexColorSrgb('0000ff'), [0, 0, 1]);
  assert.deepEqual(parseHexColorSrgb('not-a-color', [0.25, 0.5, 0.75]), [0.25, 0.5, 0.75]);
});

test('quatRotateVec3 and quatRotateInvVec3 are exact inverses and preserve length', () => {
  const q = quaternionFromEulerDeg(37, -22, 14);
  const v: [number, number, number] = [1.2, -0.7, 2.4];
  const rotated = quatRotateVec3(q, v);
  const back = quatRotateInvVec3(q, rotated);

  assert.ok(Math.abs(Math.hypot(...rotated) - Math.hypot(...v)) < 1e-6);
  assert.ok(Math.abs(back[0] - v[0]) < 1e-6);
  assert.ok(Math.abs(back[1] - v[1]) < 1e-6);
  assert.ok(Math.abs(back[2] - v[2]) < 1e-6);
});

test('createTwoUniverseGraph builds matched bridged mouths and packs accurately', () => {
  const graph = createTwoUniverseGraph({
    position: [1.25, -0.5, 3.0],
    radius: 1.4,
    shape: 3, // Torus
    scale: [1.5, 0.8, 1.1],
    gravityRatio: 0.65,
    universeA: {
      mode: 'grid',
      grid: {
        verticalColor: '#ff0000',
        horizontalColor: '#0000ff',
        cols: 18,
        rows: 9,
        rotationY: 45,
        rotationX: -15,
      },
      isHdr: false,
      exposureEv: 0,
      textureLayer: -1,
    },
    universeB: {
      mode: 'texture',
      grid: {
        verticalColor: '#00ffff',
        horizontalColor: '#ff00ff',
        cols: 24,
        rows: 12,
        rotationY: -90,
        rotationX: 30,
      },
      isHdr: true,
      exposureEv: 1.5,
      textureLayer: 0,
    },
  });

  const packed = packMultiverseToFloatBuffers(graph);

  assert.equal(packed.universeCount, 2);
  assert.equal(packed.universeWidth, UNIVERSE_TEXELS_PER_ROW);
  assert.equal(packed.universeHeight, 2);
  assert.equal(packed.mouthCount, 2);
  assert.equal(packed.mouthWidth, MOUTH_TEXELS_PER_ROW);
  assert.equal(packed.mouthHeight, 2);

  // Universe 0 (A)
  const u0 = 0;
  assert.equal(packed.universeData[u0 + 0], 0); // grid mode
  assert.equal(packed.universeData[u0 + 1], -1); // no texture layer
  assert.equal(packed.universeData[u0 + 2], 0); // not HDR
  assert.ok(Math.abs(packed.universeData[u0 + 3] - 1) < 1e-6); // 2^0
  assert.equal(packed.universeData[u0 + 4], 18);
  assert.equal(packed.universeData[u0 + 5], 9);
  assert.ok(Math.abs(packed.universeData[u0 + 6] - Math.PI / 4) < 1e-5);
  assert.ok(Math.abs(packed.universeData[u0 + 7] + Math.PI / 12) < 1e-5);
  assert.deepEqual(
    [packed.universeData[u0 + 8], packed.universeData[u0 + 9], packed.universeData[u0 + 10]],
    [1, 0, 0]
  );
  assert.equal(packed.universeData[u0 + 11], 0); // mouthOffset = 0
  assert.deepEqual(
    [packed.universeData[u0 + 12], packed.universeData[u0 + 13], packed.universeData[u0 + 14]],
    [0, 0, 1]
  );
  assert.equal(packed.universeData[u0 + 15], 1); // mouthCount = 1

  // Universe 1 (B)
  const u1 = UNIVERSE_TEXELS_PER_ROW * 4;
  assert.equal(packed.universeData[u1 + 0], 1); // texture mode
  assert.equal(packed.universeData[u1 + 1], 0); // layer 0
  assert.equal(packed.universeData[u1 + 2], 1); // HDR enabled
  assert.ok(Math.abs(packed.universeData[u1 + 3] - Math.pow(2, 1.5)) < 1e-5);
  assert.equal(packed.universeData[u1 + 11], 1); // mouthOffset = 1
  assert.equal(packed.universeData[u1 + 15], 1); // mouthCount = 1

  // Both bridged mouths share exact shape, radius, scale, gravityRatio and link symmetrically
  const m0 = 0;
  const m1 = MOUTH_TEXELS_PER_ROW * 4;
  assert.ok(Math.abs(packed.mouthData[m0 + 3] - 1.4) < 1e-5);
  assert.ok(Math.abs(packed.mouthData[m1 + 3] - 1.4) < 1e-5);
  assert.equal(packed.mouthData[m0 + 11], 3); // shape Torus
  assert.equal(packed.mouthData[m1 + 11], 3); // shape Torus
  assert.ok(Math.abs(packed.mouthData[m0 + 12] - 0.65) < 1e-5);
  assert.ok(Math.abs(packed.mouthData[m1 + 12] - 0.65) < 1e-5);

  // Bidirectional portal links: mouth 0 (in U0) -> mouth 1 (in U1), and vice versa
  assert.equal(packed.mouthData[m0 + 13], 0); // universeIdx
  assert.equal(packed.mouthData[m0 + 14], 1); // destUniverseIdx
  assert.equal(packed.mouthData[m0 + 15], 1); // destMouthIdx
  assert.equal(packed.mouthData[m1 + 13], 1);
  assert.equal(packed.mouthData[m1 + 14], 0);
  assert.equal(packed.mouthData[m1 + 15], 0);
});

test('packMultiverseToFloatBuffers handles arbitrary N universes, self-loops, isolated nodes, and buffer reuse', () => {
  const graph: MultiverseGraph = {
    universes: [
      {
        id: 'u-alpha',
        name: 'Alpha',
        skyMode: 'grid',
        gridVerticalColor: '#ff0000',
        gridHorizontalColor: '#00ff00',
        gridCols: 12,
        gridRows: 6,
        rotationY: 0,
        rotationX: 0,
        isHdr: false,
        exposureEv: 0,
        textureLayer: -1,
      },
      {
        id: 'u-beta',
        name: 'Beta',
        skyMode: 'texture',
        gridVerticalColor: '#00ffff',
        gridHorizontalColor: '#ff00ff',
        gridCols: 16,
        gridRows: 8,
        rotationY: 90,
        rotationX: 0,
        isHdr: false,
        exposureEv: 0,
        textureLayer: 0,
      },
      {
        id: 'u-gamma-empty',
        name: 'Gamma (Isolated)',
        skyMode: 'grid',
        gridVerticalColor: '#ffffff',
        gridHorizontalColor: '#888888',
        gridCols: 8,
        gridRows: 4,
        rotationY: 0,
        rotationX: 0,
        isHdr: false,
        exposureEv: 0,
        textureLayer: -1,
      },
      {
        id: 'u-delta',
        name: 'Delta',
        skyMode: 'texture',
        gridVerticalColor: '#ffff00',
        gridHorizontalColor: '#ff8800',
        gridCols: 20,
        gridRows: 10,
        rotationY: 0,
        rotationX: 0,
        isHdr: true,
        exposureEv: -1,
        textureLayer: 1,
      },
    ],
    wormholes: [
      {
        id: 'wh-ab',
        name: 'Alpha-Beta Bridge',
        shape: 1, // Cuboid
        radius: 1.0,
        scale: [1, 1, 1],
        gravityRatio: 0.5,
        mouthA: { universeId: 'u-alpha', position: [-3, 0, 0], quaternion: [0, 0, 0, 1] },
        mouthB: { universeId: 'u-beta', position: [0, 2, 0], quaternion: [0, 0.7071, 0, 0.7071] },
      },
      {
        id: 'wh-self-alpha',
        name: 'Alpha Shortcut',
        shape: 0, // Sphere
        radius: 0.8,
        scale: [1, 1, 1],
        gravityRatio: 0.3,
        mouthA: { universeId: 'u-alpha', position: [3, 0, 0], quaternion: [0, 0, 0, 1] },
        mouthB: { universeId: 'u-alpha', position: [3, 4, 0], quaternion: [0, 0, 0, 1] },
      },
      {
        id: 'wh-bd',
        name: 'Beta-Delta Spire',
        shape: 4, // Octahedron
        radius: 1.2,
        scale: [1, 2, 1],
        gravityRatio: 0.75,
        mouthA: { universeId: 'u-beta', position: [0, -3, 0], quaternion: [0, 0, 0, 1] },
        mouthB: { universeId: 'u-delta', position: [0, 0, 5], quaternion: [0, 0, 0, 1] },
      },
    ],
  };

  const packed = packMultiverseToFloatBuffers(graph);

  assert.equal(packed.universeCount, 4);
  assert.equal(packed.mouthCount, 6);

  // Per-universe contiguous mouth slices: Alpha has 3 mouths, Beta has 2, Gamma has 0, Delta has 1
  assert.equal(packed.universeMouthSlices[0].offset, 0);
  assert.equal(packed.universeMouthSlices[0].count, 3);
  assert.equal(packed.universeMouthSlices[1].offset, 3);
  assert.equal(packed.universeMouthSlices[1].count, 2);
  assert.equal(packed.universeMouthSlices[2].offset, 5);
  assert.equal(packed.universeMouthSlices[2].count, 0);
  assert.equal(packed.universeMouthSlices[3].offset, 5);
  assert.equal(packed.universeMouthSlices[3].count, 1);

  // Every mouth link must be strictly symmetric: partner(partner(m)) === m, and share identical shape
  for (let m = 0; m < packed.mouthCount; m++) {
    const meta = packed.mouthLookup[m];
    const partner = packed.mouthLookup[meta.destMouthIndex];
    assert.equal(partner.destMouthIndex, m);
    assert.equal(partner.universeIndex, meta.destUniverseIndex);
    assert.equal(partner.destUniverseIndex, meta.universeIndex);

    const baseM = m * MOUTH_TEXELS_PER_ROW * 4;
    const baseP = meta.destMouthIndex * MOUTH_TEXELS_PER_ROW * 4;
    assert.equal(packed.mouthData[baseM + 11], packed.mouthData[baseP + 11]); // same shape
    assert.equal(packed.mouthData[baseM + 3], packed.mouthData[baseP + 3]); // same radius
    assert.equal(packed.mouthData[baseM + 12], packed.mouthData[baseP + 12]); // same gravityRatio

    // Quaternion must be unit-length
    const qLen = Math.hypot(
      packed.mouthData[baseM + 4],
      packed.mouthData[baseM + 5],
      packed.mouthData[baseM + 6],
      packed.mouthData[baseM + 7]
    );
    assert.ok(Math.abs(qLen - 1) < 1e-5);
  }

  // Alpha cluster sphere must enclose all 3 of Alpha's mouths
  const alphaSlice = packed.universeMouthSlices[0];
  assert.ok(alphaSlice.clusterRadius >= 3.5);

  // Reusing pre-allocated Float32Arrays returns exact same array instances (zero GC)
  const repacked = packMultiverseToFloatBuffers(graph, {
    universeData: packed.universeData,
    mouthData: packed.mouthData,
  });
  assert.strictEqual(repacked.universeData, packed.universeData);
  assert.strictEqual(repacked.mouthData, packed.mouthData);
});

test('Stage 2 presets have non-overlapping mouths and exact oriented camera crossing', () => {
  const params = makeDefaultParams();

  // Trinity loop: 3 universes, 3 wormholes (6 mouths, 2 per universe)
  const trinity = createPresetMultiverseGraph('trinity-loop', params);
  const packedTrinity = packMultiverseToFloatBuffers(trinity);
  assert.equal(packedTrinity.universeCount, 3);
  assert.equal(packedTrinity.mouthCount, 6);
  for (let u = 0; u < 3; u++) {
    assert.equal(packedTrinity.universeMouthSlices[u].count, 2);
  }

  // Near the right Torus in Universe A -> nearest mouth is wh-ac-torus bridging to Universe C (index 2)
  const nearTorus = findNearestMouthInUniverse(trinity, 0, [2.35, 0, 1.8]);
  assert.ok(nearTorus);
  assert.equal(nearTorus.wormhole.id, 'wh-ac-torus');
  assert.equal(nearTorus.destUniverseIndex, 2);

  // Crossing the oriented Torus from Universe A exits strictly outside partner mouth in Universe C
  const crossed = crossMouthWorld(nearTorus, [2.35, 0, 2.25], [0, 0, 1], true);
  assert.equal(crossed.destUniverseIndex, 2);
  const exitGap = evaluateMouthSdfWorld(
    nearTorus.partnerMouth,
    nearTorus.wormhole.shape,
    nearTorus.wormhole.radius,
    nearTorus.wormhole.scale,
    crossed.exitPosition
  );
  assert.ok(exitGap > 0.05, `expected exit outside partner mouth, got gap=${exitGap}`);

  // Portal Hall: 2 universes, 2 wormholes (3 mouths in Universe A including self-loop A<->A, 1 mouth in B)
  const hall = createPresetMultiverseGraph('portal-hall', params);
  const packedHall = packMultiverseToFloatBuffers(hall);
  assert.equal(packedHall.universeCount, 2);
  assert.equal(packedHall.mouthCount, 4);
  assert.equal(packedHall.universeMouthSlices[0].count, 3);
  assert.equal(packedHall.universeMouthSlices[1].count, 1);

  // Crossing the left Cuboid in Universe A exits at the right Cuboid STILL in Universe A (index 0)
  const nearLeftPortal = findNearestMouthInUniverse(hall, 0, [-2.25, -0.15, 2.4]);
  assert.ok(nearLeftPortal);
  assert.equal(nearLeftPortal.wormhole.id, 'wh-aa-hall');
  assert.equal(nearLeftPortal.destUniverseIndex, 0);
  const hallExit = crossMouthWorld(nearLeftPortal, [-2.25, -0.15, 2.7], [0, 0, 1], true);
  assert.equal(hallExit.destUniverseIndex, 0);
  assert.ok(hallExit.exitPosition[0] > 1.0, `expected exit near right portal (+X), got x=${hallExit.exitPosition[0]}`);
});

test('eulerDegFromQuaternion round-trips quaternionFromEulerDeg across general orientations', () => {
  const cases: Array<[number, number, number]> = [
    [0, 0, 0],
    [45, 0, 0],
    [-120, 32, -18],
    [15, -45, 60],
    [-170, 12, 85],
    [30, 89.99, 0], // near gimbal lock
  ];

  for (const [yaw, pitch, roll] of cases) {
    const q1 = quaternionFromEulerDeg(yaw, pitch, roll);
    const [ry, rp, rr] = eulerDegFromQuaternion(q1);
    const q2 = quaternionFromEulerDeg(ry, rp, rr);

    // q1 and q2 represent the same rotation iff |dot(q1, q2)| ≈ 1
    const dot = Math.abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]);
    assert.ok(
      dot > 0.99999,
      `failed round-trip for (${yaw}, ${pitch}, ${roll}): got (${ry.toFixed(2)}, ${rp.toFixed(2)}, ${rr.toFixed(2)}), dot=${dot}`
    );
  }
});

test('detectMouthOverlaps flags intersecting mouths in same universe and ignores separated or different-universe mouths', () => {
  const params = makeDefaultParams();

  // Presets should have zero overlaps
  assert.equal(detectMouthOverlaps(createPresetMultiverseGraph('classic-binary', params)).length, 0);
  assert.equal(detectMouthOverlaps(createPresetMultiverseGraph('trinity-loop', params)).length, 0);
  assert.equal(detectMouthOverlaps(createPresetMultiverseGraph('portal-hall', params)).length, 0);

  // Overlapping Sphere (r=0.95) and Torus (tube r=0.33) in Universe A -> detected
  const overlappingGraph = createPresetMultiverseGraph('trinity-loop', params);
  overlappingGraph.wormholes[1].mouthA.position = [-1.4, 0, 3.2];
  const warnings = detectMouthOverlaps(overlappingGraph);
  assert.equal(warnings.length, 1);
  assert.equal(warnings[0].universeId, 'A');
  assert.ok(warnings[0].penetrationDepth > 0.25, `expected penetration > 0.25, got ${warnings[0].penetrationDepth}`);

  // Overlapping same-universe self-loop (mouthA and mouthB in A too close) -> detected
  const selfOverlap = createPresetMultiverseGraph('portal-hall', params);
  selfOverlap.wormholes[0].mouthB.position = [-1.6, -0.15, 3.2];
  const selfWarnings = detectMouthOverlaps(selfOverlap);
  assert.ok(selfWarnings.some((w) => w.wormholeAId === 'wh-aa-hall' && w.wormholeBId === 'wh-aa-hall'));
});

test('createDefaultUniverseNode and createDefaultWormholeEdge auto-place without overlaps', () => {
  const graph = createPresetMultiverseGraph('classic-binary', makeDefaultParams());
  const uC = createDefaultUniverseNode(graph.universes);
  assert.equal(uC.id, 'C');
  assert.equal(uC.name, 'Universe C');

  const nextGraph: MultiverseGraph = {
    universes: [...graph.universes, uC],
    wormholes: [...graph.wormholes],
  };

  // Add 3 wormholes into Universe A and verify zero overlaps
  for (let i = 0; i < 3; i++) {
    const wh = createDefaultWormholeEdge(nextGraph, 'A', i === 2 ? 'A' : 'C');
    nextGraph.wormholes.push(wh);
  }
  assert.equal(detectMouthOverlaps(nextGraph).length, 0);
});

test('serializeMultiverseGraph and parseMultiverseGraphJson round-trip cleanly and warn on custom textures', () => {
  const graph = createPresetMultiverseGraph('trinity-loop', makeDefaultParams());
  graph.universes[0].skyMode = 'texture';
  graph.universes[0].textureLayer = 0;
  graph.universes[1].rotationY = 180;
  graph.universes[1].rotationX = -145;

  const json = serializeMultiverseGraph(graph);
  const rawPayload = JSON.parse(json);
  assert.deepEqual(
    rawPayload.wormholes.map((w: { shape: unknown }) => w.shape),
    ['sphere', 'torus', 'octahedron']
  );

  const parsed = parseMultiverseGraphJson(json);

  assert.equal(parsed.graph.universes.length, 3);
  assert.equal(parsed.graph.wormholes.length, 3);
  assert.equal(parsed.graph.universes[0].skyMode, 'grid');
  assert.equal(parsed.graph.universes[0].textureLayer, -1);
  assert.equal(parsed.graph.universes[1].rotationY, 180);
  assert.equal(parsed.graph.universes[1].rotationX, -145);
  assert.equal(parsed.warnings.length, 1);
  assert.equal(parsed.graph.wormholes[0].shape, 0); // Sphere
  assert.equal(parsed.graph.wormholes[1].shape, 3); // Torus preserved
  assert.equal(parsed.graph.wormholes[2].shape, 4); // Octahedron preserved
  assert.deepEqual(parsed.graph.wormholes[2].scale, [1, 1.15, 1]);

  // Case-insensitive names, legacy numeric IDs, and invalid shape warning
  rawPayload.universes[0].skyMode = 'grid';
  delete rawPayload.universes[0].hadCustomTexture;
  rawPayload.wormholes[0].shape = 'Cuboid';
  rawPayload.wormholes[1].shape = 2; // legacy Cylinder ID
  rawPayload.wormholes[2].shape = 'pyramid'; // unknown -> warning + fallback sphere
  const compatParsed = parseMultiverseGraphJson(JSON.stringify(rawPayload));
  assert.equal(compatParsed.graph.wormholes[0].shape, 1);
  assert.equal(compatParsed.graph.wormholes[1].shape, 2);
  assert.equal(compatParsed.graph.wormholes[2].shape, 0);
  assert.equal(compatParsed.warnings.length, 1);
  assert.match(compatParsed.warnings[0], /unrecognized shape/i);

  // Malformed JSON throws descriptive error
  assert.throws(() => parseMultiverseGraphJson('{not valid'), /Invalid JSON/);
  assert.throws(() => parseMultiverseGraphJson('{"universes":[]}'), /at least one universe/);
});

test('per-universe grid line color overrides pack cleanly, clear on reuse, and round-trip JSON', () => {
  const params = makeDefaultParams();
  params.universeA.grid.lonOverrides = { 0: '#facc15', 9: '#ffffff', [-1]: '#123456', 99: '#654321' };
  params.universeA.grid.latOverrides = { 4: '#22c55e', 40: '#ff0000' };
  params.universeB.grid.lonOverrides = { 3: '#38bdf8' };

  const graph = createPresetMultiverseGraph('classic-binary', params);
  const packed = packMultiverseToFloatBuffers(graph);

  assert.equal(packed.gridOverrideWidth, GRID_OVERRIDE_TEX_WIDTH);
  assert.equal(packed.gridOverrideHeight, 2);
  assert.equal(packed.gridOverrideData.length, GRID_OVERRIDE_TEX_WIDTH * 2 * 4);

  const readTexel = (uIdx: number, x: number) => {
    const off = (uIdx * GRID_OVERRIDE_TEX_WIDTH + x) * 4;
    return [
      packed.gridOverrideData[off],
      packed.gridOverrideData[off + 1],
      packed.gridOverrideData[off + 2],
      packed.gridOverrideData[off + 3],
    ];
  };

  // Universe A overrides: lon 0 (#facc15), lon 9 (#ffffff), lat 4 (#22c55e)
  assert.deepEqual(readTexel(0, 0), [0xfa, 0xcc, 0x15, 255]);
  assert.deepEqual(readTexel(0, 9), [0xff, 0xff, 0xff, 255]);
  assert.deepEqual(readTexel(0, 1), [0, 0, 0, 0]); // non-overridden
  assert.deepEqual(readTexel(0, GRID_OVERRIDE_LAT_OFFSET + 4), [0x22, 0xc5, 0x5e, 255]);
  assert.deepEqual(readTexel(0, GRID_OVERRIDE_LAT_OFFSET + 5), [0, 0, 0, 0]);

  // Universe B overrides: lon 3 (#38bdf8), lon 0 unset
  assert.deepEqual(readTexel(1, 3), [0x38, 0xbd, 0xf8, 255]);
  assert.deepEqual(readTexel(1, 0), [0, 0, 0, 0]);

  // Removing an override and reusing the buffer zeroes the previous texel without reallocation
  const updatedGraph: MultiverseGraph = {
    ...graph,
    universes: graph.universes.map((u) =>
      u.id === 'A' ? { ...u, gridLonOverrides: { 9: '#ffffff' } } : u
    ),
  };
  const repacked = packMultiverseToFloatBuffers(updatedGraph, {
    universeData: packed.universeData,
    mouthData: packed.mouthData,
    gridOverrideData: packed.gridOverrideData,
  });
  assert.strictEqual(repacked.gridOverrideData, packed.gridOverrideData);
  assert.deepEqual(readTexel(0, 0), [0, 0, 0, 0]); // cleared
  assert.deepEqual(readTexel(0, 9), [0xff, 0xff, 0xff, 255]); // kept

  // JSON serialization & import preserves valid overrides and drops invalid entries
  const serialized = serializeMultiverseGraph(graph);
  const rawObj = JSON.parse(serialized);
  rawObj.universes[0].gridLonOverrides['bad'] = '#ffffff';
  rawObj.universes[0].gridLonOverrides['2'] = 'not-hex';
  rawObj.universes[0].gridLatOverrides['-5'] = '#00ff00';
  const imported = parseMultiverseGraphJson(JSON.stringify(rawObj));

  assert.deepEqual(imported.graph.universes[0].gridLonOverrides, { 0: '#facc15', 9: '#ffffff' });
  assert.deepEqual(imported.graph.universes[0].gridLatOverrides, { 4: '#22c55e' });
  assert.deepEqual(imported.graph.universes[1].gridLonOverrides, { 3: '#38bdf8' });
});

test('serializeMultiverseGraph and parseMultiverseGraphJson persist multi-universe camera keyframes', () => {
  const graph = createPresetMultiverseGraph('trinity-loop', makeDefaultParams());
  const keyframes = [
    {
      id: 'kf-a',
      label: 'Start A',
      universeId: 'A',
      room: 0,
      position: [0, 0, -1] as [number, number, number],
      quaternion: [0, 0, 0, 1] as [number, number, number, number],
      target: [0, 0, 3] as [number, number, number],
    },
    {
      id: 'kf-c',
      label: 'End C',
      universeId: 'C',
      room: 2,
      position: [-2, 0.5, 5] as [number, number, number],
      quaternion: [0, 1, 0, 0] as [number, number, number, number],
      target: [-2, 0.5, 8] as [number, number, number],
    },
  ];

  const json = serializeMultiverseGraph(graph, keyframes);
  const parsed = parseMultiverseGraphJson(json);

  assert.equal(parsed.keyframes.length, 2);
  assert.equal(parsed.keyframes[0].universeId, 'A');
  assert.equal(parsed.keyframes[0].room, 0);
  assert.equal(parsed.keyframes[1].universeId, 'C');
  assert.equal(parsed.keyframes[1].room, 2);
  assert.deepEqual(parsed.keyframes[1].position, [-2, 0.5, 5]);
});



