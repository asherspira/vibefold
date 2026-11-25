import * as THREE from 'three';
import type {
  MouthOverlapWarning,
  MultiverseGraph,
  WormholeEdge,
  WormholeMouthPlacement,
} from './wormholeMultiverse';
import type { WormholeShapeId } from './wormholeSdf';

/**
 * Creates a wireframe LineSegments geometry matching wormholeSdf.ts proportions
 * at unit radius (scaled afterwards by [radius * sx, radius * sy, radius * sz]).
 */
function createUnitShapeWireframeGeometry(shape: WormholeShapeId): THREE.BufferGeometry {
  switch (shape) {
    case 1: {
      // Cuboid half-extents (1.2, 0.72, 0.9) => full extents (2.4, 1.44, 1.8)
      const box = new THREE.BoxGeometry(2.4, 1.44, 1.8);
      const edges = new THREE.EdgesGeometry(box, 15);
      box.dispose();
      return edges;
    }
    case 2: {
      // Capped cylinder along Y: radius 0.85, half-height 1.45 => height 2.9
      const cyl = new THREE.CylinderGeometry(0.85, 0.85, 2.9, 24, 2);
      const edges = new THREE.EdgesGeometry(cyl, 18);
      cyl.dispose();
      return edges;
    }
    case 3: {
      // Torus with hole along Z: major 1.0, minor 0.36
      const torus = new THREE.TorusGeometry(1.0, 0.36, 10, 28);
      const wire = new THREE.WireframeGeometry(torus);
      torus.dispose();
      return wire;
    }
    case 4: {
      // Regular octahedron of size 1.55 (vertices at ±1.55 on axes)
      const octa = new THREE.OctahedronGeometry(1.55, 0);
      const edges = new THREE.EdgesGeometry(octa, 5);
      octa.dispose();
      return edges;
    }
    case 5: {
      // Double cone (bicone): apexes at y = ±1.5, equator radius = 1.0
      const profile = [
        new THREE.Vector2(0.0001, -1.5),
        new THREE.Vector2(1.0, 0),
        new THREE.Vector2(0.0001, 1.5),
      ];
      const lathe = new THREE.LatheGeometry(profile, 20);
      const edges = new THREE.EdgesGeometry(lathe, 18);
      lathe.dispose();
      return edges;
    }
    case 0:
    default: {
      // Unit sphere
      const sphere = new THREE.SphereGeometry(1.0, 18, 12);
      const wire = new THREE.WireframeGeometry(sphere);
      sphere.dispose();
      return wire;
    }
  }
}

/**
 * Builds a small orientation crosshair (+Z portal forward & +Y up tick)
 * so rotating a mouth's Yaw/Pitch/Roll is immediately readable in 3D space.
 */
function createOrientationAxisGeometry(): THREE.BufferGeometry {
  const positions = new Float32Array([
    // +Z forward axis
    0, 0, 0,
    0, 0, 1.45,
    // Arrowhead on +Z
    0, 0, 1.45,
    0.14, 0, 1.22,
    0, 0, 1.45,
    -0.14, 0, 1.22,
    // +Y up indicator
    0, 0, 0,
    0, 0.7, 0,
  ]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  return geo;
}

export function disposeOverlayGroup(group: THREE.Group): void {
  group.traverse((obj) => {
    if (
      obj instanceof THREE.Line ||
      obj instanceof THREE.LineSegments ||
      obj instanceof THREE.Mesh
    ) {
      obj.geometry.dispose();
      if (Array.isArray(obj.material)) {
        for (const m of obj.material) m.dispose();
      } else {
        obj.material.dispose();
      }
    }
  });
  group.clear();
}

export interface OverlayKeyframeMarker {
  position: readonly [number, number, number];
  isStart: boolean;
  isEnd: boolean;
}

export interface OverlaySyncParams {
  graph: MultiverseGraph;
  activeUniverseIndex: number;
  selectedWormholeId: string | null;
  selectedEnd?: 'A' | 'B' | null;
  overlaps: readonly MouthOverlapWarning[];
  /** Optional sampled camera trajectory polylines inside the active universe. */
  cameraPathPolylines?: ReadonlyArray<ReadonlyArray<readonly [number, number, number]>>;
  /** Optional camera keyframe positions inside the active universe. */
  keyframeMarkers?: readonly OverlayKeyframeMarker[];
}

/**
 * Rebuilds the translucent 3D wireframe placement overlay for all wormhole mouths
 * residing in the observer's active universe.
 *
 * Uses depthTest: false so cages draw cleanly over the raymarched full-screen quad
 * without requiring a depth buffer or altering shader rendering.
 */
export function syncWormholeOverlayGroup(
  group: THREE.Group,
  params: OverlaySyncParams
): void {
  disposeOverlayGroup(group);

  const universe = params.graph.universes[params.activeUniverseIndex];
  if (!universe) return;

  // Track which (wormholeId, end) pairs are currently flagged as overlapping
  const overlappingKeys = new Set<string>();
  for (const w of params.overlaps) {
    if (w.universeId === universe.id) {
      overlappingKeys.add(`${w.wormholeAId}:${w.endA}`);
      overlappingKeys.add(`${w.wormholeBId}:${w.endB}`);
    }
  }

  const addMouthWireframe = (wh: WormholeEdge, end: 'A' | 'B', mouth: WormholeMouthPlacement) => {
    const key = `${wh.id}:${end}`;
    const isOverlapping = overlappingKeys.has(key);
    const isSelected =
      params.selectedWormholeId === wh.id &&
      (!params.selectedEnd || params.selectedEnd === end);
    const isSameWormholeOtherEnd =
      params.selectedWormholeId === wh.id && params.selectedEnd && params.selectedEnd !== end;

    const color = isOverlapping
      ? 0xfb7185 // Rose warning
      : isSelected
        ? 0xf0abfc // Bright fuchsia selected
        : isSameWormholeOtherEnd
          ? 0xc084fc // Soft violet partner mouth in same universe
          : 0x38bdf8; // Calm cyan

    const opacity = isOverlapping ? 0.92 : isSelected ? 0.85 : isSameWormholeOtherEnd ? 0.65 : 0.42;

    const holder = new THREE.Group();
    holder.position.set(mouth.position[0], mouth.position[1], mouth.position[2]);
    holder.quaternion.set(
      mouth.quaternion[0],
      mouth.quaternion[1],
      mouth.quaternion[2],
      mouth.quaternion[3]
    ).normalize();

    const r = Math.max(wh.radius, 0.05);
    const sx = Math.max(wh.scale[0], 0.05);
    const sy = Math.max(wh.scale[1], 0.05);
    const sz = Math.max(wh.scale[2], 0.05);

    const wireGeo = createUnitShapeWireframeGeometry(wh.shape);
    const wireMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthTest: false,
      depthWrite: false,
    });
    const wireLines = new THREE.LineSegments(wireGeo, wireMat);
    wireLines.scale.set(r * sx, r * sy, r * sz);
    wireLines.renderOrder = isOverlapping ? 30 : isSelected ? 20 : 10;
    holder.add(wireLines);

    // Local orientation arrow scaled to uniform radius so it's readable without distortion
    const axisGeo = createOrientationAxisGeometry();
    const axisMat = new THREE.LineBasicMaterial({
      color: isOverlapping ? 0xfda4af : isSelected ? 0xfef08a : 0x94a3b8,
      transparent: true,
      opacity: isSelected || isOverlapping ? 0.9 : 0.45,
      depthTest: false,
      depthWrite: false,
    });
    const axisLines = new THREE.LineSegments(axisGeo, axisMat);
    const axisScale = Math.max(0.45, r * Math.max(sx, sy, sz) * 0.85);
    axisLines.scale.setScalar(axisScale);
    axisLines.renderOrder = isOverlapping ? 31 : isSelected ? 21 : 11;
    holder.add(axisLines);

    group.add(holder);
  };

  for (const wh of params.graph.wormholes) {
    if (wh.mouthA.universeId === universe.id) {
      addMouthWireframe(wh, 'A', wh.mouthA);
    }
    if (wh.mouthB.universeId === universe.id) {
      addMouthWireframe(wh, 'B', wh.mouthB);
    }
  }

  // Stage 4: Camera trajectory preview curve inside the active universe
  if (params.cameraPathPolylines && params.cameraPathPolylines.length > 0) {
    for (const poly of params.cameraPathPolylines) {
      if (poly.length < 2) continue;
      const flat = new Float32Array(poly.length * 3);
      for (let i = 0; i < poly.length; i++) {
        flat[i * 3] = poly[i][0];
        flat[i * 3 + 1] = poly[i][1];
        flat[i * 3 + 2] = poly[i][2];
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xfacc15,
        transparent: true,
        opacity: 0.85,
        depthTest: false,
        depthWrite: false,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.renderOrder = 35;
      group.add(line);
    }
  }

  // Stage 4: Keyframe waypoint diamond markers inside the active universe
  if (params.keyframeMarkers && params.keyframeMarkers.length > 0) {
    for (const marker of params.keyframeMarkers) {
      const oct = new THREE.OctahedronGeometry(0.16, 0);
      const edges = new THREE.EdgesGeometry(oct, 5);
      oct.dispose();
      const color = marker.isStart ? 0x34d399 : marker.isEnd ? 0xf472b6 : 0xfacc15;
      const mat = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
        depthWrite: false,
      });
      const diamond = new THREE.LineSegments(edges, mat);
      diamond.position.set(marker.position[0], marker.position[1], marker.position[2]);
      diamond.renderOrder = 40;
      group.add(diamond);
    }
  }
}
