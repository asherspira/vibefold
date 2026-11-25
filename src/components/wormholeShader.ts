// Gravitational wormhole integrator adapted from Kirill Osipov ("Zabidon"),
// Shadertoy 4dccRr "Working wormhole" (MIT), upgraded to WebGL2 / GLSL ES 3.00
// with multi-mouth per-universe raymarching, oriented mouth transforms,
// two-tier near/far-field gravity evaluation, cluster escape, and portal bounce caps.
//
// Each wormhole bridges two mouths that share the same signed distance field
// shape, radius, scale, and gravity ratio (see src/utils/wormholeSdf.ts and
// src/utils/wormholeMultiverse.ts). Every throat shape is symmetric across its
// local XY plane, f(x, y, -z) = f(x, y, z), so gluing across R_z = diag(1, 1, -1)
// with exit direction R_z * reflect(incoming, normal) in mouth-local coordinates
// is an orientation-preserving SO(3) transition (det = +1) that keeps world +Y
// upright across the threshold instead of flipping the far sky upside down.

export const WORMHOLE_VERTEX_SHADER = /* glsl */ `
void main() {
  // PlaneGeometry(2, 2) is already in clip space. Bypass the camera so the
  // quad cannot be clipped by the near plane.
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const WORMHOLE_FRAGMENT_SHADER = /* glsl */ `
precision highp sampler2DArray;

layout(location = 0) out vec4 fragColor;

uniform vec2 uResolution;
uniform vec3 uCameraPos;
uniform vec4 uCameraQuat;
uniform int uMaxSteps;
uniform int uMaxCrossings;
uniform int uDiagnosticMode; // 0 = normal, 1 = diagnostic view, 2 = raw RGBA float data
uniform int uStartUniverse;
uniform int uUniverseCount;
uniform int uMouthCount;
uniform highp sampler2D uUniverseData;      // width = 5, height = max(1, uUniverseCount), RGBA32F
uniform highp sampler2D uMouthData;         // width = 4, height = max(1, uMouthCount), RGBA32F
uniform highp sampler2D uGridLineOverrides; // width = 112, height = max(1, uUniverseCount), RGBA8
uniform highp sampler2DArray uSkyArray;     // RGBA16F equirectangular sky layers (bottom-up v=0..1)

const float PI = 3.14159265359;
const float PI2 = 6.28318530718;
const float LIGHT_SPEED = 0.3;
const float TAN_HALF_FOV = 0.57735026919; // 60 degree vertical field of view
const int MAX_MOUTHS_PER_UNIVERSE = 32;
const int GRID_OVERRIDE_MAX_LON = 72;
const int GRID_OVERRIDE_MAX_LAT = 37;
const int GRID_OVERRIDE_LAT_OFFSET = 72;

struct UniverseParams {
  int skyMode;      // 0 = procedural grid, 1 = texture
  int texLayer;     // layer index into uSkyArray, or -1 if none
  int isHdr;        // 0 = LDR sRGB, 1 = linear HDRI
  float exposure;   // linear exposure gain (2^EV)
  vec2 gridCells;   // (longitude cells, latitude cells)
  vec2 rot;         // (yaw radians, pitch radians)
  vec3 colorLon;    // sRGB 0..1
  vec3 colorLat;    // sRGB 0..1
  int mouthOffset;  // first row in uMouthData belonging to this universe
  int mouthCount;   // number of mouths in this universe
  vec3 clusterCenter;
  float clusterRadius;
};

struct MouthParams {
  vec3 center;
  float radius;
  vec4 quat;
  vec3 scale;
  int shape;
  float gravityRatio;
  int universeIdx;
  int destUniverseIdx;
  int destMouthIdx;
};

vec3 quatRotate(vec4 quat, vec3 dir) {
  return dir + 2.0 * cross(quat.xyz, cross(quat.xyz, dir) + quat.w * dir);
}

vec3 quatRotateInv(vec4 quat, vec3 dir) {
  vec4 inv = vec4(-quat.xyz, quat.w);
  return dir + 2.0 * cross(inv.xyz, cross(inv.xyz, dir) + inv.w * dir);
}

UniverseParams fetchUniverse(int uIdx) {
  int safeIdx = clamp(uIdx, 0, max(uUniverseCount - 1, 0));
  vec4 t0 = texelFetch(uUniverseData, ivec2(0, safeIdx), 0);
  vec4 t1 = texelFetch(uUniverseData, ivec2(1, safeIdx), 0);
  vec4 t2 = texelFetch(uUniverseData, ivec2(2, safeIdx), 0);
  vec4 t3 = texelFetch(uUniverseData, ivec2(3, safeIdx), 0);
  vec4 t4 = texelFetch(uUniverseData, ivec2(4, safeIdx), 0);

  UniverseParams u;
  u.skyMode = int(floor(t0.x + 0.5));
  u.texLayer = int(floor(t0.y + 0.5));
  u.isHdr = int(floor(t0.z + 0.5));
  u.exposure = t0.w;
  u.gridCells = t1.xy;
  u.rot = t1.zw;
  u.colorLon = t2.xyz;
  u.mouthOffset = int(floor(t2.w + 0.5));
  u.colorLat = t3.xyz;
  u.mouthCount = int(floor(t3.w + 0.5));
  u.clusterCenter = t4.xyz;
  u.clusterRadius = t4.w;
  return u;
}

MouthParams fetchMouth(int mIdx) {
  int safeIdx = clamp(mIdx, 0, max(uMouthCount - 1, 0));
  vec4 t0 = texelFetch(uMouthData, ivec2(0, safeIdx), 0);
  vec4 t1 = texelFetch(uMouthData, ivec2(1, safeIdx), 0);
  vec4 t2 = texelFetch(uMouthData, ivec2(2, safeIdx), 0);
  vec4 t3 = texelFetch(uMouthData, ivec2(3, safeIdx), 0);

  MouthParams m;
  m.center = t0.xyz;
  m.radius = max(t0.w, 0.001);
  m.quat = t1;
  m.scale = max(t2.xyz, vec3(0.05));
  m.shape = int(floor(t2.w + 0.5));
  m.gravityRatio = t3.x;
  m.universeIdx = int(floor(t3.y + 0.5));
  m.destUniverseIdx = int(floor(t3.z + 0.5));
  m.destMouthIdx = int(floor(t3.w + 0.5));
  return m;
}

// Proportions match wormholeSdf.ts. Each field is even: f(-p) = f(p).
float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCappedCylinder(vec3 p, float h, float r) {
  vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdTorus(vec3 p, float major, float minor) {
  return length(vec2(length(p.xy) - major, p.z)) - minor;
}

float sdOctahedron(vec3 p, float s) {
  p = abs(p);
  float m = p.x + p.y + p.z - s;
  vec3 q;
  if (3.0 * p.x < m) q = p.xyz;
  else if (3.0 * p.y < m) q = p.yzx;
  else if (3.0 * p.z < m) q = p.zxy;
  else return m * 0.57735026919;
  float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
  return length(vec3(q.x, q.y - s + k, q.z - k));
}

float distToSegment(vec2 p, vec2 a, vec2 b) {
  vec2 ab = b - a;
  float denom = dot(ab, ab);
  float t = denom > 0.0 ? clamp(dot(p - a, ab) / denom, 0.0, 1.0) : 0.0;
  return length(p - a - ab * t);
}

float sdBicone(vec3 p, float h, float r) {
  float rho = length(p.xz);
  vec2 q = vec2(rho, p.y);
  float d = min(
    distToSegment(q, vec2(0.0, h), vec2(r, 0.0)),
    distToSegment(q, vec2(0.0, -h), vec2(r, 0.0))
  );
  float limit = r * (1.0 - abs(p.y) / h);
  return (abs(p.y) <= h && rho <= limit) ? -d : d;
}

float sdMouth(vec3 p, int shape, float radius, vec3 shapeScale) {
  float r = max(radius, 0.001);
  vec3 s = max(shapeScale, vec3(0.05));
  if (shape == 1) return sdBox(p, vec3(1.2, 0.72, 0.9) * r * s);
  vec3 q = p / s;
  float sMin = min(s.x, min(s.y, s.z));
  if (shape == 2) return sMin * sdCappedCylinder(q, 1.45 * r, 0.85 * r);
  if (shape == 3) return sMin * sdTorus(q, r, 0.36 * r);
  if (shape == 4) return sMin * sdOctahedron(q, 1.55 * r);
  if (shape == 5) return sMin * sdBicone(q, 1.5 * r, r);
  return sMin * (length(q) - r);
}

vec3 sdfMouthNormal(vec3 p, int shape, float radius, vec3 shapeScale) {
  vec3 s = max(shapeScale, vec3(0.05));
  float sMin = min(s.x, min(s.y, s.z));
  float e = max(radius * sMin * 1.5e-3, 1e-4);
  vec3 g = vec3(
    sdMouth(p + vec3(e, 0.0, 0.0), shape, radius, s) - sdMouth(p - vec3(e, 0.0, 0.0), shape, radius, s),
    sdMouth(p + vec3(0.0, e, 0.0), shape, radius, s) - sdMouth(p - vec3(0.0, e, 0.0), shape, radius, s),
    sdMouth(p + vec3(0.0, 0.0, e), shape, radius, s) - sdMouth(p - vec3(0.0, 0.0, e), shape, radius, s)
  );
  float glen = length(g);
  if (glen > 1e-6) return g / glen;
  float plen = length(p);
  if (plen > 1e-6) return p / plen;
  return vec3(0.0, 0.0, 1.0);
}

// Two-tier near/far-field evaluator:
// - Single-mouth universes or mouths within 2.4 * maxExtent use exact oriented SDF + 6-tap gradient.
// - Distant mouths (> 3.2 * maxExtent) in multi-mouth universes use a fast analytical radial field.
// - Smoothstep blends across [2.4, 3.2] * maxExtent so there is zero lensing seam.
void evalMouthField(
  MouthParams m,
  vec3 position,
  bool allowFarApprox,
  out float gap,
  out vec3 outwardN,
  out vec3 localPos,
  out vec3 localN,
  out float dist,
  out float effectiveR,
  out float maxExtent,
  out bool isExactNear
) {
  vec3 worldOffset = position - m.center;
  dist = length(worldOffset);
  float minScale = min(m.scale.x, min(m.scale.y, m.scale.z));
  float maxScale = max(m.scale.x, max(m.scale.y, m.scale.z));
  effectiveR = max(m.radius * minScale, 0.001);
  maxExtent = m.radius * maxScale;

  float farStart = maxExtent * 2.4;
  float farEnd = maxExtent * 3.2;

  if (allowFarApprox && dist >= farEnd) {
    float avgScale = (m.scale.x + m.scale.y + m.scale.z) * 0.33333333;
    gap = max(dist - m.radius * avgScale, 0.0);
    outwardN = dist > 1e-4 ? worldOffset / dist : vec3(0.0, 0.0, 1.0);
    localPos = vec3(0.0);
    localN = vec3(0.0, 0.0, 1.0);
    isExactNear = false;
    return;
  }

  localPos = quatRotateInv(m.quat, worldOffset);
  float exactGap = sdMouth(localPos, m.shape, m.radius, m.scale);
  localN = sdfMouthNormal(localPos, m.shape, m.radius, m.scale);
  vec3 exactN = quatRotate(m.quat, localN);

  if (allowFarApprox && dist > farStart) {
    float blend = smoothstep(farStart, farEnd, dist);
    float avgScale = (m.scale.x + m.scale.y + m.scale.z) * 0.33333333;
    float farGap = max(dist - m.radius * avgScale, 0.0);
    vec3 farN = dist > 1e-4 ? worldOffset / dist : exactN;
    gap = mix(exactGap, farGap, blend);
    vec3 blendedN = mix(exactN, farN, blend);
    float bnLen = length(blendedN);
    outwardN = bnLen > 1e-5 ? blendedN / bnLen : exactN;
    isExactNear = false;
  } else {
    gap = exactGap;
    outwardN = exactN;
    isExactNear = true;
  }
}

vec2 directionToEquirect(vec3 dir, vec2 rot) {
  // Sample R^{-1} dir for a sky rotated by Rx(pitch) then Ry(yaw).
  // Both axes use the inverse: Rx(-pitch) first, then Ry(-yaw).
  float cx = cos(rot.y);
  float sx = sin(rot.y);
  vec3 pitched = vec3(
    dir.x,
    dir.y * cx + dir.z * sx,
    -dir.y * sx + dir.z * cx
  );
  float cy = cos(rot.x);
  float sy = sin(rot.x);
  vec3 rDir = vec3(
    pitched.x * cy - pitched.z * sy,
    pitched.y,
    pitched.x * sy + pitched.z * cy
  );

  // Align +Z (dir = (0, 0, 1) directly facing wormhole) to image center u = 0.5, v = 0.5
  float u = fract((atan(rDir.z, rDir.x) - 0.5 * PI) / PI2 + 0.5);
  float v = clamp(asin(clamp(rDir.y, -1.0, 1.0)) / PI + 0.5, 0.001, 0.999);
  return vec2(u, v);
}

vec2 directionToEquirectDiagnostic(vec3 dir, vec2 rot, out float cosLat) {
  float cx = cos(rot.y);
  float sx = sin(rot.y);
  vec3 pitched = vec3(
    dir.x,
    dir.y * cx + dir.z * sx,
    -dir.y * sx + dir.z * cx
  );
  float cy = cos(rot.x);
  float sy = sin(rot.x);
  vec3 rDir = vec3(
    pitched.x * cy - pitched.z * sy,
    pitched.y,
    pitched.x * sy + pitched.z * cy
  );

  float clampedY = clamp(rDir.y, -1.0, 1.0);
  cosLat = sqrt(max(1.0 - clampedY * clampedY, 1e-6));
  float u = fract((atan(rDir.z, rDir.x) - 0.5 * PI) / PI2 + 0.5);
  float v = asin(clampedY) / PI + 0.5;
  return vec2(u, v);
}

vec3 hslToRgb(float h, float s, float l) {
  vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
}

// Distinct flat muted/pastel tint per discrete (universe, crossings) phase region
vec3 diagnosticRegionColor(int uIdx, int crossings, bool hitCap) {
  if (hitCap) {
    return vec3(0.06, 0.065, 0.08);
  }
  // Golden-ratio-inspired hue separation between universes and crossing orders
  float hue = fract(0.58 + float(uIdx) * 0.381966 + float(crossings) * 0.217391);
  // Modulate luminance slightly across crossing orders so adjacent bounce shells contrast crisply
  float band = mod(float(crossings), 3.0);
  float sat = 0.36 + 0.06 * mod(float(uIdx + crossings), 2.0);
  float lum = 0.27 + 0.075 * band;
  return hslToRgb(hue, sat, lum);
}

float gridLine(float coord, float cells) {
  float x = coord * cells;
  float distToLine = min(fract(x), 1.0 - fract(x));
  // fwidth jumps by a full turn where equirectangular u wraps 1 -> 0.
  float width = max(min(fwidth(x), 0.5) * 1.35, 0.018);
  return 1.0 - smoothstep(width * 0.15, width, distToLine);
}

// Longitude (U) uses verticalColor, latitude (V) uses horizontalColor, with optional
// per-line color overrides fetched from uGridLineOverrides for spatial landmarks/waypoints.
vec3 proceduralGrid(
  vec3 dir,
  int safeUIdx,
  vec2 rot,
  vec2 cells,
  vec3 verticalColor,
  vec3 horizontalColor
) {
  vec2 uv = directionToEquirect(dir, rot);
  float cols = max(cells.x, 1.0);
  float rows = max(cells.y, 1.0);
  float vertical = gridLine(uv.x, cols);
  float horizontal = gridLine(uv.y, rows);

  vec3 lonCol = verticalColor;
  if (vertical > 0.001) {
    int colCount = max(int(floor(cols + 0.5)), 1);
    int lonIdx = int(floor(uv.x * cols + 0.5)) % colCount;
    if (lonIdx < 0) lonIdx += colCount;
    if (lonIdx < GRID_OVERRIDE_MAX_LON) {
      vec4 ov = texelFetch(uGridLineOverrides, ivec2(lonIdx, safeUIdx), 0);
      if (ov.a > 0.5) lonCol = ov.rgb;
    }
  }

  vec3 latCol = horizontalColor;
  if (horizontal > 0.001) {
    int rowCount = max(int(floor(rows + 0.5)), 1);
    int latIdx = clamp(int(floor(uv.y * rows + 0.5)), 0, rowCount);
    if (latIdx < GRID_OVERRIDE_MAX_LAT) {
      vec4 ov = texelFetch(uGridLineOverrides, ivec2(GRID_OVERRIDE_LAT_OFFSET + latIdx, safeUIdx), 0);
      if (ov.a > 0.5) latCol = ov.rgb;
    }
  }

  return clamp(lonCol * vertical + latCol * horizontal, 0.0, 1.0);
}

// ACES Filmic tone mapping (Narkowicz 2015) + IEC sRGB OETF for linear HDRIs
vec3 acesToneMap(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linearToSrgb(vec3 c) {
  vec3 clamped = max(c, vec3(0.0));
  vec3 low = clamped * 12.92;
  vec3 high = 1.055 * pow(clamped, vec3(1.0 / 2.4)) - 0.055;
  return mix(low, high, step(vec3(0.0031308), clamped));
}

vec3 sampleSkyLayer(vec2 uv, int layer, int isHdr, float exposure) {
  vec3 rgb = texture(uSkyArray, vec3(uv, float(layer))).rgb;
  if (isHdr == 1) {
    rgb = max(rgb, vec3(0.0)) * max(exposure, 0.0);
    rgb = linearToSrgb(acesToneMap(rgb));
  }
  return rgb;
}

vec3 sampleUniverse(vec3 dir, int uIdx) {
  int safeIdx = clamp(uIdx, 0, max(uUniverseCount - 1, 0));
  UniverseParams u = fetchUniverse(safeIdx);
  if (u.skyMode == 0 || u.texLayer < 0) {
    return proceduralGrid(dir, safeIdx, u.rot, u.gridCells, u.colorLon, u.colorLat);
  }
  return sampleSkyLayer(directionToEquirect(dir, u.rot), u.texLayer, u.isHdr, u.exposure);
}

void main() {
  vec2 p = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
  p.x *= uResolution.x / max(uResolution.y, 1.0);
  vec3 camRayRaw = vec3(p * TAN_HALF_FOV, -1.0);
  vec3 ray = normalize(camRayRaw);
  ray = quatRotate(normalize(uCameraQuat), ray);

  vec3 position = uCameraPos;
  int currentUniverse = clamp(uStartUniverse, 0, max(uUniverseCount - 1, 0));
  UniverseParams activeU = fetchUniverse(currentUniverse);

  int maxCrossings = max(uMaxCrossings, 1);
  int crossings = 0;
  bool hitBounceCap = false;
  int lastExitedMouth = -1;
  int exitGuardSteps = 0;

  for (int i = 0; i < 150; i++) {
    if (i >= uMaxSteps || uMouthCount <= 0 || activeU.mouthCount <= 0) break;

    // Cluster escape: stop integrating once outbound well past every mouth in this universe.
    // For a single-mouth universe, clusterCenter == mouth.center and clusterRadius == maxExtent.
    vec3 clusterOffset = position - activeU.clusterCenter;
    float clusterDist = length(clusterOffset);
    vec3 fromCluster = clusterDist > 1e-4 ? clusterOffset / clusterDist : vec3(0.0, 0.0, 1.0);
    if (dot(ray, fromCluster) > 0.05 && clusterDist > max(activeU.clusterRadius, 0.001) * 16.0) {
      break;
    }

    int localMouthCount = min(activeU.mouthCount, MAX_MOUTHS_PER_UNIVERSE);
    bool allowFar = localMouthCount > 1;

    int hitMouthIdx = -1;
    MouthParams hitMouth;
    vec3 hitLocalPos = vec3(0.0);
    vec3 hitLocalN = vec3(0.0, 0.0, 1.0);
    vec3 hitLocalRay = ray;
    float hitEffectiveR = 0.1;
    float hitEpsBest = 1e-4;
    float hitGap = 1e9;

    vec3 accel1 = vec3(0.0);
    float dt = 1e9;

    // Pass 1: evaluate all mouths in the current universe for crossing + gravity + safe dt
    for (int k = 0; k < MAX_MOUTHS_PER_UNIVERSE; k++) {
      if (k >= localMouthCount) break;
      int mIdx = activeU.mouthOffset + k;
      MouthParams m = fetchMouth(mIdx);

      float gap;
      vec3 outwardN;
      vec3 localPos;
      vec3 localN;
      float dist;
      float effectiveR;
      float maxExtent;
      bool isExactNear;

      evalMouthField(
        m,
        position,
        allowFar,
        gap,
        outwardN,
        localPos,
        localN,
        dist,
        effectiveR,
        maxExtent,
        isExactNear
      );

      float hitEps = max(effectiveR * 0.02, 1e-4);
      bool guarded = (mIdx == lastExitedMouth && exitGuardSteps > 0);

      if (isExactNear && !guarded && gap < hitEps) {
        vec3 localRay = quatRotateInv(m.quat, ray);
        if (dot(localRay, localN) < 0.15 && gap < hitGap) {
          hitMouthIdx = mIdx;
          hitMouth = m;
          hitLocalPos = localPos;
          hitLocalN = localN;
          hitLocalRay = localRay;
          hitEffectiveR = effectiveR;
          hitEpsBest = hitEps;
          hitGap = gap;
        }
      }

      // Per-mouth adaptive step size and gravitational acceleration
      float shell = effectiveR * 0.1;
      float gravityScale = m.radius * LIGHT_SPEED * LIGHT_SPEED;
      float scale = max(dist, effectiveR);
      float scaled = effectiveR * clamp(0.18 * scale / effectiveR, 0.05, 1.1);
      float legacy = clamp(0.18 * scale, 0.05, 1.1);
      float dtM = max(max(scaled, legacy), effectiveR * 0.015);

      float denom = gap + m.radius * m.gravityRatio;
      float safe = max(denom, shell);
      float amount = m.gravityRatio / safe;
      float accelMag = gravityScale * amount * amount;

      float bend = accelMag * dtM / LIGHT_SPEED;
      if (bend > 0.35) dtM *= 0.35 / bend;

      if (dot(ray, outwardN) < 0.0) {
        float cover = 0.85 * max(gap, effectiveR * 0.004) / LIGHT_SPEED;
        if (dtM > cover) dtM = cover;
      }

      dt = min(dt, dtM);
      accel1 += -outwardN * accelMag;
    }

    if (hitMouthIdx >= 0) {
      crossings += 1;
      if (crossings > maxCrossings) {
        hitBounceCap = true;
        break;
      }

      // Heading into the surface: glue across R_z = diag(1, 1, -1) with direction
      // R_z * reflect(hitLocalRay, hitLocalN) in mouth-local coordinates. Both bridged
      // mouths share the same throat shape/radius/scale so exitLocal sits on the
      // exact partner surface with zero vertical flip.
      vec3 exitLocalRay = reflect(hitLocalRay, hitLocalN);
      exitLocalRay.z = -exitLocalRay.z;
      vec3 exitLocalN = vec3(hitLocalN.x, hitLocalN.y, -hitLocalN.z);
      float cosOut = clamp(dot(exitLocalRay, exitLocalN), 0.0, 1.0);
      float exitStep = mix(0.28, 0.06, smoothstep(0.15, 0.65, cosOut));
      vec3 exitLocal = vec3(hitLocalPos.x, hitLocalPos.y, -hitLocalPos.z) + exitLocalRay * hitEffectiveR * exitStep;
      float left = sdMouth(exitLocal, hitMouth.shape, hitMouth.radius, hitMouth.scale);
      if (left < hitEpsBest) {
        vec3 nExit = sdfMouthNormal(exitLocal, hitMouth.shape, hitMouth.radius, hitMouth.scale);
        exitLocal += nExit * (hitEpsBest - left + hitEffectiveR * 0.04);
      }

      MouthParams destMouth = fetchMouth(hitMouth.destMouthIdx);
      currentUniverse = hitMouth.destUniverseIdx;
      position = destMouth.center + quatRotate(destMouth.quat, exitLocal);
      ray = normalize(quatRotate(destMouth.quat, exitLocalRay));

      lastExitedMouth = hitMouth.destMouthIdx;
      exitGuardSteps = 2;
      activeU = fetchUniverse(currentUniverse);
    } else {
      if (exitGuardSteps > 0) exitGuardSteps -= 1;

      // Velocity Verlet kick, then evaluate second-half multi-body acceleration
      vec3 velocity = ray * LIGHT_SPEED + 0.5 * accel1 * dt;
      position += velocity * dt;

      vec3 accel2 = vec3(0.0);
      for (int k = 0; k < MAX_MOUTHS_PER_UNIVERSE; k++) {
        if (k >= localMouthCount) break;
        int mIdx = activeU.mouthOffset + k;
        MouthParams m = fetchMouth(mIdx);

        float gap2;
        vec3 n2;
        vec3 lp2;
        vec3 ln2;
        float d2;
        float effR2;
        float maxExt2;
        bool exact2;

        evalMouthField(
          m,
          position,
          allowFar,
          gap2,
          n2,
          lp2,
          ln2,
          d2,
          effR2,
          maxExt2,
          exact2
        );

        float shell2 = effR2 * 0.1;
        float gravityScale2 = m.radius * LIGHT_SPEED * LIGHT_SPEED;
        float denom2 = gap2 + m.radius * m.gravityRatio;
        float safe2 = max(denom2, shell2);
        float amount2 = m.gravityRatio / safe2;
        accel2 += -n2 * gravityScale2 * amount2 * amount2;
      }

      velocity += 0.5 * accel2 * dt;
      float speed = length(velocity);
      if (speed > 1e-5) ray = velocity / speed;
    }
  }

  if (uDiagnosticMode == 0) {
    vec3 color = sampleUniverse(ray, currentUniverse);
    if (hitBounceCap) {
      color *= 0.12;
    }
    fragColor = vec4(color, 1.0);
    return;
  }

  // Diagnostic Caustic & Topology Mode (uniform control flow for dFdx/dFdy)
  int safeFinalU = clamp(currentUniverse, 0, max(uUniverseCount - 1, 0));
  UniverseParams finalU = fetchUniverse(safeFinalU);
  float cosLat;
  vec2 diagUv = directionToEquirectDiagnostic(ray, finalU.rot, cosLat);

  // Unwrap screen-space longitude derivatives across the u = 0 <-> 1 seam
  // (all derivatives evaluated unconditionally in uniform control flow)
  float du_dx = dFdx(diagUv.x);
  du_dx -= round(du_dx);
  float du_dy = dFdy(diagUv.x);
  du_dy -= round(du_dy);
  float dv_dx = dFdx(diagUv.y);
  float dv_dy = dFdy(diagUv.y);
  vec3 dRay_dx = dFdx(ray);
  vec3 dRay_dy = dFdy(ray);

  // 2x2 Jacobian determinant of equirectangular (u, v) w.r.t. screen pixels (x, y)
  float detJ = du_dx * dv_dy - du_dy * dv_dx;

  // Unlensed baseline pixel solid angle & latitude-adjusted baseline Jacobian
  float pixelAng = (2.0 * TAN_HALF_FOV) / max(uResolution.y, 1.0);
  float camRayLen2 = dot(camRayRaw, camRayRaw);
  float baselineSolidAngle = (pixelAng * pixelAng) / (camRayLen2 * sqrt(camRayLen2));
  float baselineDetJ = baselineSolidAngle / max(2.0 * PI * PI * cosLat, 1e-6);

  // Near the exact north/south equirectangular poles (within ~1 deg), fall back to the
  // algebraically equivalent spherical triple product (sign-matched to (u, v)) so polar
  // atan() wrap cannot spuriously zero detJ or flip parity.
  float sphDet = -dot(ray, cross(dRay_dx, dRay_dy));
  float signedRatio = cosLat < 0.02
    ? sphDet / max(baselineSolidAngle, 1e-12)
    : detJ / max(baselineDetJ, 1e-12);

  // Normalized fold metric: 0.0 right on caustic folds (det(J) = 0),
  // ~0.8 in unwarped space (|signedRatio| = 1.0), approaching 1.0 under demagnification.
  float absRatio = abs(signedRatio);
  float foldMetric = clamp(absRatio / (absRatio + 0.25), 0.0, 1.0);

  // Encoded topological state: currentUniverse + crossings / max(uMaxCrossings, 1), or -1.0 if bounce-capped
  float topoState = hitBounceCap
    ? -1.0
    : float(currentUniverse) + float(crossings) / float(maxCrossings);

  if (uDiagnosticMode == 2) {
    fragColor = vec4(diagUv.x, diagUv.y, topoState, foldMetric);
    return;
  }

  // uDiagnosticMode == 1: Human-readable topological phase regions + caustic fold curves
  vec3 regionCol = diagnosticRegionColor(currentUniverse, crossings, hitBounceCap);

  float topoJump = fwidth(topoState);
  float detSign = signedRatio >= 0.0 ? 1.0 : -1.0;
  float detSignJump = fwidth(detSign);

  bool isTopoSeam = (topoJump > 1e-4) || hitBounceCap;
  float parityEdge = (!isTopoSeam && detSignJump > 0.5) ? 1.0 : 0.0;

  float foldGlow = isTopoSeam ? 0.0 : (1.0 - smoothstep(0.0, 0.08, foldMetric));
  float foldCore = isTopoSeam ? 0.0 : max(1.0 - smoothstep(0.0, 0.025, foldMetric), parityEdge * 0.85);

  vec3 diagCol = regionCol;
  if (!hitBounceCap && topoJump > 1e-4) {
    diagCol *= 0.65;
  }
  diagCol = mix(diagCol, vec3(1.0, 0.72, 0.18), foldGlow * 0.85);
  diagCol = mix(diagCol, vec3(1.0, 0.99, 0.92), foldCore);

  fragColor = vec4(diagCol, 1.0);
}
`;
