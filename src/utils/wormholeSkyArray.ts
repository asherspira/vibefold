import * as THREE from 'three';

/**
 * Precomputed 0..255 sRGB byte -> IEEE 754 binary16 (HalfFloat) LUT so
 * LDR images pack into the RGBA16F DataArrayTexture in milliseconds.
 */
const UINT8_TO_HALF_LUT = (() => {
  const lut = new Uint16Array(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = THREE.DataUtils.toHalfFloat(i / 255);
  }
  return lut;
})();

const HALF_ONE = THREE.DataUtils.toHalfFloat(1.0);

function configureSkyArrayTexture(texture: THREE.DataArrayTexture): THREE.DataArrayTexture {
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.HalfFloatType;
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}

function createDummySkyArrayTexture(): THREE.DataArrayTexture {
  const data = new Uint16Array([0, 0, 0, HALF_ONE]);
  return configureSkyArrayTexture(new THREE.DataArrayTexture(data, 1, 1, 1));
}

/**
 * Resamples an HTMLImageElement / CanvasImageSource onto a targetW x targetH
 * HalfFloat RGBA layer stored bottom-up (y=0 -> v=0 nadir, y=H-1 -> v=1 zenith)
 * to match DataArrayTexture's flipY=false WebGL2 convention.
 */
export function packImageToHalfFloatLayer(
  image: CanvasImageSource,
  targetW: number,
  targetH: number
): Uint16Array {
  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const out = new Uint16Array(targetW * targetH * 4);
  if (!ctx) return out;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, 0, 0, targetW, targetH);

  const src = ctx.getImageData(0, 0, targetW, targetH).data;
  const rowFloats = targetW * 4;

  // Flip vertically: canvas row 0 is image top (v=1), DataArrayTexture row 0 is bottom (v=0).
  for (let y = 0; y < targetH; y++) {
    const srcRow = (targetH - 1 - y) * rowFloats;
    const dstRow = y * rowFloats;
    for (let i = 0; i < rowFloats; i += 4) {
      out[dstRow + i] = UINT8_TO_HALF_LUT[src[srcRow + i]];
      out[dstRow + i + 1] = UINT8_TO_HALF_LUT[src[srcRow + i + 1]];
      out[dstRow + i + 2] = UINT8_TO_HALF_LUT[src[srcRow + i + 2]];
      out[dstRow + i + 3] = HALF_ONE;
    }
  }

  return out;
}

/**
 * Resamples a loaded HDRI DataTexture (HalfFloatType or FloatType from HDRLoader/EXRLoader)
 * into a targetW x targetH RGBA16F layer stored bottom-up (v=0 nadir, v=1 zenith).
 * Honors texture.flipY so both .hdr (flipY=true) and .exr (flipY=false) stay right-side-up.
 */
export function packDataTextureToHalfFloatLayer(
  texture: THREE.DataTexture,
  targetW: number,
  targetH: number
): Uint16Array {
  const img = texture.image as {
    data?: Uint16Array | Float32Array;
    width: number;
    height: number;
  };
  const out = new Uint16Array(targetW * targetH * 4);
  if (!img || !img.data || img.width <= 0 || img.height <= 0) {
    return out;
  }

  const srcW = img.width;
  const srcH = img.height;
  const src = img.data;
  const isHalf = texture.type === THREE.HalfFloatType && src instanceof Uint16Array;
  const isFloat = texture.type === THREE.FloatType && src instanceof Float32Array;
  if (!isHalf && !isFloat) {
    return out;
  }

  const flipY = texture.flipY;

  // Fast path: exact dimensions and already HalfFloat
  if (isHalf && srcW === targetW && srcH === targetH) {
    if (!flipY) {
      out.set(src);
      return out;
    }
    const rowStride = targetW * 4;
    for (let y = 0; y < targetH; y++) {
      const srcRow = (targetH - 1 - y) * rowStride;
      const dstRow = y * rowStride;
      out.set(src.subarray(srcRow, srcRow + rowStride), dstRow);
    }
    return out;
  }

  const readChannel = (idx: number): number =>
    isHalf ? THREE.DataUtils.fromHalfFloat(src[idx]) : src[idx];

  // Bilinear resample with horizontal wraparound across the equirectangular 360° seam.
  for (let y = 0; y < targetH; y++) {
    const vNorm = (y + 0.5) / targetH;
    const srcV = flipY ? 1.0 - vNorm : vNorm;
    const fy = Math.max(0, Math.min(srcH - 1, srcV * srcH - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(srcH - 1, y0 + 1);
    const wy = fy - y0;
    const wy0 = 1 - wy;

    const row0 = y0 * srcW;
    const row1 = y1 * srcW;
    const dstRow = y * targetW;

    for (let x = 0; x < targetW; x++) {
      const uNorm = (x + 0.5) / targetW;
      const fx = uNorm * srcW - 0.5;
      const x0Raw = Math.floor(fx);
      const wx = fx - x0Raw;
      const wx0 = 1 - wx;
      const x0 = ((x0Raw % srcW) + srcW) % srcW;
      const x1 = (x0 + 1) % srcW;

      const i00 = (row0 + x0) * 4;
      const i10 = (row0 + x1) * 4;
      const i01 = (row1 + x0) * 4;
      const i11 = (row1 + x1) * 4;

      const w00 = wx0 * wy0;
      const w10 = wx * wy0;
      const w01 = wx0 * wy;
      const w11 = wx * wy;

      const r =
        readChannel(i00) * w00 +
        readChannel(i10) * w10 +
        readChannel(i01) * w01 +
        readChannel(i11) * w11;
      const g =
        readChannel(i00 + 1) * w00 +
        readChannel(i10 + 1) * w10 +
        readChannel(i01 + 1) * w01 +
        readChannel(i11 + 1) * w11;
      const b =
        readChannel(i00 + 2) * w00 +
        readChannel(i10 + 2) * w10 +
        readChannel(i01 + 2) * w01 +
        readChannel(i11 + 2) * w11;

      const dstIdx = (dstRow + x) * 4;
      out[dstIdx] = THREE.DataUtils.toHalfFloat(r);
      out[dstIdx + 1] = THREE.DataUtils.toHalfFloat(g);
      out[dstIdx + 2] = THREE.DataUtils.toHalfFloat(b);
      out[dstIdx + 3] = HALF_ONE;
    }
  }

  return out;
}

/**
 * Manages a WebGL2 sampler2DArray (THREE.DataArrayTexture, RGBA16F) holding an
 * arbitrary number of equirectangular universe skies (both LDR images and linear HDRIs).
 *
 * Universes using procedural grids do not allocate a layer in this array.
 */
export class SkyArrayManager {
  private readonly layerWidth: number;
  private readonly layerHeight: number;
  private readonly layers = new Map<string, Uint16Array>();
  private order: string[] = [];
  private texture: THREE.DataArrayTexture;

  constructor(layerWidth: number, layerHeight: number) {
    this.layerWidth = Math.max(64, Math.round(layerWidth));
    this.layerHeight = Math.max(32, Math.round(layerHeight));
    this.texture = createDummySkyArrayTexture();
  }

  getLayerWidth(): number {
    return this.layerWidth;
  }

  getLayerHeight(): number {
    return this.layerHeight;
  }

  getLayerCount(): number {
    return this.order.length;
  }

  getLayerIndex(universeId: string): number {
    return this.order.indexOf(universeId);
  }

  getTexture(): THREE.DataArrayTexture {
    return this.texture;
  }

  setLayer(universeId: string, pixels: Uint16Array): number {
    const expectedLen = this.layerWidth * this.layerHeight * 4;
    if (pixels.length !== expectedLen) {
      throw new Error(
        `Sky layer length mismatch for "${universeId}": expected ${expectedLen}, got ${pixels.length}.`
      );
    }

    const existingIdx = this.order.indexOf(universeId);
    this.layers.set(universeId, pixels);

    if (existingIdx >= 0 && this.texture.image.depth === this.order.length) {
      // In-place single-layer update without reallocating the GPU texture array
      const arrayData = this.texture.image.data as Uint16Array;
      arrayData.set(pixels, existingIdx * expectedLen);
      this.texture.addLayerUpdate(existingIdx);
      this.texture.needsUpdate = true;
      return existingIdx;
    }

    if (existingIdx < 0) {
      this.order.push(universeId);
    }
    this.rebuildTexture();
    return this.order.indexOf(universeId);
  }

  removeLayer(universeId: string): void {
    const idx = this.order.indexOf(universeId);
    if (idx < 0) return;
    this.layers.delete(universeId);
    this.order.splice(idx, 1);
    this.rebuildTexture();
  }

  private rebuildTexture(): void {
    const previous = this.texture;
    const depth = this.order.length;

    if (depth === 0) {
      this.texture = createDummySkyArrayTexture();
      previous.dispose();
      return;
    }

    const layerStride = this.layerWidth * this.layerHeight * 4;
    const combined = new Uint16Array(layerStride * depth);
    for (let i = 0; i < depth; i++) {
      const slice = this.layers.get(this.order[i]);
      if (slice) {
        combined.set(slice, i * layerStride);
      }
    }

    this.texture = configureSkyArrayTexture(
      new THREE.DataArrayTexture(combined, this.layerWidth, this.layerHeight, depth)
    );
    previous.dispose();
  }

  dispose(): void {
    this.layers.clear();
    this.order = [];
    this.texture.dispose();
  }
}
