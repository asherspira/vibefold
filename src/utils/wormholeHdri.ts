import * as THREE from 'three';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';

export type SkyFileKind = 'hdr' | 'exr' | 'ldr';

/**
 * Identifies whether an uploaded file is a Radiance RGBE HDRI (.hdr/.pic),
 * an OpenEXR HDRI (.exr), or a standard LDR image (.png/.jpg/.webp/etc.).
 * Checks both filename extension and binary magic bytes.
 */
export async function detectSkyFileKind(file: File): Promise<SkyFileKind> {
  const lower = file.name.toLowerCase();
  if (lower.endsWith('.hdr') || lower.endsWith('.pic') || lower.endsWith('.rgbe')) {
    return 'hdr';
  }
  if (lower.endsWith('.exr')) {
    return 'exr';
  }

  // Inspect file header magic bytes as fallback for extensionless files
  try {
    const headerBuf = await file.slice(0, 16).arrayBuffer();
    const bytes = new Uint8Array(headerBuf);
    // OpenEXR magic: 0x76 0x2f 0x31 0x01
    if (bytes.length >= 4 && bytes[0] === 0x76 && bytes[1] === 0x2f && bytes[2] === 0x31 && bytes[3] === 0x01) {
      return 'exr';
    }
    // Radiance RGBE magic: "#?RADIANCE" or "#?RGBE"
    const ascii = String.fromCharCode(...bytes);
    if (ascii.startsWith('#?RADIANCE') || ascii.startsWith('#?RGBE')) {
      return 'hdr';
    }
  } catch {
    // Fall through to standard image loader
  }

  return 'ldr';
}

/**
 * Downsamples an oversized RGBA DataTexture (e.g., 8K/16K HDRI on a 4K-capped GPU)
 * by integer box-averaging so WebGL never fails the texture upload.
 */
function clampDataTextureSize(texture: THREE.DataTexture, maxTextureSize: number): THREE.DataTexture {
  const img = texture.image as { data: Uint16Array | Float32Array; width: number; height: number };
  if (!img || !img.data || (img.width <= maxTextureSize && img.height <= maxTextureSize)) {
    return texture;
  }

  const maxEdge = Math.max(img.width, img.height);
  const stride = Math.ceil(maxEdge / maxTextureSize);
  const newW = Math.max(1, Math.floor(img.width / stride));
  const newH = Math.max(1, Math.floor(img.height / stride));
  const isHalf = texture.type === THREE.HalfFloatType && img.data instanceof Uint16Array;
  const isFloat = texture.type === THREE.FloatType && img.data instanceof Float32Array;
  if (texture.format !== THREE.RGBAFormat || (!isHalf && !isFloat)) {
    return texture;
  }

  const outData = isHalf ? new Uint16Array(newW * newH * 4) : new Float32Array(newW * newH * 4);
  const src = img.data;
  const srcW = img.width;
  const invCount = 1 / (stride * stride);

  for (let y = 0; y < newH; y++) {
    const sy0 = y * stride;
    for (let x = 0; x < newW; x++) {
      const sx0 = x * stride;
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let dy = 0; dy < stride; dy++) {
        const rowOffset = (sy0 + dy) * srcW;
        for (let dx = 0; dx < stride; dx++) {
          const idx = (rowOffset + sx0 + dx) * 4;
          if (isHalf) {
            r += THREE.DataUtils.fromHalfFloat(src[idx]);
            g += THREE.DataUtils.fromHalfFloat(src[idx + 1]);
            b += THREE.DataUtils.fromHalfFloat(src[idx + 2]);
            a += THREE.DataUtils.fromHalfFloat(src[idx + 3]);
          } else {
            r += src[idx];
            g += src[idx + 1];
            b += src[idx + 2];
            a += src[idx + 3];
          }
        }
      }
      const dstIdx = (y * newW + x) * 4;
      if (isHalf) {
        outData[dstIdx] = THREE.DataUtils.toHalfFloat(r * invCount);
        outData[dstIdx + 1] = THREE.DataUtils.toHalfFloat(g * invCount);
        outData[dstIdx + 2] = THREE.DataUtils.toHalfFloat(b * invCount);
        outData[dstIdx + 3] = THREE.DataUtils.toHalfFloat(a * invCount);
      } else {
        outData[dstIdx] = r * invCount;
        outData[dstIdx + 1] = g * invCount;
        outData[dstIdx + 2] = b * invCount;
        outData[dstIdx + 3] = a * invCount;
      }
    }
  }

  const resized = new THREE.DataTexture(outData, newW, newH, THREE.RGBAFormat, texture.type);
  resized.colorSpace = texture.colorSpace;
  resized.flipY = texture.flipY;
  resized.wrapS = THREE.RepeatWrapping;
  resized.wrapT = THREE.ClampToEdgeWrapping;
  resized.minFilter = THREE.LinearFilter;
  resized.magFilter = THREE.LinearFilter;
  resized.generateMipmaps = false;
  resized.needsUpdate = true;

  texture.dispose();
  return resized;
}

/**
 * Loads a Radiance (.hdr/.pic) or OpenEXR (.exr) HDRI File into a WebGL-ready
 * equirectangular HalfFloat DataTexture (preserving the loader's flipY orientation).
 */
export async function loadHdriTexture(
  file: File,
  kind: 'hdr' | 'exr',
  maxTextureSize: number
): Promise<THREE.DataTexture> {
  const blobUrl = URL.createObjectURL(file);
  try {
    const loader = kind === 'hdr' ? new HDRLoader() : new EXRLoader();
    loader.setDataType(THREE.HalfFloatType);
    const texture = await loader.loadAsync(blobUrl);

    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;

    // Cap to GPU maxTextureSize (and 4096 max to keep VRAM & frame times snappy)
    const safeCap = Math.min(Math.max(maxTextureSize, 1024), 4096);
    return clampDataTextureSize(texture, safeCap);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}
