import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4ArrayBufferTarget } from 'mp4-muxer';
import { Muxer as WebmMuxer, ArrayBufferTarget as WebmArrayBufferTarget } from 'webm-muxer';
import type { MovieFormat } from './wormholeAnimation';

export interface RenderMovieOptions {
  /** Source WebGL canvas containing ONLY the rendered wormhole (no UI DOM elements). */
  sourceCanvas: HTMLCanvasElement;
  /** Exact even pixel width of the output video. */
  width: number;
  /** Exact even pixel height of the output video. */
  height: number;
  /** Total animation duration in seconds. */
  durationSeconds: number;
  /** Target frames per second (default: 30). */
  fps?: number;
  /** Requested output container format. */
  format: MovieFormat;
  /**
   * Synchronously updates camera/shader uniforms for normalized time t in [0, 1]
   * and executes `renderer.render(...)` into `sourceCanvas`.
   */
  renderFrameAt: (tNorm: number, frameIndex: number, totalFrames: number) => void;
  /** Progress callback invoked after each rendered frame. */
  onProgress?: (renderedFrame: number, totalFrames: number) => void;
  /** AbortSignal to cleanly cancel rendering mid-flight. */
  signal?: AbortSignal;
}

export interface RenderedMovieResult {
  blob: Blob;
  format: MovieFormat;
  mimeType: string;
  extension: 'mp4' | 'webm';
  totalFrames: number;
  fps: number;
  width: number;
  height: number;
  usedFallbackRecorder: boolean;
}

function yieldToBrowser(delayMs = 0): Promise<void> {
  if (delayMs > 0 || document.hidden) {
    return new Promise((resolve) => window.setTimeout(resolve, delayMs));
  }
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

function computeBitrate(width: number, height: number, fps: number): number {
  // ~0.22 bits per pixel at 30fps yields sharp grid lines and smooth gradients (~6–12 Mbps at 1080p)
  const pixelsPerSecond = width * height * fps;
  return Math.round(Math.min(16_000_000, Math.max(2_500_000, pixelsPerSecond * 0.22)));
}

interface ResolvedWebCodecsConfig {
  encoderConfig: VideoEncoderConfig;
  muxerCodec: string;
}

async function resolveWebCodecsConfig(
  format: MovieFormat,
  width: number,
  height: number,
  fps: number,
  bitrate: number
): Promise<ResolvedWebCodecsConfig | null> {
  if (typeof VideoEncoder === 'undefined' || typeof VideoFrame === 'undefined') {
    return null;
  }

  const candidates: Array<{
    codec: string;
    muxerCodec: string;
    avc?: { format: 'avc' | 'annexb' };
  }> =
    format === 'mp4'
      ? [
          { codec: 'avc1.640032', muxerCodec: 'avc', avc: { format: 'avc' } },
          { codec: 'avc1.640028', muxerCodec: 'avc', avc: { format: 'avc' } },
          { codec: 'avc1.4d002a', muxerCodec: 'avc', avc: { format: 'avc' } },
          { codec: 'avc1.42001f', muxerCodec: 'avc', avc: { format: 'avc' } },
        ]
      : [
          { codec: 'vp09.00.10.08', muxerCodec: 'V_VP9' },
          { codec: 'vp8', muxerCodec: 'V_VP8' },
        ];

  for (const candidate of candidates) {
    const config: VideoEncoderConfig = {
      codec: candidate.codec,
      width,
      height,
      bitrate,
      framerate: fps,
      ...(candidate.avc ? { avc: candidate.avc } : {}),
    };
    try {
      const support = await VideoEncoder.isConfigSupported(config);
      if (support.supported) {
        return {
          encoderConfig: support.config ?? config,
          muxerCodec: candidate.muxerCodec,
        };
      }
    } catch {
      // Try next codec candidate
    }
  }

  return null;
}

async function renderViaWebCodecs(
  options: RenderMovieOptions,
  stagingCanvas: HTMLCanvasElement,
  stagingCtx: CanvasRenderingContext2D,
  fps: number,
  totalFrames: number,
  resolved: ResolvedWebCodecsConfig
): Promise<RenderedMovieResult> {
  const { sourceCanvas, width, height, format, renderFrameAt, onProgress, signal } = options;

  let encoderError: Error | null = null;
  let mp4Muxer: Mp4Muxer<Mp4ArrayBufferTarget> | null = null;
  let webmMuxer: WebmMuxer<WebmArrayBufferTarget> | null = null;
  let mp4Target: Mp4ArrayBufferTarget | null = null;
  let webmTarget: WebmArrayBufferTarget | null = null;

  if (format === 'mp4') {
    mp4Target = new Mp4ArrayBufferTarget();
    mp4Muxer = new Mp4Muxer({
      target: mp4Target,
      video: {
        codec: 'avc',
        width,
        height,
        frameRate: fps,
      },
      fastStart: 'in-memory',
      firstTimestampBehavior: 'offset',
    });
  } else {
    webmTarget = new WebmArrayBufferTarget();
    webmMuxer = new WebmMuxer({
      target: webmTarget,
      video: {
        codec: resolved.muxerCodec,
        width,
        height,
        frameRate: fps,
      },
      firstTimestampBehavior: 'offset',
    });
  }

  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      if (mp4Muxer) {
        mp4Muxer.addVideoChunk(chunk, meta);
      } else if (webmMuxer) {
        webmMuxer.addVideoChunk(chunk, meta);
      }
    },
    error: (err) => {
      encoderError = err instanceof Error ? err : new Error(String(err));
    },
  });

  encoder.configure(resolved.encoderConfig);

  const frameDurationUs = Math.round(1_000_000 / fps);

  try {
    for (let i = 0; i < totalFrames; i++) {
      if (signal?.aborted) {
        throw new DOMException('Render cancelled', 'AbortError');
      }
      if (encoderError) {
        throw encoderError;
      }

      // Backpressure guard so long renders never overwhelm the encoder queue
      while (encoder.encodeQueueSize > 6) {
        if (signal?.aborted) {
          throw new DOMException('Render cancelled', 'AbortError');
        }
        if (encoderError) {
          throw encoderError;
        }
        await yieldToBrowser(12);
      }

      const tNorm = totalFrames <= 1 ? 1 : i / (totalFrames - 1);
      renderFrameAt(tNorm, i, totalFrames);

      // Copy synchronously right after WebGL render so the drawing buffer is captured intact
      stagingCtx.drawImage(sourceCanvas, 0, 0, width, height);

      const timestampUs = Math.round((i * 1_000_000) / fps);
      const frame = new VideoFrame(stagingCanvas, {
        timestamp: timestampUs,
        duration: frameDurationUs,
      });

      const keyFrame = i === 0 || i % fps === 0;
      encoder.encode(frame, { keyFrame });
      frame.close();

      onProgress?.(i + 1, totalFrames);

      // Yield a paint tick so the user sees live frame-by-frame progress and can click Cancel
      await yieldToBrowser(0);
    }

    if (signal?.aborted) {
      throw new DOMException('Render cancelled', 'AbortError');
    }

    await encoder.flush();
    if (encoderError) {
      throw encoderError;
    }

    if (mp4Muxer && mp4Target) {
      mp4Muxer.finalize();
      return {
        blob: new Blob([mp4Target.buffer], { type: 'video/mp4' }),
        format: 'mp4',
        mimeType: 'video/mp4',
        extension: 'mp4',
        totalFrames,
        fps,
        width,
        height,
        usedFallbackRecorder: false,
      };
    }

    if (webmMuxer && webmTarget) {
      webmMuxer.finalize();
      return {
        blob: new Blob([webmTarget.buffer], { type: 'video/webm' }),
        format: 'webm',
        mimeType: 'video/webm',
        extension: 'webm',
        totalFrames,
        fps,
        width,
        height,
        usedFallbackRecorder: false,
      };
    }

    throw new Error('Muxer failed to initialize.');
  } finally {
    if (encoder.state !== 'closed') {
      try {
        encoder.close();
      } catch {
        // Ignore close errors during abort
      }
    }
  }
}

async function renderViaMediaRecorderFallback(
  options: RenderMovieOptions,
  stagingCanvas: HTMLCanvasElement,
  stagingCtx: CanvasRenderingContext2D,
  fps: number,
  totalFrames: number,
  bitrate: number
): Promise<RenderedMovieResult> {
  const { sourceCanvas, width, height, format, renderFrameAt, onProgress, signal } = options;

  if (typeof MediaRecorder === 'undefined' || typeof stagingCanvas.captureStream !== 'function') {
    throw new Error('Video encoding is not supported in this browser.');
  }

  const mimeCandidates: Array<{ mime: string; ext: 'mp4' | 'webm' }> =
    format === 'mp4'
      ? [
          { mime: 'video/mp4;codecs=avc1.42E01E', ext: 'mp4' },
          { mime: 'video/mp4', ext: 'mp4' },
          { mime: 'video/webm;codecs=vp9', ext: 'webm' },
          { mime: 'video/webm;codecs=vp8', ext: 'webm' },
          { mime: 'video/webm', ext: 'webm' },
        ]
      : [
          { mime: 'video/webm;codecs=vp9', ext: 'webm' },
          { mime: 'video/webm;codecs=vp8', ext: 'webm' },
          { mime: 'video/webm', ext: 'webm' },
          { mime: 'video/mp4', ext: 'mp4' },
        ];

  const picked = mimeCandidates.find((c) => MediaRecorder.isTypeSupported(c.mime));
  if (!picked) {
    throw new Error('No supported video MediaRecorder MIME type found in this browser.');
  }

  // captureStream(0) lets us manually push each rendered frame via requestFrame()
  const stream = stagingCanvas.captureStream(0);
  const videoTrack = stream.getVideoTracks()[0] as
    | (MediaStreamTrack & { requestFrame?: () => void })
    | undefined;

  const chunks: BlobPart[] = [];
  const recorder = new MediaRecorder(stream, {
    mimeType: picked.mime,
    videoBitsPerSecond: bitrate,
  });

  const stoppedPromise = new Promise<void>((resolve, reject) => {
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };
    recorder.onerror = () => reject(new Error('MediaRecorder encountered an encoding error.'));
    recorder.onstop = () => resolve();
  });

  recorder.start();
  const frameIntervalMs = Math.max(16, Math.round(1000 / fps));

  try {
    for (let i = 0; i < totalFrames; i++) {
      if (signal?.aborted) {
        throw new DOMException('Render cancelled', 'AbortError');
      }

      const tNorm = totalFrames <= 1 ? 1 : i / (totalFrames - 1);
      renderFrameAt(tNorm, i, totalFrames);
      stagingCtx.drawImage(sourceCanvas, 0, 0, width, height);
      videoTrack?.requestFrame?.();

      onProgress?.(i + 1, totalFrames);
      await yieldToBrowser(frameIntervalMs);
    }

    if (recorder.state !== 'inactive') {
      recorder.stop();
    }
    await stoppedPromise;

    return {
      blob: new Blob(chunks, { type: picked.mime }),
      format: picked.ext,
      mimeType: picked.mime,
      extension: picked.ext,
      totalFrames,
      fps,
      width,
      height,
      usedFallbackRecorder: true,
    };
  } finally {
    if (recorder.state !== 'inactive') {
      try {
        recorder.stop();
      } catch {
        // Ignore
      }
    }
    stream.getTracks().forEach((track) => track.stop());
  }
}

/**
 * Renders the wormhole animation deterministically one frame at a time into a
 * downloadable video file containing ONLY the rendered WebGL scene (no UI).
 */
export async function renderWormholeMovie(
  options: RenderMovieOptions
): Promise<RenderedMovieResult> {
  const fps = Math.max(12, Math.min(60, Math.round(options.fps ?? 30)));
  const safeDuration = Math.max(0.5, Math.min(120, options.durationSeconds));
  const totalFrames = Math.max(2, Math.round(safeDuration * fps));
  const bitrate = computeBitrate(options.width, options.height, fps);

  const stagingCanvas = document.createElement('canvas');
  stagingCanvas.width = options.width;
  stagingCanvas.height = options.height;
  const stagingCtx = stagingCanvas.getContext('2d', { alpha: false });
  if (!stagingCtx) {
    throw new Error('Could not initialize 2D capture buffer for movie rendering.');
  }

  const webCodecsConfig = await resolveWebCodecsConfig(
    options.format,
    options.width,
    options.height,
    fps,
    bitrate
  );

  if (webCodecsConfig) {
    return renderViaWebCodecs(
      options,
      stagingCanvas,
      stagingCtx,
      fps,
      totalFrames,
      webCodecsConfig
    );
  }

  return renderViaMediaRecorderFallback(
    options,
    stagingCanvas,
    stagingCtx,
    fps,
    totalFrames,
    bitrate
  );
}

export function downloadVideoBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
