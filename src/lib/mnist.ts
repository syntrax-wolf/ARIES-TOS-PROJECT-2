export const RAW_SIZE = 28;
export const RAW_PIXELS = RAW_SIZE * RAW_SIZE;
export const FEATURE_SIZE = 14; // downsampled resolution used for training features
export const FEATURE_PIXELS = FEATURE_SIZE * FEATURE_SIZE;
export const NUM_CLASSES = 10;

export interface MnistRaw {
  images: Uint8Array; // count * RAW_PIXELS, row-major, one byte per pixel (0-255)
  labels: Uint8Array; // count
  count: number;
}

let cache: Promise<MnistRaw> | null = null;

export function loadMnistRaw(): Promise<MnistRaw> {
  if (!cache) {
    cache = Promise.all([
      fetch("/mnist/images28.bin").then((r) => r.arrayBuffer()),
      fetch("/mnist/labels.bin").then((r) => r.arrayBuffer()),
    ]).then(([imgBuf, lblBuf]) => {
      const labels = new Uint8Array(lblBuf);
      return { images: new Uint8Array(imgBuf), labels, count: labels.length };
    });
  }
  return cache;
}

export function getRawImage(raw: MnistRaw, index: number): Uint8Array {
  return raw.images.subarray(index * RAW_PIXELS, (index + 1) * RAW_PIXELS);
}

/** 2x2 block-average downsample from 28x28 to 14x14 — cuts feature count 4x for fast training. */
export function downsampleImage(image28: Uint8Array): Float32Array {
  const out = new Float32Array(FEATURE_PIXELS);
  for (let r = 0; r < FEATURE_SIZE; r++) {
    for (let c = 0; c < FEATURE_SIZE; c++) {
      const r0 = r * 2;
      const c0 = c * 2;
      const sum =
        image28[r0 * RAW_SIZE + c0] +
        image28[r0 * RAW_SIZE + c0 + 1] +
        image28[(r0 + 1) * RAW_SIZE + c0] +
        image28[(r0 + 1) * RAW_SIZE + c0 + 1];
      out[r * FEATURE_SIZE + c] = sum / 4;
    }
  }
  return out;
}

/** Renders a size x size grayscale pixel buffer (0-255 range) as a PNG data URL. */
export function pixelsToDataUrl(pixels: ArrayLike<number>, size: number, scale = 1): string {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = Math.max(0, Math.min(255, Math.round(pixels[i])));
    imageData.data[i * 4] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  if (scale === 1) return canvas.toDataURL();

  const scaled = document.createElement("canvas");
  scaled.width = size * scale;
  scaled.height = size * scale;
  const sctx = scaled.getContext("2d")!;
  sctx.imageSmoothingEnabled = false;
  sctx.drawImage(canvas, 0, 0, scaled.width, scaled.height);
  return scaled.toDataURL();
}

/** Average pixel-wise image across a set of raw 28x28 images — a blurry "prototype" digit. */
export function averageImage(rawImages: Uint8Array[]): Float32Array {
  const out = new Float32Array(RAW_PIXELS);
  if (rawImages.length === 0) return out;
  for (const img of rawImages) {
    for (let i = 0; i < RAW_PIXELS; i++) out[i] += img[i];
  }
  for (let i = 0; i < RAW_PIXELS; i++) out[i] /= rawImages.length;
  return out;
}
