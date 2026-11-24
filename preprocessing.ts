import type { QualityIssue, QualityReport } from "./types";

export const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
export const MAX_FILE_MB = 10;

export interface FileCheck {
  ok: boolean;
  error?: string;
}

/** Checks type, size and the real file signature (not only the extension). */
export async function validateFile(file: File): Promise<FileCheck> {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return { ok: false, error: "Unsupported file format. Please choose a JPG, JPEG or PNG photograph." };
  }
  if (file.size === 0) return { ok: false, error: "This file is empty. Please choose another photograph." };
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    return { ok: false, error: `This file is too large (maximum ${MAX_FILE_MB} MB). Please choose a smaller photograph.` };
  }
  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  const isJpg = head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff;
  const isPng = head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4e && head[3] === 0x47;
  if (!isJpg && !isPng) {
    return { ok: false, error: "This file does not look like a real JPG or PNG image." };
  }
  return { ok: true };
}

/** Decodes an image from an object URL. Browsers apply the EXIF rotation of phone photos automatically. */
export function loadImageElement(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The image could not be read. It may be corrupted."));
    img.src = url;
  });
}

/** Draws the image onto a square canvas of the size the model expects (same "squash" resize as in training). */
export function imageToCanvas(img: HTMLImageElement, size: number): HTMLCanvasElement {
  // Step-wise halving gives smoother down-scaling (closer to the resize used while preparing training data).
  let w = img.naturalWidth;
  let h = img.naturalHeight;
  let source: CanvasImageSource = img;
  while (w / 2 > size * 1.5 || h / 2 > size * 1.5) {
    w = Math.max(size, Math.floor(w / 2));
    h = Math.max(size, Math.floor(h / 2));
    const step = document.createElement("canvas");
    step.width = w;
    step.height = h;
    const sctx = step.getContext("2d")!;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(source, 0, 0, w, h);
    source = step;
  }
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, size, size);
  return canvas;
}

/**
 * Simple quality checks on a small grayscale copy of the photo:
 *  - brightness (mean grey value)       -> too dark / overexposed
 *  - share of saturated white pixels     -> overexposed
 *  - contrast (std. deviation)           -> flat / uniform images
 *  - variance of the Laplacian           -> blur
 */
export function assessQuality(img: HTMLImageElement): QualityReport {
  const issues: QualityIssue[] = [];
  const minSide = Math.min(img.naturalWidth, img.naturalHeight);
  if (minSide < 100) {
    issues.push({
      code: "too_small",
      severity: "blocking",
      message: "The photograph is too small (under 100 pixels). Please use a larger photograph.",
    });
  }

  const S = 256;
  const c = document.createElement("canvas");
  c.width = S;
  c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, S, S);
  const { data } = ctx.getImageData(0, 0, S, S);
  const gray = new Float32Array(S * S);
  let sum = 0;
  let bright = 0;
  for (let i = 0; i < S * S; i++) {
    const g = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    gray[i] = g;
    sum += g;
    if (g > 245) bright++;
  }
  const mean = sum / (S * S);
  let varSum = 0;
  for (let i = 0; i < gray.length; i++) varSum += (gray[i] - mean) ** 2;
  const std = Math.sqrt(varSum / gray.length);

  // Laplacian variance (higher = sharper)
  let lsum = 0;
  let lsq = 0;
  let n = 0;
  for (let y = 1; y < S - 1; y++) {
    for (let x = 1; x < S - 1; x++) {
      const i = y * S + x;
      const l = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - S] - gray[i + S];
      lsum += l;
      lsq += l * l;
      n++;
    }
  }
  const sharpness = lsq / n - (lsum / n) ** 2;

  const brightShare = bright / (S * S);
  if (mean < 20) {
    issues.push({
      code: "too_dark",
      severity: "blocking",
      message: "The photograph is extremely dark, so the skin cannot be seen. Please retake it in better lighting.",
    });
  } else if (brightShare > 0.85) {
    issues.push({
      code: "overexposed",
      severity: "blocking",
      message: "The photograph is almost completely white (overexposed). Avoid direct flash or harsh light and retake it.",
    });
  } else if (std < 8) {
    issues.push({
      code: "low_contrast",
      severity: "blocking",
      message: "The photograph looks almost uniform, so there is nothing for the model to analyse. Please retake it.",
    });
  } else {
    if (mean < 45) {
      issues.push({
        code: "too_dark",
        severity: "warning",
        message: "The photograph is very dark. Move to better lighting and retake it for a more reliable result.",
      });
    } else if (mean > 225 || brightShare > 0.4) {
      issues.push({
        code: "overexposed",
        severity: "warning",
        message: "The photograph is overexposed (too bright). Avoid direct flash or harsh light and retake it.",
      });
    }
    if (sharpness < 25) {
      issues.push({
        code: "blurry",
        severity: "warning",
        message: "The photograph looks blurry. Hold the phone steady, focus on the skin and retake it.",
      });
    }
  }

  return {
    issues,
    blocking: issues.some((i) => i.severity === "blocking"),
    brightness: mean,
    sharpness,
  };
}
