import type * as TF from "@tensorflow/tfjs";
import type { ModelInfo } from "./types";

// Files live in /public/models/skin and are copied into the production build.
// Absolute paths work on Netlify (and locally) because the site is served from the domain root.
export const MODEL_URL = "/models/skin/model.json";
export const MODEL_INFO_URL = "/models/skin/model_info.json";

export class ModelUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelUnavailableError";
  }
}

type TFModule = typeof TF;
let tfPromise: Promise<TFModule> | null = null;
let modelPromise: Promise<{ tf: TFModule; model: TF.LayersModel; info: ModelInfo }> | null = null;

async function getTf(): Promise<TFModule> {
  if (!tfPromise) {
    tfPromise = (async () => {
      const tf = await import("@tensorflow/tfjs");
      try {
        await tf.setBackend("webgl");
      } catch {
        /* fall through to cpu */
      }
      if (tf.getBackend() !== "webgl") await tf.setBackend("cpu");
      await tf.ready();
      return tf;
    })();
  }
  return tfPromise;
}

/** Reads the small metadata file that is written together with the trained model. */
export async function fetchModelInfo(): Promise<ModelInfo | null> {
  try {
    const res = await fetch(MODEL_INFO_URL, { cache: "no-cache" });
    if (!res.ok) return null;
    const info = (await res.json()) as ModelInfo;
    return info.trained ? info : null;
  } catch {
    return null;
  }
}

/** Checks the manifest and its weight files before the user starts an analysis. */
export async function checkModelAssets(): Promise<boolean> {
  try {
    const response = await fetch(MODEL_URL, { cache: "no-cache" });
    if (!response.ok) return false;
    const manifest = (await response.json()) as {
      weightsManifest?: Array<{ paths?: string[] }>;
    };
    const paths = manifest.weightsManifest?.flatMap((group) => group.paths ?? []) ?? [];
    if (paths.length === 0) return false;

    const checks = await Promise.all(
      paths.map(async (path) => {
        const assetUrl = new URL(path, new URL(MODEL_URL, window.location.origin));
        const asset = await fetch(assetUrl, { method: "HEAD", cache: "no-cache" });
        return asset.ok;
      }),
    );
    return checks.every(Boolean);
  } catch {
    return false;
  }
}

/** Loads TensorFlow.js and the trained model once, then re-uses it. */
export function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const info = await fetchModelInfo();
      if (!info) {
        throw new ModelUnavailableError(
          "No trained model was found on this website (models/skin/model_info.json is missing).",
        );
      }
      try {
        const tf = await getTf();
        const model = await tf.loadLayersModel(MODEL_URL);
        const outputShape = model.outputs[0]?.shape;
        if (outputShape?.at(-1) !== info.classes.length) {
          model.dispose();
          throw new Error("The model output does not match the categories in model_info.json.");
        }
        // warm-up run so the first real prediction is fast
        tf.tidy(() => {
          const out = model.predict(tf.zeros([1, info.inputSize, info.inputSize, 3])) as TF.Tensor;
          out.dataSync();
        });
        return { tf, model, info };
      } catch (e) {
        throw new ModelUnavailableError(
          `The AI model could not be loaded (${e instanceof Error ? e.message : "unknown error"}).`,
        );
      }
    })();
    modelPromise.catch(() => {
      modelPromise = null; // allow a retry
    });
  }
  return modelPromise;
}
