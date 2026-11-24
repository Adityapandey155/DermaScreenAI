// Verifies that the exported TensorFlow.js model (the one the website loads) gives the same scores as Keras.
// Usage: node training/verify_tfjs.mjs <model_dir> <reference.json> <pixels_dir>
//   reference.json : Keras outputs for 3 test images (written by train.py / finetune.py)
//   pixels_dir     : ref_0.raw, ref_1.raw, ref_2.raw  (224x224x3 uint8 RGB, decoded with PIL)
import fs from "node:fs";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";

const [modelDir, refFile, pixDir] = process.argv.slice(2);
const json = JSON.parse(fs.readFileSync(path.join(modelDir, "model.json"), "utf8"));

const handler = {
  async load() {
    const specs = [];
    const bufs = [];
    for (const group of json.weightsManifest) {
      specs.push(...group.weights);
      for (const p of group.paths) bufs.push(fs.readFileSync(path.join(modelDir, p)));
    }
    const all = Buffer.concat(bufs);
    return {
      modelTopology: json.modelTopology,
      weightSpecs: specs,
      weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
      format: json.format,
      generatedBy: json.generatedBy,
      convertedBy: json.convertedBy,
    };
  },
};

await tf.setBackend("cpu");
const model = await tf.loadLayersModel(handler);
console.log("loaded model, input", JSON.stringify(model.inputs[0].shape), "output", JSON.stringify(model.outputs[0].shape));
const ref = JSON.parse(fs.readFileSync(refFile, "utf8"));
let worst = 0;
for (let i = 0; i < ref.length; i++) {
  const px = new Uint8Array(fs.readFileSync(path.join(pixDir, `ref_${i}.raw`)));
  const x = tf.tensor4d(Float32Array.from(px), [1, 224, 224, 3]); // raw 0..255, exactly like the website
  const out = Array.from(model.predict(x).dataSync());
  const diff = Math.max(...out.map((v, k) => Math.abs(v - ref[i].probs[k])));
  worst = Math.max(worst, diff);
  console.log(`image ${i}: max |tfjs - keras| = ${diff.toExponential(2)}  tfjs top=${out.indexOf(Math.max(...out))} keras top=${ref[i].probs.indexOf(Math.max(...ref[i].probs))}`);
}
console.log(worst < 0.02 ? "PASS: TF.js output matches Keras" : "FAIL: outputs differ");
process.exit(worst < 0.02 ? 0 : 1);
