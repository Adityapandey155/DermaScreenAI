"""
Step 2c - Export a trained Keras model (model.h5) to TensorFlow.js.

TensorFlow.js has no `Normalization` layer (used inside Keras' EfficientNet). It is mathematically the same as an
inference-mode BatchNormalization with gamma=1, beta=0, moving_mean=mean, moving_variance=variance, so we swap it.
The converted Keras model is checked against the original before exporting.

Usage: python export_tfjs.py --model /tmp/scin/out_ft/model.h5 --data /tmp/scin/data --out /tmp/scin/out_ft
"""
import argparse, json, os, shutil
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="/tmp/scin/data")
ap.add_argument("--out", required=True)
a = ap.parse_args()

orig = keras.models.load_model(a.model, compile=False)
print("loaded", orig.name)


def convert_backbone(bb):
    """clone the EfficientNet backbone, replacing Normalization by BatchNormalization, and copy weights by layer name"""
    stats = {}

    def clone_fn(layer):
        if isinstance(layer, keras.layers.Normalization):
            mean, var = layer.get_weights()[:2]
            stats[layer.name] = (mean.reshape(-1), var.reshape(-1))
            return keras.layers.BatchNormalization(axis=-1, epsilon=1e-7, name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    new_bb = keras.models.clone_model(bb, clone_function=clone_fn)
    orig_layers = {l.name: l for l in bb.layers}
    for l in new_bb.layers:
        if l.name in stats:
            mean, var = stats[l.name]
            l.set_weights([np.ones_like(mean), np.zeros_like(mean), mean, var])
        elif orig_layers[l.name].get_weights():
            l.set_weights(orig_layers[l.name].get_weights())
    return new_bb, stats


backbone = [l for l in orig.layers if isinstance(l, keras.Model)][0]
new_bb, nw = convert_backbone(backbone)
inp = keras.Input(shape=orig.input_shape[1:], name="image")
x = new_bb(inp, training=False)
for l in orig.layers:           # the small head: Dropout / Dense
    if isinstance(l, (keras.layers.InputLayer, keras.Model)):
        continue
    clone = l.__class__.from_config(l.get_config())
    x = clone(x)
    if l.get_weights():
        clone.set_weights(l.get_weights())
new = keras.Model(inp, x, name=orig.name)
print("replaced Normalization layers:", list(nw))
assert not any(isinstance(l, keras.layers.Normalization) for l in new_bb.layers)

# ---- check: converted model == original model on real test images ----------------------------
rows = json.load(open(f"{a.data}/index.json"))
test = [r[0] for r in rows if r[3] == "test"][:16]
X = np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in test])
p0, p1 = orig.predict(X, verbose=0), new.predict(X, verbose=0)
diff = float(np.abs(p0 - p1).max())
print("max |original - converted| on 16 test images:", diff)
assert diff < 1e-3, "conversion changed the model output"

import tensorflowjs as tfjs
d = f"{a.out}/tfjs"
shutil.rmtree(d, ignore_errors=True)
tfjs.converters.save_keras_model(new, d, quantization_dtype_map={"float16": "*"})
print("exported", os.listdir(d))
ref = [{"file": test[i], "probs": [float(v) for v in p1[i]]} for i in range(3)]
json.dump(ref, open(f"{a.out}/reference.json", "w"), indent=1)
