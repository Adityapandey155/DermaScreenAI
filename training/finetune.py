"""
Step 2b (optional, better) - partial fine-tuning of EfficientNet-B0.

Same data / split / evaluation / export as train.py, but the upper blocks of EfficientNet-B0
(from block6a on) are trained too, with on-the-fly augmentation (flips, rotation, zoom, brightness, contrast).
BatchNorm layers stay frozen. The epoch with the best VALIDATION macro-F1 is kept.

Usage:  python finetune.py --data /tmp/scin/data --out /tmp/scin/out_ft --epochs 10
"""
import argparse, json, os, time
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from train import load, CLASSES, SIZE

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/scin/data")
ap.add_argument("--out", default="/tmp/scin/out_ft")
ap.add_argument("--epochs", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tf.keras.utils.set_random_seed(0); np.random.seed(0)

rows = json.load(open(f"{a.data}/index.json"))
Xtr, ytr, ctr = load(rows, "train"); Xva, yva, cva = load(rows, "val"); Xte, yte, cte = load(rows, "test")
assert not (set(ctr) & set(cva)) and not (set(ctr) & set(cte)) and not (set(cva) & set(cte))
print("images", len(Xtr), len(Xva), len(Xte), flush=True)

aug = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
    keras.layers.RandomContrast(0.15),
])

inp = keras.Input(shape=(SIZE, SIZE, 3), name="image")
backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(SIZE, SIZE, 3), pooling="avg")
backbone.trainable = True
unfreeze = False
for l in backbone.layers:
    if l.name.startswith("block6a"):
        unfreeze = True
    l.trainable = unfreeze and not isinstance(l, keras.layers.BatchNormalization)
x = backbone(inp, training=False)           # BN layers run in inference mode
x = keras.layers.Dropout(0.4)(x)
out = keras.layers.Dense(len(CLASSES), activation="softmax")(x)
model = keras.Model(inp, out, name="skin_screening_efficientnetb0")
print("trainable params:", sum(int(np.prod(w.shape)) for w in model.trainable_weights), flush=True)

counts = np.bincount(ytr, minlength=len(CLASSES))
cw = (len(ytr) / (len(CLASSES) * counts)) ** 0.5
sample_w = cw[ytr].astype("float32")
Ytr_oh = keras.utils.to_categorical(ytr, len(CLASSES)).astype("float32")
def gen():
    idx = np.random.permutation(len(Xtr))
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]
        yield Xtr[b].astype("float32"), Ytr_oh[b], sample_w[b]
ds = tf.data.Dataset.from_generator(gen, output_signature=(
    tf.TensorSpec((None, SIZE, SIZE, 3), tf.float32), tf.TensorSpec((None, len(CLASSES)), tf.float32),
    tf.TensorSpec((None,), tf.float32)))
ds = ds.map(lambda x, y, w: (aug(x, training=True), y, w)).prefetch(1)

def predict(X):
    return np.concatenate([model.predict(X[i:i + 32].astype("float32"), verbose=0) for i in range(0, len(X), 32)])

model.compile(optimizer=keras.optimizers.Adam(3e-4),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

best, best_w = -1, None
for ep in range(a.epochs):
    t0 = time.time()
    h = model.fit(ds, epochs=1, verbose=0)
    pva = predict(Xva)
    f = f1_score(yva, pva.argmax(1), average="macro")
    print(f"epoch {ep+1}: train_loss {h.history['loss'][0]:.3f} train_acc {h.history['accuracy'][0]:.3f} "
          f"val_acc {(pva.argmax(1)==yva).mean():.3f} val_macroF1 {f:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if f > best:
        best, best_w = f, model.get_weights()
model.set_weights(best_w)
print("best val macro-F1", round(best, 3), flush=True)

pva = predict(Xva)
pte = predict(Xte)
pred = pte.argmax(1)
labs = list(range(len(CLASSES)))
rep = classification_report(yte, pred, labels=labs, target_names=CLASSES, output_dict=True, zero_division=0)
print(classification_report(yte, pred, labels=labs, target_names=CLASSES, zero_division=0))

conf_va, ok_va = pva.max(1), pva.argmax(1) == yva
# Reject threshold chosen on the VALIDATION set only: the lowest score for which the accepted validation
# answers are at least 60% correct while still covering >= 5% of images (strict, because the model is weak).
thr = None
for t in np.arange(0.30, 0.99, 0.01):
    m = conf_va >= t
    if m.mean() >= 0.05 and ok_va[m].mean() >= 0.6:
        thr = float(round(t, 2)); break
if thr is None:  # fall back: accept only the most confident 10% of validation images
    thr = float(round(np.quantile(conf_va, 0.9), 2))
conf_te, ok_te = pte.max(1), pred == yte
m = conf_te >= thr
cov = float(m.mean()); acc_acc = float(ok_te[m].mean()) if m.any() else None
print(f"threshold {thr}: test coverage {cov:.3f}, accuracy on accepted {acc_acc}")

np.savez(f"{a.out}/probs.npz", pva=pva, pte=pte, yva=yva, yte=yte)
model.save(f"{a.out}/model.h5")
metrics = {
    "backbone": "EfficientNet-B0 (ImageNet weights, blocks 6-7 + head fine-tuned on SCIN)",
    "classes": CLASSES,
    "images": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
    "train_class_counts": {CLASSES[i]: int(c) for i, c in enumerate(counts)},
    "test_class_counts": {CLASSES[i]: int(c) for i, c in enumerate(np.bincount(yte, minlength=len(CLASSES)))},
    "test_accuracy": float((pred == yte).mean()), "test_macro_f1": float(f1_score(yte, pred, average="macro")),
    "majority_class_baseline_accuracy": float(np.bincount(yte).max() / len(yte)),
    "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])} for c in CLASSES},
    "confusion_matrix": confusion_matrix(yte, pred, labels=labs).tolist(),
    "val_macro_f1": float(best),
    "reject_threshold": thr, "test_coverage_at_threshold": cov, "test_accuracy_on_accepted": acc_acc,
}
json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=1)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, f"{a.out}/tfjs", quantization_dtype_map={"float16": "*"})
ref = []
test_files = [r[0] for r in rows if r[3] == "test"]
for i in (0, 1, 2):
    ref.append({"file": test_files[i], "probs": [float(v) for v in model.predict(Xte[i:i+1].astype("float32"), verbose=0)[0]]})
json.dump(ref, open(f"{a.out}/reference.json", "w"), indent=1)
print("exported", os.listdir(f"{a.out}/tfjs"))
