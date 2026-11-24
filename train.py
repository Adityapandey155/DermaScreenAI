"""
Step 2 - Train + evaluate + export (TensorFlow 2.15 / Keras 2.15 / tensorflowjs 4.17).

Transfer learning with EfficientNet-B0 (ImageNet weights):
  1. the pretrained backbone is kept frozen and used as a feature extractor
     (train images are also extracted with flips / 90-degree rotations = simple augmentation)
  2. a small classification head (Dense 256 -> Dense N softmax) is trained with
     class weights (class imbalance) and label smoothing, early-stopped on VALIDATION loss
  3. the head is attached to the backbone to make ONE model:  image (0-255 RGB) -> class scores
  4. evaluated on the held-out TEST split (case-level split, no leakage)
  5. exported to TensorFlow.js layers format (float16 weights) for the website

Usage:
  python train.py --data /tmp/scin/data --out /tmp/scin/out
"""
import argparse, json, os, collections
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ["eczema_dermatitis", "psoriasis", "fungal_infection", "urticaria",
           "insect_bite", "acne_folliculitis", "other"]
SIZE = 224


def load(rows, split):
    X, y, cid = [], [], []
    for fn, c, cat, s in rows:
        if s != split:
            continue
        X.append(np.asarray(Image.open(fn).convert("RGB"), dtype=np.uint8))
        y.append(CLASSES.index(cat))
        cid.append(c)
    return np.stack(X), np.array(y), cid


def extract(backbone, X, augment=False, bs=32):
    variants = [lambda a: a]
    if augment:
        variants += [lambda a: a[:, ::-1], lambda a: a[::-1], lambda a: np.rot90(a, 1, (1, 2))]
    feats, labels_idx = [], []
    for v in variants:
        f = []
        for i in range(0, len(X), bs):
            b = np.ascontiguousarray(v(X[i:i + bs])).astype("float32")
            f.append(backbone.predict(b, verbose=0))
        feats.append(np.concatenate(f))
    return np.concatenate(feats), len(variants)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/scin/data")
    ap.add_argument("--out", default="/tmp/scin/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tf.random.set_seed(0); np.random.seed(0)

    rows = json.load(open(f"{a.data}/index.json"))
    Xtr, ytr, ctr = load(rows, "train")
    Xva, yva, cva = load(rows, "val")
    Xte, yte, cte = load(rows, "test")
    assert not (set(ctr) & set(cva)) and not (set(ctr) & set(cte)) and not (set(cva) & set(cte)), "case leakage!"
    print("images train/val/test:", len(Xtr), len(Xva), len(Xte), "(no shared case IDs)", flush=True)

    # EfficientNetB0 from keras.applications has its own preprocessing inside -> expects 0..255 pixels
    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_shape=(SIZE, SIZE, 3), pooling="avg")
    backbone.trainable = False

    cache = f"{a.out}/features.npz"
    if os.path.exists(cache):
        z = np.load(cache); Ftr, Fva, Fte, k = z["Ftr"], z["Fva"], z["Fte"], int(z["k"])
    else:
        Ftr, k = extract(backbone, Xtr, augment=True)
        Fva, _ = extract(backbone, Xva); Fte, _ = extract(backbone, Xte)
        np.savez(cache, Ftr=Ftr, Fva=Fva, Fte=Fte, k=k)
    Ytr = np.tile(ytr, k)
    print("features", Ftr.shape, Fva.shape, Fte.shape, flush=True)

    counts = np.bincount(ytr, minlength=len(CLASSES))
    oh = lambda y: keras.utils.to_categorical(y, len(CLASSES))

    class BestMacroF1(keras.callbacks.Callback):
        """keeps the weights of the epoch with the best VALIDATION macro-F1 (test set is never touched here)"""
        def on_train_begin(self, logs=None): self.best, self.w, self.ep = -1, None, 0
        def on_epoch_end(self, epoch, logs=None):
            f = f1_score(yva, self.model.predict(Fva, verbose=0).argmax(1), average="macro")
            if f > self.best: self.best, self.w, self.ep = f, self.model.get_weights(), epoch + 1

    def build(hidden, drop, l2, lr):
        layers = [keras.layers.Input(shape=(Ftr.shape[1],)), keras.layers.Dropout(drop)]
        if hidden:
            layers += [keras.layers.Dense(hidden, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)),
                       keras.layers.Dropout(drop)]
        layers += [keras.layers.Dense(len(CLASSES), activation="softmax", kernel_regularizer=keras.regularizers.l2(l2))]
        m = keras.Sequential(layers)
        m.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
        return m

    best = None
    for hidden in (0, 256):
        for drop in (0.5, 0.7):
            for l2 in (1e-3, 1e-2):
                for power in (0.5, 1.0):
                    cw = {i: float((len(ytr) / (len(CLASSES) * max(1, counts[i]))) ** power) for i in range(len(CLASSES))}
                    tf.keras.utils.set_random_seed(0)
                    h = build(hidden, drop, l2, 1e-3)
                    cb = BestMacroF1()
                    h.fit(Ftr, oh(Ytr), epochs=30, batch_size=64, class_weight=cw, verbose=0, callbacks=[cb])
                    print(f"hidden={hidden} drop={drop} l2={l2} cw_power={power}: best val macro-F1 {cb.best:.3f} @epoch {cb.ep}", flush=True)
                    if best is None or cb.best > best[0]:
                        h.set_weights(cb.w); best = (cb.best, h, dict(hidden=hidden, drop=drop, l2=l2, cw_power=power, epoch=cb.ep))
    print("selected on validation:", best[2], "val macro-F1", round(best[0], 3))
    head = best[1]
    selected_cfg = best[2]

    # ---- evaluation on test --------------------------------------------------------
    pva, pte = head.predict(Fva, verbose=0), head.predict(Fte, verbose=0)
    pred = pte.argmax(1)
    acc = float((pred == yte).mean())
    mf1 = float(f1_score(yte, pred, average="macro"))
    maj = float(np.bincount(yte).max() / len(yte))
    rep = classification_report(yte, pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=list(range(len(CLASSES)))).tolist()
    print(classification_report(yte, pred, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0))

    # ---- reject threshold chosen on VALIDATION set only ----------------------------
    conf_va, ok_va = pva.max(1), pva.argmax(1) == yva
    thr = None
    for t in np.arange(0.30, 0.95, 0.01):
        m = conf_va >= t
        if m.mean() >= 0.10 and ok_va[m].mean() >= 0.5:
            thr = float(round(t, 2)); break
    if thr is None:  # fall back: accept the most confident 20% of validation images
        thr = float(round(np.quantile(conf_va, 0.8), 2))
    conf_te, ok_te = pte.max(1), pred == yte
    m = conf_te >= thr
    cov = float(m.mean())
    acc_acc = float(ok_te[m].mean()) if m.any() else None
    print(f"threshold {thr}: test coverage {cov:.3f}, accuracy on accepted {acc_acc}")

    metrics = {
        "backbone": "EfficientNet-B0 (ImageNet weights, frozen feature extractor) + trained dense head",
        "classes": CLASSES,
        "images": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "train_class_counts": {CLASSES[i]: int(c) for i, c in enumerate(counts)},
        "test_class_counts": {CLASSES[i]: int(c) for i, c in enumerate(np.bincount(yte, minlength=len(CLASSES)))},
        "test_accuracy": acc, "test_macro_f1": mf1, "majority_class_baseline_accuracy": maj,
        "per_class": {c: {k2: float(rep[c][k2]) for k2 in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])} for c in CLASSES},
        "confusion_matrix": cm,
        "selected_head_config": selected_cfg, "val_macro_f1": float(best[0]),
        "reject_threshold": thr, "test_coverage_at_threshold": cov, "test_accuracy_on_accepted": acc_acc,
    }
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=1)

    # ---- assemble the full model: image -> scores ---------------------------------------
    inp = keras.Input(shape=(SIZE, SIZE, 3), name="image")
    x = backbone(inp, training=False)
    for layer in head.layers:
        x = layer(x)
    full = keras.Model(inp, x, name="skin_screening_efficientnetb0")
    # sanity: full model must reproduce the head outputs
    chk = full.predict(Xte[:8].astype("float32"), verbose=0)
    assert np.allclose(chk, pte[:8], atol=1e-3), "assembled model mismatch"

    import tensorflowjs as tfjs
    tfjs_dir = f"{a.out}/tfjs"
    tfjs.converters.save_keras_model(full, tfjs_dir, quantization_dtype_map={"float16": "*"})
    print("exported to", tfjs_dir, os.listdir(tfjs_dir))

    # reference outputs for verifying the browser/TFJS inference later
    ref = []
    for i in (0, 1, 2):
        fn = [r for r in rows if r[3] == "test"][i][0]
        ref.append({"file": fn, "probs": [float(v) for v in full.predict(Xte[i:i + 1].astype("float32"), verbose=0)[0]]})
    json.dump(ref, open(f"{a.out}/reference.json", "w"), indent=1)


if __name__ == "__main__":
    main()
