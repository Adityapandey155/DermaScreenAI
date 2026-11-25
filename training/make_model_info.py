"""
Step 3 - Install the trained model into the website.

Copies the TensorFlow.js files produced by train.py into public/models/skin/ and writes
model_info.json (class names, reject threshold and the REAL evaluation numbers from metrics.json).
The website reads model_info.json at runtime, so nothing about accuracy is hard-coded in the UI.

Usage:  python make_model_info.py --out /tmp/scin/out --site ..
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/scin/out", help="folder written by train.py")
ap.add_argument("--site", default="..", help="project root")
a = ap.parse_args()

m = json.load(open(f"{a.out}/metrics.json"))
dst = os.path.join(a.site, "public", "models", "skin")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(f"{a.out}/tfjs", dst)

info = {
    "trained": True,
    "version": "1.0",
    "backbone": m["backbone"],
    "inputSize": 224,
    "classes": m["classes"],
    "dataset": "SCIN - Skin Condition Image Network (Google Research)",
    "trainedOn": "SCIN cases with dermatologist labels, case-level train/val/test split",
    "rejectThreshold": m["reject_threshold"],
    "images": m["images"],
    "metrics": {k: m[k] for k in (
        "test_accuracy", "test_macro_f1", "majority_class_baseline_accuracy",
        "test_coverage_at_threshold", "test_accuracy_on_accepted", "per_class",
        "train_class_counts", "test_class_counts")},
}
json.dump(info, open(os.path.join(dst, "model_info.json"), "w"), indent=1)
shutil.copy(f"{a.out}/metrics.json", os.path.join(a.site, "training", "metrics.json"))
print("installed model into", dst, os.listdir(dst))
