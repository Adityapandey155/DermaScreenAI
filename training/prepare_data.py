"""
Step 1 - Prepare the SCIN dataset.

* reads scin_cases.csv + scin_labels.csv (public SCIN bucket)
* keeps cases that dermatologists graded (non-empty weighted label)
* maps the top-weighted dermatologist label to a small set of categories
* splits by CASE ID (60/20/20) so images of one case never leak across splits
* downloads every image of each case, drops corrupt / tiny files,
  and stores a 224x224 RGB JPEG copy.

Usage:  python prepare_data.py --csv_dir /tmp/scin --out /tmp/scin/data
"""
import argparse, ast, csv, json, os, random, collections, io, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

BASE = "https://storage.googleapis.com/dx-scin-public-data/"

# category -> list of SCIN labels. Everything else that is graded -> "other".
GROUPS = {
    "eczema_dermatitis": ["Eczema", "Allergic Contact Dermatitis", "Irritant Contact Dermatitis",
                          "Acute dermatitis, NOS", "CD - Contact dermatitis", "Acute and chronic dermatitis",
                          "Chronic dermatitis, NOS", "Stasis Dermatitis", "Seborrheic Dermatitis",
                          "Lichen Simplex Chronicus"],
    "psoriasis": ["Psoriasis"],
    "fungal_infection": ["Tinea", "Tinea Versicolor", "Candida", "Onychomycosis"],
    "urticaria": ["Urticaria"],
    "insect_bite": ["Insect Bite"],
    "acne_folliculitis": ["Acne", "Folliculitis"],
}
OTHER_CAP = 450  # cases; keeps "other" from swamping everything
SIZE = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="/tmp/scin")
    ap.add_argument("--out", default="/tmp/scin/data")
    a = ap.parse_args()

    lab2cat = {l: c for c, ls in GROUPS.items() for l in ls}
    labels = {}
    for r in csv.DictReader(open(f"{a.csv_dir}/labels.csv")):
        try:
            d = ast.literal_eval(r["weighted_skin_condition_label"])
        except Exception:
            d = {}
        if d:
            labels[r["case_id"]] = max(d, key=d.get)

    cases = {r["case_id"]: r for r in csv.DictReader(open(f"{a.csv_dir}/cases.csv"))}
    by_cat = collections.defaultdict(list)
    for cid, top in labels.items():
        if cid not in cases:
            continue
        by_cat[lab2cat.get(top, "other")].append(cid)
    rng = random.Random(1)
    by_cat["other"].sort()
    rng.shuffle(by_cat["other"])
    by_cat["other"] = by_cat["other"][:OTHER_CAP]

    # case-level split, stratified by category
    split_of = {}
    for cat, ids in by_cat.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(n * 0.6), int(n * 0.2)
        for i, cid in enumerate(ids):
            split_of[cid] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    jobs = []
    for cat, ids in by_cat.items():
        for cid in ids:
            for k in (1, 2, 3):
                p = cases[cid].get(f"image_{k}_path")
                if p:
                    jobs.append((cid, cat, split_of[cid], k, p))
    print("cases per category:", {c: len(v) for c, v in by_cat.items()})
    print("images to download:", len(jobs), flush=True)

    for s in ("train", "val", "test"):
        os.makedirs(f"{a.out}/{s}", exist_ok=True)

    def work(j):
        cid, cat, split, k, p = j
        fn = f"{a.out}/{split}/{cat}__{cid}__{k}.jpg"
        if os.path.exists(fn):
            return (fn, cid, cat, split)
        try:
            data = urllib.request.urlopen(BASE + p, timeout=60).read()
            im = Image.open(io.BytesIO(data))
            im.load()
            im = im.convert("RGB")
            if min(im.size) < 64:
                return None
            im.resize((SIZE, SIZE), Image.BILINEAR).save(fn, quality=92)
            return (fn, cid, cat, split)
        except Exception as e:
            print("skip", p, type(e).__name__, flush=True)
            return None

    rows = []
    with ThreadPoolExecutor(12) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            if r:
                rows.append(r)
            if i % 300 == 0:
                print("progress", i, "/", len(jobs), flush=True)

    summary = collections.defaultdict(collections.Counter)
    for fn, cid, cat, split in rows:
        summary[split][cat] += 1
    json.dump({s: dict(c) for s, c in summary.items()}, open(f"{a.out}/summary.json", "w"), indent=1)
    json.dump(rows, open(f"{a.out}/index.json", "w"))
    print(json.dumps({s: dict(c) for s, c in summary.items()}, indent=1))
    print("total images kept:", len(rows))


if __name__ == "__main__":
    main()
