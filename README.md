# DermaScreen AI – AI-Powered Skin Condition Screening Using Smartphone Images

An **academic screening prototype**. The user uploads or captures a skin photograph, a deep-learning model
(EfficientNet-B0, TensorFlow.js) runs **inside the browser**, and the app shows possible condition *categories*,
the raw model score, basic educational information and a recommendation to consult a healthcare professional.
It is **not** a medical diagnosis and does not replace a dermatologist.

## Honest status

| Part | Status |
|---|---|
| Website (home, upload/camera, analysis, results, about) | Implemented; on-device inference is verified with the included model |
| Real trained model (SCIN dataset, EfficientNet-B0, TF.js) | Trained model and both weight shards are included; TF.js output matches Keras on sample images |
| Model quality | **Weak** – see below. It is a student prototype, not a useful clinical tool |
| Netlify static export | Production build passes; `out/` includes the model files. A live Netlify deployment has not been performed |

### Real evaluation (held-out test split, 974 images, split by case ID, nothing hard-coded – see `training/metrics.json`)

* Accuracy **33.0%**, macro-F1 **0.22**, 7 categories. Always predicting the largest class (eczema/dermatitis) gives 42.3% accuracy, so **the model does not beat that trivial baseline on raw accuracy**.
* Per category F1: eczema/dermatitis 0.51, insect bite 0.23, acne/folliculitis 0.22, other 0.19, urticaria 0.18, psoriasis 0.16, **fungal infection 0.02** (essentially not learned).
* Because of this the app only names a category when the top score ≥ **0.78** (threshold chosen on the validation set). On the test set that happens for **14.3%** of images, and **51.8%** of those answers were correct. For all other images the app says *"Unable to assess this image reliably…"*.
* Why so weak: SCIN is small for this task (≈3 000 usable cases), its labels are dermatologists' differential opinions from photos (noisy, many look-alike classes), and training was done on a small CPU machine for 10 epochs. Better results would need more data/GPU time/tuning.

Two other approaches were tried and rejected on **validation** data: a frozen EfficientNet-B0 feature extractor + dense head (validation macro-F1 0.23, `training/train.py`) – the partially fine-tuned model (validation macro-F1 0.28, `training/finetune.py`) was selected.

## Model files and accuracy

The trained TensorFlow.js model and both weight shards are included in `public/models/skin/`. The shard files are about 8 MB combined. The app checks for the model assets before enabling analysis and reports a clear error if deployment omits them.

The supplied SCIN model is a working academic prototype, but its measured test accuracy (33.0%) is below the 42.3% majority-class baseline. The reject threshold controls when the app abstains; it does not raise the model's accuracy. Do not present its output as a diagnosis.

## Model and data

* Dataset: **SCIN – Skin Condition Image Network** (https://github.com/google-research-datasets/scin), the only dataset used.
* Cases with dermatologist labels only; the top-weighted label is grouped into 7 categories (`training/prepare_data.py`): Eczema/Dermatitis, Psoriasis, Fungal infection, Urticaria, Insect bite, Acne/Folliculitis, Other. Acne alone (~60 cases) and scabies (~20) were too small, so acne is merged with folliculitis and scabies falls under "Other". The "Other" class is capped at 450 cases.
* 60/20/20 train/val/test split **by case ID** (no case appears in two splits; asserted in the code). Corrupt/tiny images are dropped. Images are resized to 224×224.
* Model: EfficientNet-B0 (ImageNet), blocks 6–7 + new Dense(7, softmax) head trained; BatchNorm frozen; label smoothing; sqrt-inverse-frequency class weights; flip/rotation/zoom/brightness/contrast augmentation; best epoch by validation macro-F1.
* Export: TF.js layers model, float16 weights (~8 MB) in `public/models/skin/`. TF.js has no `Normalization` layer, so `training/export_tfjs.py` replaces EfficientNet's one with an equivalent BatchNormalization (outputs checked identical, max difference 1.7e-6). `training/verify_tfjs.mjs` confirmed the exported files give the same scores in TF.js as in Keras (max difference ≈3e-3 from float16 rounding).

## Verification performed

* `npm run typecheck` passes.
* `NEXT_STATIC_EXPORT=1 npm run build` passes with Next.js 16.4.6.
* TensorFlow.js loaded the exported model (input 224�224�3, output 7 categories) and matched Keras predictions on 3 held-out images; maximum score difference was 0.0034.
* The retrained model was evaluated on 974 held-out SCIN images: accuracy 33.0%, macro-F1 0.22, and accepted-case accuracy 51.8% at 14.3% coverage.
* Real phone camera behavior and live Netlify deployment were not checked.

## Run locally

```bash
npm install
npm run dev            # http://localhost:3000
```

Production check of the Netlify build:

```bash
NEXT_STATIC_EXPORT=1 npm run build     # creates the static site in ./out
npx serve out                           # or any static server
```

## Deploy to Netlify

The project is a Next.js app exported as a **static site** (no server, no functions, no database, no API key). `netlify.toml` already contains everything:

* Build command: `npm run build`
* Publish directory: `out`
* Environment variable `NEXT_STATIC_EXPORT=1` (set in `netlify.toml`; this switches Next.js to `output: "export"`)

Steps:
1. Push this folder to GitHub.
2. Netlify → *Add new site* → *Import an existing project* → choose the repository. The settings are read from `netlify.toml` (check that Publish directory = `out`).
3. Deploy. Netlify serves `/screening`, `/results`, `/about` from the exported `.html` files; unknown URLs show `404.html`. No SPA redirect is needed.
4. Open the deployed HTTPS site and run one analysis; the model is fetched from `/models/skin/model.json` on the same domain.

Alternative (manual): run the static build locally and drag the `out` folder onto https://app.netlify.com/drop.

## Training the model yourself

`training/` contains everything (Python 3.11, `tensorflow-cpu==2.15`, `keras==2.15`, `tensorflowjs==4.17`; see `training/README_COLAB.md` for a copy-paste Google Colab recipe):

```bash
python training/prepare_data.py --csv_dir <folder with scin_cases.csv + scin_labels.csv as cases.csv/labels.csv> --out data
python training/finetune.py     --data data --out out_ft --epochs 10      # train + evaluate + saves model.h5
python training/export_tfjs.py  --model out_ft/model.h5 --data data --out out_ft   # TF.js-compatible export
cd training && python make_model_info.py --out ../out_ft --site ..        # copies model to public/models/skin + model_info.json
```

The website reads `public/models/skin/model_info.json` (classes, threshold and real metrics). **If that file or the model is missing the app shows "AI analysis is currently unavailable" and never fabricates predictions.**

## Project structure

```
public/models/skin/      trained TF.js model + model_info.json
src/app/                 pages: / (home), /screening (upload + analysis), /results, /about
src/components/          Navbar, Footer, Logo, Disclaimer, ImageUploader, ResultCard, AnalysisContext
src/ml/                  model.ts (loading), preprocessing.ts (validation, resize, quality checks), prediction.ts, conditions.ts (educational text)
training/                data preparation, training, export, verification scripts + metrics.json
netlify.toml             Netlify configuration
```

## Privacy and safety

* No login, no personal data, no analytics. Photos are processed only in the browser and kept only in memory (nothing in localStorage, cookies or any server).
* The text never says a user has or does not have a disease, never recommends medicines, and always recommends professional consultation. Scores are labelled as raw model outputs, not probabilities.

## Limitations

Weak accuracy, some categories barely learned (fungal infection), "Other" is a mixture, dataset-specific population and camera conditions, no skin-cancer detection, simple heuristics for photo quality. See the About page.
