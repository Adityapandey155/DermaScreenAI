# Training the model on Google Colab (optional)

The scripts in this folder were run on a CPU machine. With a free Colab GPU they run much faster, and you can train
for more epochs or fine-tune more layers.

```python
# 1. new Colab notebook -> Runtime > Change runtime type > GPU (T4)
!pip install -q "tensorflow==2.15.0" "keras==2.15.0" "numpy<2" "protobuf==4.25.3" pillow scikit-learn
!pip install -q --no-deps tensorflowjs==4.17.0
!pip install -q tensorflow_hub packaging h5py importlib_resources "setuptools<70"

# tensorflowjs imports two optional packages we do not need - create harmless stubs
import site, os, pathlib
sp = site.getsitepackages()[0]
os.makedirs(f"{sp}/tensorflow_decision_forests", exist_ok=True)
pathlib.Path(f"{sp}/tensorflow_decision_forests/__init__.py").write_text("")
pathlib.Path(f"{sp}/tensorflowjs/converters/jax_conversion.py").write_text("def convert_jax(*a, **k): raise NotImplementedError\n")

# 2. upload the training/ folder of this project (or clone your repository), then:
!mkdir -p /content/scin && cd /content/scin && \
  wget -q https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv -O cases.csv && \
  wget -q https://storage.googleapis.com/dx-scin-public-data/dataset/scin_labels.csv -O labels.csv
%cd /content/training
!python prepare_data.py --csv_dir /content/scin --out /content/scin/data      # download + clean + case-level split
!python finetune.py --data /content/scin/data --out /content/scin/out_ft --epochs 12   # train, evaluate, export

# 3. download /content/scin/out_ft/tfjs/ and metrics.json
```

On your computer, install the result into the website:

```bash
cd training
python make_model_info.py --out <folder containing metrics.json and tfjs/> --site ..
```

This copies the model to `public/models/skin/` and writes `model_info.json` with the **real** evaluation numbers.
Then run `npm run build` again.
