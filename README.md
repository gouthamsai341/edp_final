<<<<<<< HEAD
# edp_final
=======
# 🌱 PlantCare AI — Plant Disease Detector

**Intelligent Plant Disease Detection** — an EDP AI/ML internship project.

**Team:** Potlacheruvu Goutham · Mohd Ibrahim · Diguva Sarvajeeth Kumar

---

## Overview

PlantCare AI is a web application that takes a photo of a plant leaf, runs it through a
trained convolutional neural network, and returns the predicted plant + disease, a
confidence score, and general symptom/management/prevention information. Low-confidence
predictions are explicitly flagged as uncertain rather than guessed.

**Important — read before you run this:** this repository ships the **complete, working
code** for the data pipeline, training pipeline, evaluation pipeline, and the full Flask
web application. It does **not** ship a pre-trained model or the PlantVillage dataset
itself (too large for source control, and licensing requires you to obtain it yourself).
Until you run the training pipeline (see below), the web app will start normally and every
page will load, but `/api/predict` will return a clear "model not available yet" message
instead of a prediction. No accuracy/precision/recall/F1 numbers are hard-coded anywhere —
every metric in `reports/` is generated from real evaluation once you train a model.

---

## Features

- Drag-and-drop / click-to-upload leaf image analysis
- Real-time inference with a confidence score and an explicit uncertainty threshold
- Disease information library (symptoms, management, prevention) — searchable
- Prediction history stored in SQLite, viewable and clearable
- Responsive UI (desktop, tablet, mobile)
- Secure file upload handling (extension allow-list, size limit, randomized filenames)
- Full ML pipeline: cleaning → leakage-safe split → augmentation → CNN baseline →
  EfficientNetB0 transfer learning → fine-tuning → evaluation → error analysis

## Architecture

```
Leaf Image (browser)
      │  drag & drop / choose file
      ▼
Flask route POST /api/predict
      │  validate (extension, size) → save securely
      ▼
src/predict.py  (same code path used in evaluate.py)
      │  RGB convert → resize 224×224 → normalize → model.predict()
      ▼
Confidence ≥ threshold?  ── no ──▶ "uncertain" response
      │ yes
      ▼
{plant, disease, confidence} → saved to SQLite → shown on Result page
                                                 → disease_info.json enriches the result
```

## Technologies

Flask, Flask-SQLAlchemy, SQLite, TensorFlow/Keras (custom CNN + EfficientNetB0), NumPy,
Pandas, Pillow, scikit-learn, Matplotlib, imagehash, PyYAML, pytest, HTML/CSS/vanilla JS.

## Dataset

Primary dataset: **PlantVillage** (leaf images organized by `<Plant>___<Disease>` folders).
This project does **not** hard-code the class list — `src/split_dataset.py` discovers
classes dynamically from whatever folders exist under `data/raw/`, so you can use the full
38-class dataset or a smaller subset.

**To obtain the dataset:**
1. Download a PlantVillage-format dataset (e.g. from Kaggle — search "PlantVillage Dataset").
2. Extract it so you have `data/raw/<ClassName>/*.jpg` for each class (folder names like
   `Tomato___Early_blight`, `Apple___healthy`, etc. — see `data/disease_info.json` for the
   full list of classes this project has curated care information for; you can add more
   classes to that file if you use additional dataset classes).

## Dataset Cleaning

Run:
```
python -m src.data_cleaning
```
This scans every class folder, verifies each image (corrupt/unreadable/too-small files are
never silently deleted — they are copied into `data/quarantine/<reason>/<class>/`), converts
everything to RGB, deduplicates near-identical images per class via perceptual hashing, and
writes `reports/dataset_report.json`. Then split the cleaned data:
```
python -m src.split_dataset
```
This performs a **stratified 70/15/15 train/val/test split** (configurable in
`config.yaml`) with a fixed random seed (42) for reproducibility, writing to
`data/split/{train,val,test}/<Class>/` and generating `models/class_names.json` (the
canonical, ordered label list every other script relies on).

**Data leakage prevention:** augmentation is applied only inside the training scripts, only
to the `train/` split, at training time — never to validation or test data, and never
before the split. See `src/preprocessing.py` (`build_augmentation_layer`) and
`src/dataset_utils.py`.

## CNN Model

`src/train_cnn.py` builds and trains a custom CNN from scratch (Conv2D → BatchNorm → ReLU →
MaxPooling blocks, GlobalAveragePooling, Dropout, Dense/softmax head) as the required
from-scratch baseline. Saved to `models/cnn_baseline.keras`.

## Transfer Learning

`src/train_efficientnet.py` builds an EfficientNetB0 (ImageNet weights) model in two stages:
1. **Head training** — backbone frozen, only the new classification head trains.
2. **Fine-tuning** — top N backbone layers unfrozen, trained with a much lower learning rate.

Uses `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, and computed class weights.
Saved to `models/best_plant_disease_model.keras`.

**Model selection is evidence-based, not assumed:** run `src/evaluate.py` against both
`models/cnn_baseline.keras` and `models/best_plant_disease_model.keras` and compare the
real test-set metrics before deciding which one the app should use (the app defaults to
`best_plant_disease_model.keras`, configurable in `config.yaml`).

## Training

```bash
python src/train_cnn.py                 # custom CNN baseline
python src/train_efficientnet.py        # EfficientNetB0 transfer learning + fine-tuning
```
Both accept `--epochs`/`--batch-size` (or `--epochs-head`/`--epochs-finetune` for
EfficientNet) overrides. Training writes accuracy/loss curves, a CSV training log, and
history JSON to `reports/`.

## Evaluation

```bash
python src/evaluate.py --model models/best_plant_disease_model.keras
python src/error_analysis.py --model models/best_plant_disease_model.keras
```
Generates, from real predictions on the held-out test set:
- `reports/test_classification_report.csv`, `test_per_class_metrics.csv`,
  `test_per_class_accuracy.csv`, `test_confusion_matrix.png`, `test_metrics.json`
- `reports/misclassified_report.csv`, `reports/top_confusion_pairs.csv`

If you add real-world photos to `data/external_test/<Class>/`, `evaluate.py` scores them
**separately** as `external_test_*` reports — PlantVillage performance is not assumed to
equal real-world performance.

## Metrics

Not shown here — see the note at the top of this README. Run the training/evaluation
commands above; the resulting `reports/test_metrics.json` will contain your model's real
accuracy, precision, recall, and F1-score (macro and weighted averages), plus a per-class
breakdown.

## Installation

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Project

```
python run.py
```
Then open `http://localhost:5000`. The app works with no trained model (every page loads;
`/api/predict` explains that a model isn't trained yet) or with a trained model in `models/`.

## Training the Model

```
python -m src.data_cleaning
python -m src.split_dataset
python src/train_cnn.py
python src/train_efficientnet.py
python src/evaluate.py
python src/error_analysis.py
```

## Running the Web App

```
python run.py
```
Set `PLANTCARE_SECRET_KEY` as an environment variable in any real deployment (never commit
a secret key); `flask.debug` in `config.yaml` should stay `false` outside local dev.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Upload an image (`multipart/form-data`, field `image`) → prediction |
| GET | `/api/history` | List prediction history |
| DELETE | `/api/history/<id>` | Delete one history record |
| DELETE | `/api/history` | Clear all history |
| GET | `/api/diseases` | Full disease library |
| GET | `/api/diseases/<class_id>` | One disease's info |
| GET | `/api/health` | `{"status": "healthy", "model_ready": bool}` |

## Project Structure

```
Plant-Disease-Detector/
├── app/                  # Flask backend + frontend templates/static
│   ├── app.py            # application factory
│   ├── config.py
│   ├── extensions.py     # SQLAlchemy instance
│   ├── models_db.py      # Prediction table
│   ├── routes/           # prediction / history / disease blueprints
│   ├── services/         # model / prediction / disease / history services
│   ├── utils/            # upload validation
│   ├── templates/
│   └── static/{css,js}
├── data/
│   ├── raw/  cleaned/  external_test/   (populate raw/ yourself)
│   └── disease_info.json
├── models/               # cnn_baseline.keras, best_plant_disease_model.keras, class_names.json
├── src/                  # the entire ML pipeline (see above)
├── reports/              # generated by training/evaluation
├── tests/
├── uploads/
├── config.yaml
├── requirements.txt
├── .gitignore
├── README.md
└── run.py
```

*Note on notebooks:* rather than duplicating logic between `notebooks/*.ipynb` and
`src/*.py`, every pipeline step lives in `src/` as a single, tested, reproducible source of
truth. If you want exploratory notebooks for the internship writeup, they can simply import
and call these same functions (`from src.data_cleaning import run_cleaning`, etc.).

## Screenshots

Not included — add your own after running the app against a trained model, so screenshots
reflect real results rather than placeholders.

## Limitations

- PlantVillage-style datasets are captured under controlled conditions; performance on
  real-world, cluttered, or poorly-lit photos may be lower (this is why `external_test/`
  exists — use it).
- The confidence threshold (`config.yaml` → `inference.confidence_threshold`, default 0.60)
  is a starting point, not a scientifically validated cutoff — tune it against your own
  validation-set confidence distribution.
- There is no dedicated "is this even a leaf" classifier; wildly out-of-domain images will
  simply tend to produce low-confidence (and therefore "uncertain") predictions rather than
  being explicitly rejected.
- This is an educational/research prototype, not a substitute for expert agricultural
  diagnosis.

## Future Improvements

- TensorFlow Lite export (`models/plant_disease_model.tflite`) for mobile deployment
- A dedicated leaf-vs-non-leaf gate model
- Multi-language disease information
- User accounts and per-user history

## Team

Potlacheruvu Goutham · Mohd Ibrahim · Diguva Sarvajeeth Kumar — EDP AI/ML Internship
>>>>>>> e6d4938 (Initial commit)
