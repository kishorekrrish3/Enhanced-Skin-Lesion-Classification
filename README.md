<div align="center">
  <h1>Enhanced Skin Lesion Classification</h1>
  <p><strong>A data-centric computer-vision research project for 14-class skin-lesion image classification.</strong></p>
  <p>Combines class-imbalance handling, image augmentation, model training, evaluation, and Grad-CAM visualizations with a web demo.</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Frontend-Next.js-black?logo=next.js" alt="Next.js">
  </p>
</div>

---

This project explores data-centric methods for skin-lesion image classification, including augmentation strategies for imbalanced classes, model training and evaluation, and Grad-CAM visualizations to inspect image regions that influence model predictions. The repository includes a Python ML pipeline, a FastAPI service, and a Next.js interface.

> **Medical disclaimer:** This is an academic/research prototype, not a medical device. Predictions may be inaccurate and must not be used to diagnose a skin condition, rule out cancer, or guide treatment. A qualified clinician should assess skin lesions. Grad-CAM visualizations are not proof that a prediction is correct or clinically explainable.

## Project components

- **Dataset and preprocessing** — configuration, dataset loading, augmentation, and class handling.
- **Training pipeline** — PyTorch model definitions, loss functions, training, and evaluation scripts.
- **Model inspection** — Grad-CAM visualizations and saved evaluation artifacts.
- **Inference API** — FastAPI endpoints for the web client.
- **Web interface** — Next.js demo for image upload, model selection, and viewing returned predictions.

## Model classes

The configured class list contains 14 categories, including actinic keratoses, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanocytic nevi, melanoma, squamous cell carcinoma, vascular lesions, and several infectious or healthy-skin categories. See `core/config.py` for the exact labels used by the current implementation.

## Tech stack

| Area | Technology |
|---|---|
| Model training | PyTorch, torchvision |
| Data processing | NumPy, pandas, OpenCV, Albumentations |
| Evaluation and plots | Matplotlib and related tooling |
| API | FastAPI |
| Frontend | Next.js, React, TypeScript, Tailwind CSS |

## Getting started

### Prerequisites

- Python 3.10+
- Node.js compatible with the version in `client/package.json`
- Dataset files and trained weights placed in the locations expected by `core/config.py`
- Optional CUDA-capable GPU for accelerated training/inference

### Install Python dependencies

```bash
git clone https://github.com/kishorekrrish3/Enhanced-Skin-Lesion-Classification.git
cd Enhanced-Skin-Lesion-Classification
python -m venv .venv
```

Activate the environment, then install the project's Python requirements:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the API

From the repository root:

```bash
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

### Run the web client

In another terminal:

```bash
cd client
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Configure the frontend API base URL to match the backend setup if required by your local environment.

## Training and evaluation

The main training/evaluation utilities are in `core/`:

- `run_pipeline.py` — pipeline entry point
- `train.py` — model training
- `evaluate.py` — evaluation
- `augmentation.py` — augmentation utilities
- `losses.py` — loss functions
- `gradcam.py` — Grad-CAM visualization

Review `core/config.py` and the dataset layout before running experiments. Training results depend on dataset composition, preprocessing, random seeds, hardware, and model configuration.

## Data and model weights

Large datasets and trained weight files may not be committed to Git. Obtain data from its licensed source, verify usage restrictions, and place local assets in the configured paths. Do not upload private, identifiable, or unlicensed medical images.

## Limitations and responsible use

- Dataset performance does not guarantee generalization across skin tones, cameras, clinical settings, or populations.
- Class imbalance and label quality can affect performance.
- Report evaluation metrics on a held-out dataset and document class-wise performance before making research claims.
- Never expose an unauthenticated inference endpoint publicly without appropriate security, upload limits, and privacy controls.
- The local development CORS configuration should be restricted before production deployment.

## Contributing

Issues and pull requests are welcome. Include reproduction steps, dataset provenance, evaluation methodology, and the exact experiment configuration where relevant.
