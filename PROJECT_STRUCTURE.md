# 🗺️ Project Structure: Enhanced Skin Lesion Classification

This document provides a comprehensive map of all files and folders in the project after the structural revamp.

---

## 📂 Root Directory
Professional project files and entry points.
```text
/
├── client/           # 🎨 Next.js Frontend (Production UI)
├── server/           # ⚙️ FastAPI Backend (Clinical API)
├── core/             # 🧠 ML Model Pipeline (Brain of the project)
├── data/             # 📊 Dataset storage (local-only, ignored by Git)
├── weights/          # 💾 Production Model Weights (.pth)
├── docs/             # 📘 Manuals and Reference documentation
├── results/          # 📈 Phase 1-5 Historical Outputs
├── PROJECT_STRUCTURE.md # This Blueprint
├── README.md         # Professional Landing Page
├── manual.MD         # Deployment & GitHub Guide
├── .gitignore        # Git exclusion rules
└── requirements.txt  # Global dependencies
```

---

## 🎨 `/client` (Frontend)
Next.js 15+ repository for the clinical dashboard.
```text
/client
├── public/                 # Static assets (logos, svgs)
├── src/
│   ├── app/                # Next.js App Router
│   │   ├── dashboard/      # Metrics visualization page
│   │   │   └── page.tsx
│   │   ├── demo/           # Diagnostic console (Inference page)
│   │   │   └── page.tsx
│   │   ├── knowledge/      # Medical knowledge base
│   │   │   └── page.tsx
│   │   ├── layout.tsx      # Global shell (Navbar, Theme)
│   │   └── page.tsx        # Homepage (Hero)
│   ├── components/
│   │   ├── ui/             # Shadcn reusable UI components
│   │   └── [custom]/       # Clinical overlays
│   └── lib/
│       ├── api.ts          # FastAPI connectivity (Typed)
│       └── utils.ts        # Dynamic class management
├── package.json            # Scripts & frontend dependencies
├── tsconfig.json           # TypeScript configuration (Typed aliases)
└── next.config.mjs         # Next.js specific settings
```

---

## ⚙️ `/server` (Backend)
FastAPI implementation for clinical service.
```text
/server
├── api/
│   └── router.py           # Endpoint definitions (/metrics, /predict)
├── tests/
│   ├── run_test.py         # Backend connectivity test
│   └── test_alb.py         # Augmentation verification
├── main.py                 # Backend Entry Point
└── requirements.txt        # Backend dependencies (uvicorn, fastapi)
```

---

## 🧠 `/core` (Machine Learning)
The Machine Learning logic, pipeline, and result generation.
```text
/core
├── results/                # Living outputs from the latest training
│   ├── metrics/            # JSON performance logs
│   ├── plots/              # Comparison plots (CM, Recall)
│   ├── gradcam/             # Grad-CAM result images
│   └── models/             # Local model backups
├── augmentation.py         # Advanced Medical Augmentation logic
├── config.py               # Global architecture and path settings
├── dataset.py              # Torch Dataset and Loader implementation
├── evaluate.py             # Validation and test split logic
├── gradcam.py              # Visual Explainability (Heatmap generation)
├── models.py               # Neural Architectures (ResNet50, SimpleCNN)
├── train.py                # Heavyweight Training Pipeline (MixUp/CutMix)
└── run_pipeline.py         # Master script to run the full end-to-end flow
```

---

## 💾 `/weights` (Production Models)
Consolidated final model weights for easy distribution.
```text
/weights
├── ImprovedResNet50.pth      # Phase 3: Final Optimized Model
├── baseline_resnet50_best.pth # Phase 2: Base ResNet
├── baseline_simplecnn_best.pth # Phase 1: Baseline CNN
└── [ClassSpecific].pth       # Targeted minority class weights
```

---

## 📘 Documentation
- **manual.MD**: Deployment, Vercel/Render set up.
- **README.md**: Public-facing repository overview.
- **PROJECT_STRUCTURE.md**: Technical file map.

---
**This structure ensures clinical-grade separation of concerns and production scalability.**
