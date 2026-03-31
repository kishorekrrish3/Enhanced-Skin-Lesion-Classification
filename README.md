# 🔬 Enhanced Skin Lesion Classification

An end-to-end medical AI application for classifying skin lesions using Deep Learning (ResNet50) and explainable AI (Grad-CAM).

## 📂 Project Overview
This repository has been reorganized for a clean, production-ready experience.

```text
/
├── client/           # 🎨 Next.js Frontend (UI/UX)
├── server/           # ⚙️ FastAPI Backend (Logic & Endpoints)
├── core/             # 🧠 ML Model Inference & Pipeline Processing
├── data/             # 📊 Dataset storage (local-only, ignored by Git)
├── weights/          # 💾 Consolidated model weights (.pth)
├── docs/             # 📘 Manuals and Reference documentation
└── manual.MD         # 🚀 Deployment and GitHub setup guide
```

## 🛠️ Tech Stack
- **Frontend**: Next.js, React, TailwindCSS, Framer Motion.
- **Backend**: FastAPI (Python), Uvicorn.
- **ML/AI**: PyTorch, Torchvision, Albumentations, Grad-CAM.
- **Explainability**: Heatmap overlays for clinical decision support.

## 🚀 Quick Start (Local Development)

### 1. Start the Backend
```powershell
cd server
python -m uvicorn main:app --reload
```

### 2. Start the Frontend
```powershell
cd client
npm run dev
```

## 📘 Production Manual
For detailed instructions on how to push this project to **GitHub** and deploy to **Vercel/Render** for free, check the [manual.MD](./manual.MD) file.

---
**Disclaimer**: This is a Capstone project for educational purposes. It is NOT for clinical diagnosis. Consult a medical professional for real-world dermatological concerns.
